    def run(self) -> None:
        logger.info(f"Starting simulation with cluster: {self._cluster}")

        # Track consecutive failed injection attempts to detect true deadlock
        consecutive_failed_injections = 0
        MAX_FAILED_INJECTIONS = 3  # Allow multiple attempts before declaring deadlock

        while not self._time_limit_reached and (
            self._event_queue
            or self._request_generator.get_next_request_arrival_time() is not None
            or not self._scheduler.is_empty()
        ):
            next_event_time = self._event_queue[0]._time if self._event_queue else None
            next_request_arrival_time = (
                self._request_generator.get_next_request_arrival_time()
            )
            if (next_request_arrival_time is not None) and (
                next_event_time is None or next_request_arrival_time <= next_event_time
            ):
                self._add_event(
                    RequestArrivalEvent(
                        next_request_arrival_time,
                        self._request_generator.get_next_request(),
                    )
                )
                continue

            # If event queue is empty but scheduler has pending work, we need to
            # manually trigger events to continue processing
            if not self._event_queue:
                from vidur.events.global_schedule_event import GlobalScheduleEvent
                from vidur.events.replica_schedule_event import ReplicaScheduleEvent
                
                # Only inject if no more requests are arriving to avoid interference
                if self._request_generator.get_next_request_arrival_time() is None:
                    global_queue_size = len(self._scheduler._request_queue)
                    replica_states = {
                        rid: (len(rs._waiting_queue), len(rs._running)) 
                        for rid, rs in self._scheduler._replica_schedulers.items()
                    }
                    replicas_with_waiting = [
                        rid for rid, rs in self._scheduler._replica_schedulers.items()
                        if len(rs._waiting_queue) > 0
                    ]
                    
                    logger.info(
                        f"Event queue empty at time {self._time}. "
                        f"Global queue: {global_queue_size} requests. "
                        f"Replica states (waiting, running): {replica_states}"
                    )
                    
                    new_events = []
                    
                    if global_queue_size > 0:
                        # There are requests in the global queue - use GlobalScheduleEvent
                        logger.debug(
                            f"Injecting GlobalScheduleEvent"
                        )
                        self._add_event(GlobalScheduleEvent(self._time))
                        
                        # Process the injected event immediately
                        event = self._event_queue[0]
                        heapq.heappop(self._event_queue)
                        new_events = event.handle_event(self._scheduler, self._cluster_metric_store)
                        self._add_events(new_events)
                    elif replicas_with_waiting:
                        # Replicas have waiting work but no global queue work
                        # Inject ReplicaScheduleEvent for each replica with waiting work
                        logger.debug(
                            f"Injecting ReplicaScheduleEvent for {len(replicas_with_waiting)} replicas"
                        )
                        for replica_id in replicas_with_waiting:
                            self._add_event(ReplicaScheduleEvent(self._time, replica_id))
                        
                        # Process all injected events
                        while self._event_queue:
                            event = heapq.heappop(self._event_queue)
                            more_events = event.handle_event(self._scheduler, self._cluster_metric_store)
                            new_events.extend(more_events)
                            self._add_events(more_events)
                    else:
                        # No work anywhere - should have exited loop normally
                        logger.warning(
                            f"Event queue empty with no pending work at {self._time}. "
                            f"Breaking to avoid infinite loop."
                        )
                        break
                    
                    # Check if we made progress (created new events)
                    if new_events:
                        consecutive_failed_injections = 0
                        logger.info(
                            f"Created {len(new_events)} new events. "
                            f"Queue size: {len(self._event_queue)}"
                        )
                    else:
                        consecutive_failed_injections += 1
                        if consecutive_failed_injections >= MAX_FAILED_INJECTIONS:
                            logger.warning(
                                f"Failed to make progress after {MAX_FAILED_INJECTIONS} event injections at {self._time}. "
                                f"Scheduler reports not empty but cannot schedule work. "
                                f"Breaking to avoid infinite loop."
                            )
                            break
                    continue
                else:
                    # This shouldn't happen due to earlier check, but be defensive
                    logger.warning(
                        "Event queue empty with pending scheduler work and future arrivals"
                    )
                    break

            event = self._event_queue[0]
            heapq.heappop(self._event_queue)
            self._set_time(event._time)
            new_events = event.handle_event(self._scheduler, self._cluster_metric_store)
            self._add_events(new_events)

            if self._config.metrics_config.write_json_trace:
                self._event_trace.append(event.to_dict())

            if self._config.metrics_config.enable_chrome_trace:
                chrome_trace = event.to_chrome_trace()
                if chrome_trace:
                    self._event_chrome_trace.append(chrome_trace)

        logger.info(f"Simulation ended at: {self._time}s")
