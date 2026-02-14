import atexit
import heapq
import json
import zipfile
from typing import List

import wandb

from vidur.config import SimulationConfig
from vidur.entities import Cluster
from vidur.events import BaseEvent, RequestArrivalEvent
from vidur.logger import init_logger
from vidur.metrics.cluster_metrics_store import ClusterMetricsStore
from vidur.request_generator import RequestGeneratorRegistry
from vidur.scheduler import GlobalSchedulerRegistry
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.utils.json_encoder import JsonEncoder

logger = init_logger(__name__)


class Simulator:
    def __init__(self, config: SimulationConfig) -> None:
        self._config: SimulationConfig = config

        self._time = 0
        self._time_limit_reached = False
        self._time_limit = self._config.time_limit
        if not self._time_limit:
            self._time_limit = float("inf")

        self._event_queue: List[BaseEvent] = []

        self._event_trace = []
        self._event_chrome_trace = []

        self._cluster = Cluster(
            cluster_config=self._config.cluster_config,
            metrics_config=self._config.metrics_config,
        )
        self._cluster_metric_store = ClusterMetricsStore(
            simulation_config=self._config,
            replicas=self._cluster.replicas,
        )
        self._request_generator = RequestGeneratorRegistry.get(
            self._config.request_generator_config.get_type(),
            self._config.request_generator_config,
        )
        self._scheduler = GlobalSchedulerRegistry.get(
            self._config.cluster_config.global_scheduler_config.get_type(),
            self._config,
            self._cluster.replicas,
        )

        self._init_event_queue()
        atexit.register(self._write_output)

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
    def _write_output(self) -> None:
        logger.info("Writing output")

        self._cluster_metric_store.plot(self._time)
        logger.info("Metrics written")

        if self._config.metrics_config.write_json_trace:
            self._write_event_trace()
            logger.info("Json event trace written")

        if self._config.metrics_config.enable_chrome_trace:
            self._write_chrome_trace()
            logger.info("Chrome event trace written")

    def _add_event(self, event: BaseEvent) -> None:
        heapq.heappush(self._event_queue, event)

    def _add_events(self, events: List[BaseEvent]) -> None:
        for event in events:
            self._add_event(event)

    def _init_event_queue(self) -> None:
        first_request = self._request_generator.get_next_request()
        if first_request:
            self._add_event(
                RequestArrivalEvent(first_request.arrived_at, first_request)
            )

    def _set_time(self, time: float) -> None:
        self._time = time
        if self._time > self._time_limit:
            logger.info(
                f"Time limit reached: {self._time_limit}s terminating the simulation."
            )
            self._time_limit_reached = True

    def _write_event_trace(self) -> None:
        trace_file = f"{self._config.metrics_config.output_dir}/event_trace.json"
        with open(trace_file, "w") as f:
            json.dump(self._event_trace, f, cls=JsonEncoder)

    def _write_chrome_trace(self) -> None:
        trace_file = f"{self._config.metrics_config.output_dir}/chrome_trace.json"

        chrome_trace = {"traceEvents": self._event_chrome_trace}

        with open(trace_file, "w") as f:
            json.dump(chrome_trace, f, cls=JsonEncoder)

        if wandb.run:
            zip_file_path = f"{self._config.output_dir}/chrome_trace.zip"
            with zipfile.ZipFile(
                zip_file_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                zf.writestr(
                    "chrome_trace.json",
                    json.dumps(chrome_trace, cls=JsonEncoder),
                )
            wandb.save(zip_file_path, policy="now")
