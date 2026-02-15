#!/usr/bin/env python3
"""
Debug script to trace exactly why requests are stuck.
Run with: python3 debug_trace.py
"""

import sys
sys.path.insert(0, '/home/kamil/vidur-rl-working')

# Patch the simulator to add debugging
import vidur.simulator
original_run = vidur.simulator.Simulator.run

def debug_run(self):
    import heapq
    from vidur.events.global_schedule_event import GlobalScheduleEvent
    from vidur.events.replica_schedule_event import ReplicaScheduleEvent
    from vidur.logger import init_logger
    
    logger = init_logger(__name__)
    
    # Track original loop variables
    consecutive_failed_injections = 0
    MAX_FAILED_INJECTIONS = 3
    
    self._init_event_queue()
    import atexit
    atexit.register(self._write_output)
    
    logger.info(f"Starting DEBUG simulation with cluster: {self._cluster}")
    
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
            from vidur.events.request_arrival_event import RequestArrivalEvent
            self._add_event(
                RequestArrivalEvent(
                    next_request_arrival_time,
                    self._request_generator.get_next_request(),
                )
            )
            continue

        # Debug: Check if we're at the stall point
        if not self._event_queue and self._request_generator.get_next_request_arrival_time() is None:
            global_queue_size = len(self._scheduler._request_queue)
            replica_states = {
                rid: (len(rs._waiting_queue), len(rs._running)) 
                for rid, rs in self._scheduler._replica_schedulers.items()
            }
            
            replicas_with_waiting = [
                rid for rid, rs in self._scheduler._replica_schedulers.items()
                if len(rs._waiting_queue) > 0
            ]
            
            if replicas_with_waiting and self._time > 1500:  # Near end of trace
                logger.info(f"=== STALL POINT DEBUG at time {self._time} ===")
                logger.info(f"Global queue: {global_queue_size}, Replicas with waiting: {len(replicas_with_waiting)}")
                
                # Detailed analysis of stuck requests
                for rid in replicas_with_waiting[:3]:  # First 3 replicas
                    rs = self._scheduler._replica_schedulers[rid]
                    if len(rs._waiting_queue) > 0:
                        # Get the first request
                        req = list(rs._waiting_queue)[0]
                        num_processed = req.num_processed_tokens
                        num_total = req.num_prefill_tokens
                        remaining = num_total - num_processed
                        
                        # Check available blocks
                        free_blocks = self._scheduler._replica_schedulers[rid]._kv_cache_manager.block_pool.get_num_free_blocks()
                        total_blocks = self._scheduler._replica_schedulers[rid]._kv_cache_manager.block_pool._num_blocks
                        
                        logger.info(f"  Replica {rid}: request {req.id}, {num_processed}/{num_total} tokens, remaining={remaining}")
                        logger.info(f"    Free blocks: {free_blocks}/{total_blocks}")
                        logger.info(f"    Running: {len(rs._running)}, max_micro_batch: {rs._max_micro_batch_size}")
                        logger.info(f"    token_budget (chunk_size): {rs._config.chunk_size}")
                        
                        # Check memory allocation
                        req_blocks = rs._kv_cache_manager.req_to_blocks.get(req.id, [])
                        logger.info(f"    Current req blocks: {len(req_blocks)}")
        
        if not self._event_queue:
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
                    logger.debug(f"Injecting GlobalScheduleEvent")
                    self._add_event(GlobalScheduleEvent(self._time))
                    event = self._event_queue[0]
                    heapq.heappop(self._event_queue)
                    new_events = event.handle_event(self._scheduler, self._cluster_metric_store)
                    self._add_events(new_events)
                elif replicas_with_waiting:
                    logger.debug(f"Injecting ReplicaScheduleEvent for {len(replicas_with_waiting)} replicas")
                    for replica_id in replicas_with_waiting:
                        self._add_event(ReplicaScheduleEvent(self._time, replica_id))
                    
                    while self._event_queue:
                        event = heapq.heappop(self._event_queue)
                        more_events = event.handle_event(self._scheduler, self._cluster_metric_store)
                        new_events.extend(more_events)
                        self._add_events(more_events)
                else:
                    logger.warning(f"Event queue empty with no pending work at {self._time}")
                    break
                
                if new_events:
                    consecutive_failed_injections = 0
                    logger.info(f"Created {len(new_events)} new events")
                else:
                    consecutive_failed_injections += 1
                    if consecutive_failed_injections >= MAX_FAILED_INJECTIONS:
                        logger.warning(f"Failed to make progress after {MAX_FAILED_INJECTIONS} injections, breaking")
                        break
                continue

        event = self._event_queue[0]
        heapq.heappop(self._event_queue)
        self._set_time(event._time)
        new_events = event.handle_event(self._scheduler, self._cluster_metric_store)
        self._add_events(new_events)
        
        if self._config.metrics_config.write_json_trace:
            self._cluster_metric_store.to_json_trace()
    
    logger.info(f"Simulation ended at: {self._time}s")
    self._write_output()

# Apply the patch
vidur.simulator.Simulator.run = debug_run

# Now run the simulation
import sys
sys.argv = [
    'debug_trace.py',
    '--replica_config_device', 'a100',
    '--replica_config_model_name', 'meta-llama/Llama-2-7b-hf',
    '--replica_config_tensor_parallel_size', '1',
    '--cluster_config_num_replicas', '16',
    '--replica_scheduler_config_type', 'vllm_v1',
    '--global_scheduler_config_type', 'lop',
    '--trace_request_generator_config_time_scale_factor', '1.5',
    '--trace_request_generator_config_trace_file', '/home/kamil/vidur-rl-working/data/processed_traces/mooncake_synthetic_trace.csv',
    '--request_generator_config_type', 'trace',
    '--cache_config_enable_prefix_caching',
    '--no-metrics_config_enable_chrome_trace',
    '--no-metrics_config_write_json_trace',
    '--metrics_config_output_dir', '/tmp/vidur_debug2',
]

from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.logger import init_logger
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = init_logger(__name__)

def main():
    config = SimulationConfig.create_from_cli_args()
    
    logger.info("Starting debug simulation...")
    simulator = Simulator(config)
    simulator.run()

if __name__ == '__main__':
    main()
