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
from vidur.metrics.queue_diagnostics import configure_queue_diagnostics
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

        # Optional queue diagnostics telemetry (enabled via ENABLE_QUEUE_DIAGNOSTICS=1)
        configure_queue_diagnostics(self._config)

        self._validate_request_capacity()
        self._validate_block_size()

        self._init_event_queue()
        # Outputs are written by an explicit write_output() call (vidur.main), not at interpreter exit: an atexit
        # writer that fails still lets the process exit 0, and a crash would leave partial outputs behind.
        self._output_written = False
        self.stuck_requests: List[dict] = []

    def run(self) -> None:
        logger.info(f"Starting simulation with cluster: {self._cluster}")

        while not self._time_limit_reached and (
            self._event_queue
            or self._request_generator.get_next_request_arrival_time() is not None
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

        if not (self._scheduler.is_empty() or self._time_limit_reached):
            # Count pending requests in global queue and replica queues
            global_queue_count = len(self._scheduler._request_queue)
            replica_waiting = 0
            replica_running = 0
            for rs in self._scheduler._replica_schedulers.values():
                replica_waiting += len(rs._waiting_queue)
                replica_running += len(rs._running)
            
            pending_total = global_queue_count + replica_waiting + replica_running
            self.stuck_requests = self._collect_stuck_requests()

            logger.warning(
                f"Simulation ended with pending requests in scheduler. "
                f"Scheduler empty: {self._scheduler.is_empty()}, "
                f"Time limit reached: {self._time_limit_reached}. "
                f"Pending: {pending_total} total "
                f"(global_queue={global_queue_count}, waiting={replica_waiting}, running={replica_running})"
            )

        logger.info(f"Simulation ended at: {self._time}s")

    def _validate_request_capacity(self) -> None:
        """
        Refuse to start a trace replay that contains a request larger than a
        replica's whole KV cache. Such a request can never be scheduled; under
        FCFS it wedges its replica and every later request routed there is
        silently lost, corrupting completion counts and latency statistics.
        """
        get_max = getattr(self._request_generator, "max_request_total_tokens", None)
        if get_max is None:
            return
        capacities = [
            rs.max_request_total_tokens
            for rs in self._scheduler._replica_schedulers.values()
            if hasattr(rs, "max_request_total_tokens")
        ]
        if not capacities:
            return
        capacity = min(capacities)
        largest = get_max()
        if largest > capacity:
            raise ValueError(
                f"Trace contains a request of {largest} tokens (prefill+decode) but a replica's KV cache "
                f"holds at most {capacity} tokens; it could never be scheduled and would wedge its replica. "
                f"Pass --trace_request_generator_config_max_tokens {capacity} (or smaller) to filter the trace."
            )

    def _collect_stuck_requests(self) -> List[dict]:
        out = [{"where": "global_queue", "request_id": r.id} for r in self._scheduler._request_queue]
        for rid, rs in self._scheduler._replica_schedulers.items():
            out += [{"where": "waiting", "replica": rid, "request_id": r.id} for r in rs._waiting_queue]
            out += [{"where": "running", "replica": rid, "request_id": r.id} for r in rs._running]
        return out

    def _validate_block_size(self) -> None:
        """Trace hash ids are only comparable with cache blocks of the same size (changelog KV-10): a trace hashed
        at 32 tokens replayed on 16-token blocks would match prefixes wrongly and silently."""
        sizes = {int(b) for b in getattr(self._request_generator, "_block_size", []) if b == b and b is not None}
        cache_block = self._config.cluster_config.cache_config.block_size
        if sizes and sizes != {cache_block}:
            raise ValueError(f"Trace block_size {sorted(sizes)} does not match the KV cache block size {cache_block}.")

    def write_output(self) -> None:
        """Write metrics and end-state files once. Raises on failure (the caller exits non-zero)."""
        if self._output_written:
            return
        self._output_written = True
        self._write_output()
        out_dir = self._config.metrics_config.output_dir
        predictor = getattr(self._scheduler, "_execution_time_predictor", None)
        info = getattr(predictor, "provenance", None)
        if callable(info):
            with open(f"{out_dir}/predictor_info.json", "w") as f:
                json.dump(info(), f, indent=1, sort_keys=True)
        if self.stuck_requests:
            with open(f"{out_dir}/stuck_requests.json", "w") as f:
                json.dump(self.stuck_requests, f)

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
