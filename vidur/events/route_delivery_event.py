from typing import List

from vidur.entities import Request
from vidur.events import BaseEvent
from vidur.metrics import ClusterMetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType
from vidur.types.replica_id import ReplicaId


class RouteDeliveryEvent(BaseEvent):
    """A request routed at t reaches its replica at t + cluster_config.route_delay_s (router hop latency, v4).
    Only created when the delay is positive, so zero-delay runs keep their exact event sequence."""

    def __init__(self, time: float, replica_id: ReplicaId, request: Request):
        super().__init__(time, EventType.ROUTE_DELIVERY)
        self._replica_id = replica_id
        self._request = request

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: ClusterMetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent

        scheduler.get_replica_scheduler(self._replica_id).add_request(self._request)
        return [ReplicaScheduleEvent(self.time, self._replica_id)]

    def to_dict(self):
        return {"time": self.time, "event_type": str(self.event_type), "replica_id": self._replica_id,
                "request": self._request.id}
