from typing import List, Tuple

from vidur.entities import Request
from vidur.events import BaseEvent
from vidur.metrics import ClusterMetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType
from vidur.types.replica_id import ReplicaId


class RouteDeliveryEvent(BaseEvent):
    """Router hop latency (v4, cluster_config.route_delay_s = d > 0).

    Semantics: the router decides at t on the state it sees at t; the whole mapping of that routing call reaches the
    replicas together at t + d. Delivery replays the zero-delay path at t + d: it adds every request to its replica,
    then returns one ReplicaScheduleEvent per distinct replica, and it sorts among same-time events exactly where the
    zero-delay GlobalScheduleEvent would (after arrivals and batch/request ends at t + d, before replica scheduling).
    So a same-instant burst still batches together, and each request reaches its replica exactly d after arrival;
    at light load (replica state unchanged over the hop) every timestamp is the zero-delay one shifted by d. Under
    load the outcome differs beyond the shift, because the router acted on state that is d old.

    Only created when the delay is positive, so zero-delay runs keep their exact event sequence.
    """

    def __init__(self, time: float, mapping: List[Tuple[ReplicaId, Request]]):
        super().__init__(time, EventType.ROUTE_DELIVERY)
        self._mapping = list(mapping)
        self._replica_set = set()

    def _get_priority_number(self):
        # Same slot as a GlobalScheduleEvent at this time (the event type label stays ROUTE_DELIVERY).
        return (self._time, EventType.GLOBAL_SCHEDULE, self._id)

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: ClusterMetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent

        for replica_id, request in self._mapping:
            self._replica_set.add(replica_id)
            scheduler.get_replica_scheduler(replica_id).add_request(request)
        return [ReplicaScheduleEvent(self.time, replica_id) for replica_id in self._replica_set]

    def to_dict(self):
        return {"time": self.time, "event_type": str(self.event_type), "replica_set": self._replica_set,
                "request_mapping": [(replica_id, request.id) for replica_id, request in self._mapping]}
