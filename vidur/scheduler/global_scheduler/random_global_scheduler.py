from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.types.replica_id import ReplicaId


class RandomGlobalScheduler(BaseGlobalScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._replica_id_list = sorted(self._replicas.keys())

    def schedule(self) -> List[Tuple[ReplicaId, Request]]:
        self.sort_requests()

        request_mapping = []
        while self._request_queue:
            request = self._request_queue.pop(0)
            # Use the scheduler's own seeded RNG (like LOR/STICKY_LOR) rather than
            # Python's global `random`, so --random_global_scheduler_config_seed
            # controls this policy and repeats can be varied deliberately.
            replica_id = self._replica_id_list[
                self._random_number_generator.randint(0, self._num_replicas - 1)
            ]
            request_mapping.append((replica_id, request))
        return request_mapping
