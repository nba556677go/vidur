from typing import List, Tuple

from vidur.entities import Request
from vidur.logger import init_logger
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler
from vidur.types.replica_id import ReplicaId


logger = init_logger(__name__, "debug")


class RoundRobinGlobalScheduler(BaseGlobalScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._request_counter = 0
        self._replica_id_list = sorted(self._replicas.keys())
        logger.debug(f"Initialized RoundRobinGlobalScheduler with {len(self._replica_id_list)} replicas")

    def schedule(self) -> List[Tuple[ReplicaId, Request]]:
        self.sort_requests()
        
        logger.debug(f"Scheduling {len(self._request_queue)} requests")
        request_mapping = []
        
        while self._request_queue:
            request = self._request_queue.pop(0)
            replica_id = self._replica_id_list[
                self._request_counter % self._num_replicas
            ]
            self._request_counter += 1
            request_mapping.append((replica_id, request))
            logger.debug(f"Assigned request {request.id} to replica {replica_id} "
                        f"(prefill={request.num_prefill_tokens}, decode={request.num_decode_tokens})")

        return request_mapping
