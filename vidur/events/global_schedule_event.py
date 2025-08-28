from typing import List

from vidur.events import BaseEvent
from vidur.logger import init_logger

logger = init_logger(__name__)
from vidur.metrics.cluster_metrics_store import ClusterMetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType


class GlobalScheduleEvent(BaseEvent):
    def __init__(self, time: float):
        super().__init__(time, EventType.GLOBAL_SCHEDULE)

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: ClusterMetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent

        logger.debug(f"Handling global schedule event at {self.time:.2f}s")
        self._replica_set = set()
        self._request_mapping = scheduler.schedule()

        if not self._request_mapping:
            logger.debug("No requests mapped to replicas")
            return []

        logger.debug(f"Mapped {len(self._request_mapping)} requests to replicas")
        for replica_id, request in self._request_mapping:
            self._replica_set.add(replica_id)
            scheduler.get_replica_scheduler(replica_id).add_request(request)
            logger.debug(f"Added request {request.id} to replica {replica_id}")

        events = [
            ReplicaScheduleEvent(self.time, replica_id)
            for replica_id in self._replica_set
        ]
        logger.debug(f"Created {len(events)} replica schedule events")
        return events

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": str(self.event_type),
            "replica_set": self._replica_set,
            "request_mapping": [
                (replica_id, request.id)
                for replica_id, request in self._request_mapping
            ],
        }
