from typing import List

from vidur.entities.batch import Batch
from vidur.events import BaseEvent
from vidur.logger import init_logger

logger = init_logger(__name__)
from vidur.metrics import ClusterMetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.scheduler.replica_scheduler.replica_scheduler_output import (
    ReplicaSchedulerOutput,
)
from vidur.types import EventType
from vidur.types.replica_id import ReplicaId


class ReplicaScheduleEvent(BaseEvent):
    def __init__(self, time: float, replica_id: ReplicaId):
        super().__init__(time, EventType.REPLICA_SCHEDULE)

        self._replica_id = replica_id

        self._batches: List[Batch] = []

    def handle_event(
        self, global_scheduler: BaseGlobalScheduler, metrics_store: ClusterMetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.batch_stage_arrival_event import BatchStageArrivalEvent

        logger.debug(f"Handling replica schedule event for replica {self._replica_id} at {self.time:.2f}s")
        
        # Attempt to create batches
        replica_scheduler = global_scheduler.get_replica_scheduler(self._replica_id)
        replica_scheduler_outputs: List[ReplicaSchedulerOutput] = []
        while replica_scheduler.can_schedule():
            replica_scheduler_output = replica_scheduler.on_schedule(self.time)
            replica_scheduler_outputs.append(replica_scheduler_output)
            if not replica_scheduler_output.batch:
                logger.debug("No more batches can be created")
                break

        # Process the batches formed, schedule next events and update metrics
        self._batches = [
            replica_scheduler_output.batch
            for replica_scheduler_output in replica_scheduler_outputs
            if replica_scheduler_output.batch
        ]
        self._requeued_requests = set(
            request
            for replica_scheduler_output in replica_scheduler_outputs
            for request in replica_scheduler_output.requeued_requests
        )

        logger.debug(f"Created {len(self._batches)} batches")
        if self._requeued_requests:
            logger.debug(f"Requeued {len(self._requeued_requests)} requests")

        for batch in self._batches:
            batch.on_schedule(self.time)
            logger.debug(f"Scheduled batch {batch.id} with {len(batch.requests)} requests")

        memory_usage_percent = replica_scheduler.memory_usage_percent
        metrics_store.on_replica_schedule(
            self.time, self._replica_id, self._batches, memory_usage_percent
        )
        logger.debug(f"Memory usage for replica {self._replica_id}: {memory_usage_percent:.1f}%")

        next_events = []
        if self._requeued_requests:
            next_events.append(ReplicaScheduleEvent(self.time, self._replica_id))
            logger.debug("Created new replica schedule event for requeued requests")

        batch_events = [
            BatchStageArrivalEvent(
                self.time,
                self._replica_id,
                0,  # stage_id
                batch,
            )
            for batch in self._batches
        ]
        if batch_events:
            logger.debug(f"Created {len(batch_events)} batch stage arrival events")
        
        return next_events + batch_events

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": str(self.event_type),
            "replica_id": self._replica_id,
            "batch_ids": [batch.id for batch in self._batches],
            "requeued_request_ids": [request.id for request in self._requeued_requests],
        }
