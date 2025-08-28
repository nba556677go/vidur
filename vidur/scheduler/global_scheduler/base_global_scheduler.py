import random
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from vidur.config import SimulationConfig
from vidur.entities import Request
from vidur.entities.batch import Batch
from vidur.entities.replica import Replica
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from vidur.kv_cache.disk_kv_cache_manager import DiskKVCacheManager
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)
from vidur.scheduler.replica_scheduler.replica_scheduler_registry import (
    ReplicaSchedulerRegistry,
)
from vidur.scheduler.replica_stage_scheduler.replica_stage_scheduler import (
    ReplicaStageScheduler,
)
from vidur.types.replica_id import ReplicaId
from vidur.utils.slo_manager import SLOManager


class BaseGlobalScheduler(ABC):
    def __init__(
        self,
        config: SimulationConfig,
        replicas: Dict[ReplicaId, Replica],
    ):
        self._config = config
        self._replicas = replicas
        self._num_replicas = len(replicas)
        self._random_number_generator = random.Random(
            config.cluster_config.global_scheduler_config.seed
        )

        self._execution_time_predictor = ExecutionTimePredictorRegistry.get(
            config.execution_time_predictor_config.get_type(),
            predictor_config=config.execution_time_predictor_config,
            replica_config=config.cluster_config.replica_config,
            cache_config=config.cluster_config.cache_config,
        )
        self._replica_schedulers: Dict[ReplicaId, BaseReplicaScheduler] = {
            replica_id: ReplicaSchedulerRegistry.get_from_str(
                self._config.cluster_config.replica_scheduler_config.get_type(),
                replica_config=self._config.cluster_config.replica_config,
                replica_scheduler_config=self._config.cluster_config.replica_scheduler_config,
                request_generator_config=self._config.request_generator_config,
                cache_config=self._config.cluster_config.cache_config,
                request_queue_config=self._config.cluster_config.request_queue_config,
                replica=replica,
                execution_time_predictor=self._execution_time_predictor,
            )
            for replica_id, replica in self._replicas.items()
        }

        if self._config.cluster_config.cache_config.enable_disk_caching:
            disk_kv_cache_manager = DiskKVCacheManager(
                block_size=self._config.cluster_config.cache_config.block_size,
                num_gpu_blocks=self._config.cluster_config.cache_config.disk_num_blocks,
                enable_caching=self._config.cluster_config.cache_config.enable_prefix_caching,
                caching_hash_algo=self._config.cluster_config.cache_config.prefix_caching_hash_algo,
                num_preallocate_tokens=self._config.cluster_config.cache_config.num_preallocate_tokens,
            )
            for _, replica in self._replica_schedulers.items():
                replica.set_disk_kv_cache(disk_kv_cache_manager)

        self._request_queue: List[Request] = []
        self._slo_manager = SLOManager(self._config.slo_config)

    def sort_requests(self) -> None:
        self._request_queue.sort(key=lambda x: (x.arrived_at, x.id))

    def add_request(self, request: Request) -> None:
        # This is the first instance the request comes into contact with the system
        self._slo_manager.set_slos(request)
        self._request_queue.append(request)

    def on_batch_end(self, batch: Batch) -> None:
        pass

    def on_prefill_end(self, request: Request) -> None:
        pass

    def on_request_end(self, request: Request) -> None:
        pass

    def get_replica_scheduler(self, replica_id: ReplicaId) -> BaseReplicaScheduler:
        return self._replica_schedulers[replica_id]

    def get_replica_stage_scheduler(
        self, replica_id: ReplicaId, stage_id: int
    ) -> ReplicaStageScheduler:
        return self._replica_schedulers[replica_id].get_replica_stage_scheduler(
            stage_id
        )

    def is_empty(self) -> bool:
        return len(self._request_queue) == 0 and all(
            replica_scheduler.is_empty()
            for replica_scheduler in self._replica_schedulers.values()
        )

    @abstractmethod
    def schedule(self) -> List[Tuple[ReplicaId, Request]]:
        pass
