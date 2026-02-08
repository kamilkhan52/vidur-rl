from vidur.scheduler.replica_scheduler.faster_transformer_replica_scheduler import (
    FasterTransformerReplicaScheduler,
)
from vidur.scheduler.replica_scheduler.orca_replica_scheduler import (
    OrcaReplicaScheduler,
)
from vidur.scheduler.replica_scheduler.sarathi_replica_scheduler import (
    SarathiReplicaScheduler,
)
from vidur.scheduler.replica_scheduler.vllm_replica_scheduler import (
    VLLMReplicaScheduler,
)
from vidur.scheduler.replica_scheduler.vllm_v1_disk_replica_scheduler import (
    VLLMV1DiskReplicaScheduler,
)
from vidur.scheduler.replica_scheduler.vllm_v1_replica_scheduler import (
    VLLMV1ReplicaScheduler,
)
from vidur.types import ReplicaSchedulerType
from vidur.utils.base_registry import BaseRegistry


class ReplicaSchedulerRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> ReplicaSchedulerType:
        scheduler_type = ReplicaSchedulerType.from_str(key_str)
        
        # Warn users if they're using the deprecated 'vllm' scheduler
        if scheduler_type == ReplicaSchedulerType.VLLM:
            raise ValueError(
                "The 'vllm' scheduler is deprecated and has known bugs. "
                "Please use 'vllm_v1' instead:\n"
                "  --replica_scheduler_config_type vllm_v1\n\n"
                "The vllm_v1 scheduler is the maintained version with bug fixes."
            )
        
        return scheduler_type


ReplicaSchedulerRegistry.register(
    ReplicaSchedulerType.FASTER_TRANSFORMER, FasterTransformerReplicaScheduler
)
ReplicaSchedulerRegistry.register(ReplicaSchedulerType.ORCA, OrcaReplicaScheduler)
ReplicaSchedulerRegistry.register(ReplicaSchedulerType.SARATHI, SarathiReplicaScheduler)
ReplicaSchedulerRegistry.register(ReplicaSchedulerType.VLLM, VLLMReplicaScheduler)
ReplicaSchedulerRegistry.register(ReplicaSchedulerType.VLLM_V1, VLLMV1ReplicaScheduler)
ReplicaSchedulerRegistry.register(
    ReplicaSchedulerType.VLLM_V1_DISK, VLLMV1DiskReplicaScheduler
)
