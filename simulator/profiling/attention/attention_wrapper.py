from math import ceil
from typing import List

import numpy as np
import torch
from sarathi.model_executor.attention import (
    AttentionBackend,
    get_attention_wrapper,
    set_attention_backend,
)

from simulator.profiling.attention.attention_input import AttentionInput
from simulator.profiling.attention.sequence_proxy import SequenceMetadataProxy
from simulator.profiling.model_config import ModelConfig
from simulator.profiling.timer_stats_store import TimerStatsStore
from simulator.profiling.utils import ProfileMethod

WARMUP_STEPS = 2
ACTIVE_STEPS = 5


class AttentionWrapper:
    def __init__(
        self,
        model_config: ModelConfig,
        num_tensor_parallel_workers: int,
        max_model_len: int,
        block_size: int,
        attention_backend: AttentionBackend,
        profile_method: ProfileMethod,
    ):
        self._profile_method = profile_method
        self.time_stats_store = TimerStatsStore(profile_method=profile_method)

        self._n_embd = model_config.embedding_dim
        self._n_q_head = model_config.num_q_heads
        self._n_kv_head = model_config.num_kv_heads
        self._num_tensor_parallel_workers = num_tensor_parallel_workers
        assert self._n_embd % self._n_q_head == 0
        self._head_dim = self._n_embd // self._n_q_head
        self._max_model_len = max_model_len
        self._block_size = block_size

        assert self._n_q_head % num_tensor_parallel_workers == 0
        self._n_worker_q_heads = self._n_q_head // num_tensor_parallel_workers
        assert self._n_kv_head % num_tensor_parallel_workers == 0
        self._n_worker_kv_heads = self._n_kv_head // num_tensor_parallel_workers

        self._dtype = torch.float16
        self._device = torch.device("cuda")

        self._attention_backend = attention_backend
        set_attention_backend(attention_backend)
        get_attention_wrapper().init(
            self._n_worker_q_heads,
            self._n_worker_kv_heads,
            self._head_dim,
            self._block_size,
            self._device,
        )
        self._blocks_per_sequence = ceil(max_model_len / self._block_size)

    def _get_input_tensors(
        self,
        attention_input: AttentionInput,
    ):
        total_num_blocks = max(
            10000, 1 + attention_input.batch_size * self._blocks_per_sequence
        )
        num_tokens_per_seq = (
            attention_input.prefill_chunk_size if attention_input.is_prefill else 1
        )
        batch_size = attention_input.batch_size
        query = torch.randn(
            batch_size * num_tokens_per_seq,
            self._n_worker_q_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        key = torch.randn(
            batch_size * num_tokens_per_seq,
            self._n_worker_kv_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        value = torch.randn(
            batch_size * num_tokens_per_seq,
            self._n_worker_kv_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        # We create (big) KV tensors every time.
        # A better solution would be to create them once and reuse them.
        kv_cache = get_attention_wrapper().get_cache_block(
            total_num_blocks, dtype=self._dtype, device=self._device
        )
        # Create SequenceMetadataProxy objects corresponding to AttentionInput
        seq_metadata_list: List[SequenceMetadataProxy] = []
        for _ in range(attention_input.batch_size):
            seq_metadata = SequenceMetadataProxy(
                is_prompt=attention_input.is_prefill,
                total_len=num_tokens_per_seq + attention_input.kv_cache_size,
                processed_len=attention_input.kv_cache_size,
                block_table=np.random.default_rng()
                .integers(low=0, high=total_num_blocks, size=self._blocks_per_sequence)
                .tolist(),
            )
            seq_metadata_list.append(seq_metadata)
        return seq_metadata_list, query, key, value, kv_cache

    @torch.inference_mode()
    def profile(
        self,
        attention_input: AttentionInput,
    ):
        # batch size is always 1 for prefill and can be different for decode
        assert attention_input.is_valid(self._max_model_len)

        try:
            seq_metadata_list, query, key, value, kv_cache = self._get_input_tensors(
                attention_input,
            )
            get_attention_wrapper().begin_forward(seq_metadata_list)

            for _ in range(WARMUP_STEPS):
                get_attention_wrapper().forward(query, key, value, kv_cache)
            torch.cuda.synchronize()

            self.time_stats_store.clear_stats()

            for _ in range(ACTIVE_STEPS):
                get_attention_wrapper().forward(query, key, value, kv_cache)
            torch.cuda.synchronize()

            get_attention_wrapper().end_forward()
        except RuntimeError as e:
            print(e)
            return None

        return {
            "time_stats": self.time_stats_store.get_stats(),
            "n_embd": self._n_embd,
            "n_q_head": self._n_q_head,
            "n_kv_head": self._n_kv_head,
            "block_size": self._block_size,
            "num_tensor_parallel_workers": self._num_tensor_parallel_workers,
            "max_model_len": self._max_model_len,
            "batch_size": attention_input.batch_size,
            "prefill_chunk_size": attention_input.prefill_chunk_size,
            "kv_cache_size": attention_input.kv_cache_size,
            "is_prefill": attention_input.is_prefill,
            "attention_backend": self._attention_backend,
            "profile_method": self._profile_method,
        }
