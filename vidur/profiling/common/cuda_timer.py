import time

import torch
from torch.profiler import record_function

from vidur.profiling.common.timer_stats_store import TimerStatsStore
from vidur.profiling.utils import ProfileMethod


def calculate_kv_cache_size(cache_size, num_heads, head_dim, batch_size=1, dtype=torch.float16):
    """Calculate KV cache memory in bytes"""
    element_size = 2 if dtype == torch.float16 else 4  # bytes per element
    
    # KV cache shape: [2, batch_size, num_heads, cache_size, head_dim]
    # 2 = key + value tensors
    total_elements = 2 * batch_size * num_heads * cache_size * head_dim
    
    return total_elements * element_size


class CudaTimer:
    def __init__(
        self,
        name,
        layer_id: int = 0,  # we don't care about layer id, it is just for compatibility with sarathi cudatimer
        aggregation_fn=sum,
        filter_str=None,
    ):
        if name:
            # beautify the names we get from vllm
            name = str(name).replace("OperationMetrics.", "")
            name = name.lower()
            self.name = f"vidur_{name}"
        else:
            self.name = None

        self.timer_stats_store = TimerStatsStore()
        self.disabled = (name is None) or self.timer_stats_store.disabled

        if self.disabled:
            return

        self.aggregation_fn = aggregation_fn
        self.filter_str = filter_str
        self.enable_memory_profiling = (name and "attn_kv_cache_save" in str(name).lower())
        self.initial_memory = None
        self.kv_cache_params = None  # Store KV cache parameters for memory calculation

        if self.timer_stats_store.profile_method == ProfileMethod.KINETO:
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                on_trace_ready=self.handle_trace,
            )
        else:
            self.profiler = None
        self.start_event = None
        self.end_event = None
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        if self.disabled:
            return

        if self.enable_memory_profiling:
            torch.cuda.synchronize()
            self.initial_memory = torch.cuda.memory_allocated()

        if self.timer_stats_store.profile_method == ProfileMethod.RECORD_FUNCTION:
            self.profiler_function_context = record_function(self.name)
            self.profiler_function_context.__enter__()
        elif self.timer_stats_store.profile_method == ProfileMethod.CUDA_EVENT:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        elif self.timer_stats_store.profile_method == ProfileMethod.KINETO:
            self.profiler.__enter__()
        elif self.timer_stats_store.profile_method == ProfileMethod.PERF_COUNTER:
            torch.cuda.synchronize()
            self.start_time = time.perf_counter()
        else:
            raise ValueError(
                f"Unknown profile method {self.timer_stats_store.profile_method}"
            )
        return self
    
    def set_kv_cache_params(self, cache_size, num_heads, head_dim, batch_size=1, dtype=torch.float16):
        """Set KV cache parameters for accurate memory calculation"""
        if self.enable_memory_profiling:
            self.kv_cache_params = {
                'cache_size': cache_size,
                'num_heads': num_heads,
                'head_dim': head_dim,
                'batch_size': batch_size,
                'dtype': dtype
            }

    def handle_trace(self, trace):
        events = trace.events()

        if self.filter_str:
            events = [e for e in events if e.name.startswith(self.filter_str)]

        total_cuda_time = self.aggregation_fn([e.cuda_time_total for e in events])
        self.timer_stats_store.record_time(
            self.name, total_cuda_time * 1e-3
        )  # convert to ms

    def __exit__(self, *args):
        if self.disabled:
            return

        if self.enable_memory_profiling:
            if self.kv_cache_params:
                # Calculate actual KV cache memory instead of measuring delta
                kv_cache_memory = calculate_kv_cache_size(
                    cache_size=self.kv_cache_params['cache_size'],
                    num_heads=self.kv_cache_params['num_heads'], 
                    head_dim=self.kv_cache_params['head_dim'],
                    batch_size=self.kv_cache_params.get('batch_size', 1),
                    dtype=self.kv_cache_params.get('dtype', torch.float16)
                )
                self.timer_stats_store.record_time(
                    f"{self.name}_memory_mb", kv_cache_memory / (1024 * 1024)
                )
            else:
                # Fallback to original delta measurement
                torch.cuda.synchronize()
                final_memory = torch.cuda.memory_allocated()
                memory_delta = (final_memory - self.initial_memory) / (1024 * 1024)  # MB
                self.timer_stats_store.record_time(
                    f"{self.name}_memory_mb", memory_delta
                )

        if self.timer_stats_store.profile_method == ProfileMethod.RECORD_FUNCTION:
            self.profiler_function_context.__exit__(*args)
        elif self.timer_stats_store.profile_method == ProfileMethod.CUDA_EVENT:
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.end_event.record()
            self.timer_stats_store.record_time(
                self.name, [self.start_event, self.end_event]
            )
        elif self.timer_stats_store.profile_method == ProfileMethod.KINETO:
            self.profiler.__exit__(*args)
        elif self.timer_stats_store.profile_method == ProfileMethod.PERF_COUNTER:
            torch.cuda.synchronize()
            self.end_time = time.perf_counter()
            self.timer_stats_store.record_time(
                self.name, (self.end_time - self.start_time) * 1e3
            )  # convert to ms
        else:
            raise ValueError(
                f"Unknown profile method {self.timer_stats_store.profile_method}"
            )