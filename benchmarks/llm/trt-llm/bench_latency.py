#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorRT-LLM benchmark script with feature parity to your vLLM bench:
- Input modes: prompts file / trace CSV / QPS Poisson arrivals
- Metrics: TTFT, inter-token, E2E, scheduling delay, percentiles, overall TPS
- Parallelism: TP/PP (DP is not used in single-process LLM API)
- Warmup, concurrency, JSON results, NVTX ranges (optional)

Notes:
* TRT-LLM's "max_num_batched_tokens" analogue is 'max_num_tokens' set at build time.
* At runtime you can set max_batch_size, max_seq_len, and KV cache policy.
"""

import asyncio
import logging
import time
import uuid
import numpy as np
import torch
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class TRTLLMBenchmark:
    def __init__(self,
                 model_or_path: str,
                 args,
                 nsys: bool = False,
                 tensor_parallel_size: int = 1,
                 pipeline_parallel_size: int = 1,
                 data_parallel_size: int = 1,
                 max_batch_size: Optional[int] = None,
                 max_seq_len: Optional[int] = None,
                 kv_cache_fraction: float = 0.85,
                 tokenizer_dir: Optional[str] = None):
        self.args = args
        self.model_or_path = model_or_path
        self.results: List[Dict[str, Any]] = []
        self.nsys = nsys
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.data_parallel_size = data_parallel_size  # kept for CLI parity (unused in single process)
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.kv_cache_fraction = kv_cache_fraction
        self.tokenizer_dir = tokenizer_dir

        # init later in initialize_engine()
        self.llm: Optional[LLM] = None
        self.tokenizer = None

    async def initialize_engine(self):
        logger.info(f"Initializing TensorRT-LLM LLM (tp={self.tensor_parallel_size}, "
                    f"pp={self.pipeline_parallel_size})...")
        kv_cfg = KvCacheConfig(
            free_gpu_memory_fraction=self.kv_cache_fraction,
            enable_block_reuse=True
        )
        # Build LLM with runtime config knobs
        self.llm = LLM(
            model=self.model_or_path,                  # HF ID or local HF folder; for engine_dir use runtime API instead
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
            kv_cache_config=kv_cfg
        )
        # Prefer LLM's tokenizer; fall back to HF if user passed custom tokenizer_dir
        if getattr(self.llm, "tokenizer", None) is not None and self.tokenizer_dir is None:
            self.tokenizer = self.llm.tokenizer
        else:
            tok_src = self.tokenizer_dir if self.tokenizer_dir else self.model_or_path
            self.tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
        logger.info("LLM initialized.")

    async def run_warmup(self, warmup_requests: int = 5):
        self._warmup_requests = warmup_requests
        if warmup_requests <= 0:
            logger.info("Skipping warmup.")
            return
        logger.info(f"Warmup: {warmup_requests} requests...")
        prompt = "Warmup request to prepare the TensorRT-LLM engine."
        params = SamplingParams(max_tokens=32, temperature=0.0)
        # Use async streaming to hit both prefill+decode paths
        tasks = []
        for _ in range(warmup_requests):
            tasks.append(asyncio.create_task(self._consume_stream(prompt, params, record_results=False)))
        await asyncio.gather(*tasks)
        logger.info("Warmup done.")

    async def _consume_stream(self, prompt: str, sampling_params: SamplingParams,
                              prompt_info: Optional[Dict[str, Any]] = None,
                              record_results: bool = True):
        """Send one request via streaming; capture TTFT/inter-token timing."""
        request_id = str(uuid.uuid4())
        arrival_time = (prompt_info or {}).get("arrival_time", time.time())
        input_tokens = (prompt_info or {}).get("input_tokens",
                                               len(self.tokenizer.encode(prompt)))

        # schedule delay = when we actually start compute - when we planned to arrive
        processing_start_time = time.time()
        schedule_delay = processing_start_time - arrival_time

        first_token_time = None
        tick_times: List[float] = []

        # NVTX: prefill range
        if self.nsys and torch.cuda.is_available():
            torch.cuda.nvtx.range_push("trtllm_prefill")

        # async streaming
        stream = self.llm.generate_async(prompt, sampling_params, streaming=True)

        final_output = None
        async for output in stream:
            now = time.time()
            if first_token_time is None:
                first_token_time = now
                if self.nsys and torch.cuda.is_available():
                    torch.cuda.nvtx.range_pop()            # end prefill
                    torch.cuda.nvtx.range_push("trtllm_decode")
            tick_times.append(now)
            final_output = output

        if self.nsys and torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()

        # Handle failures
        if final_output is None:
            logger.error(f"Request {request_id} produced no output.")
            return

        end_time = time.time()

        # Try to get token_ids from TRT-LLM output; fallback to re-tokenizing text.
        try:
            output_tokens = len(final_output.outputs[0].token_ids)
        except Exception:
            try:
                output_tokens = len(self.tokenizer.encode(final_output.outputs[0].text))
            except Exception:
                output_tokens = 0

        total_latency = end_time - processing_start_time
        e2e_latency = end_time - arrival_time
        ttft = (first_token_time - processing_start_time) if first_token_time else 0.0
        inter_token_latencies = [tick_times[i] - tick_times[i-1] for i in range(1, len(tick_times))]
        avg_itl = float(np.mean(inter_token_latencies)) if inter_token_latencies else 0.0

        if record_results:
            self.results.append({
                "request_id": request_id,
                "prompt": prompt if not (prompt_info or {}).get("is_trace") else f"Trace-generated prompt ({input_tokens} tokens)",
                "output": final_output.outputs[0].text,
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "total_tokens": int(input_tokens + output_tokens),
                "arrival_time": arrival_time,
                "processing_start_time": processing_start_time,
                "schedule_delay": schedule_delay,
                "total_latency": total_latency,
                "e2e_latency": e2e_latency,
                "time_to_first_token": ttft,
                "avg_inter_token_latency": avg_itl,
                "token_latencies": inter_token_latencies,
            })

    def _parse_trace_file(self, trace_path: str) -> List[Dict[str, Any]]:
        logger.info(f"Parsing trace CSV: {trace_path}")
        df = pd.read_csv(trace_path)
        req_cols = {'request_arrived_at', 'request_num_prefill_tokens', 'request_num_decode_tokens'}
        df.columns = df.columns.str.strip()
        if not req_cols.issubset(df.columns):
            raise ValueError(f"Trace file must contain columns: {req_cols}. Found: {list(df.columns)}")
        requests = []
        for _, row in df.iterrows():
            num_prefill = int(row['request_num_prefill_tokens'])
            num_decode = int(row['request_num_decode_tokens'])
            arrival = float(row['request_arrived_at'])
            # craft a dummy prompt with desired prefill length
            prompt_ids = [self.tokenizer.bos_token_id] * max(1, num_prefill)
            prompt_text = self.tokenizer.decode(prompt_ids)
            params = SamplingParams(max_tokens=num_decode)
            requests.append({
                "prompt": prompt_text,
                "params": params,
                "arrived_at": arrival,
                "arrival_time": arrival,
                "input_tokens": num_prefill,
                "is_trace": True
            })
        logger.info(f"Trace requests: {len(requests)}")
        return requests

    def _generate_poisson_arrivals(self, qps: float, n: int) -> List[float]:
        mean = 1.0 / qps
        inter = np.random.exponential(mean, max(0, n - 1))
        t, out = 0.0, [0.0]
        for d in inter:
            t += d
            out.append(t)
        logger.info(f"Poisson window ~{out[-1]:.2f}s; realized QPS ~{n / max(out[-1], 1e-6):.2f}")
        return out

    async def run_benchmark(self,
                            prompts: List[str] = None,
                            trace_path: str = None,
                            gen_params: Dict = None,
                            concurrency: int = 1,
                            qps_mode: bool = False,
                            qps: float = 1.0,
                            num_requests: int = None,
                            duration: float = None):
        if self.llm is None:
            await self.initialize_engine()

        self._benchmark_start_mono = time.monotonic()
        tasks = []

        if qps_mode:
            if not prompts:
                raise ValueError("Prompts must be provided for QPS mode.")
            if duration:
                total = int(qps * duration)
            elif num_requests:
                total = num_requests
            else:
                total = len(prompts)

            arrival_times = self._generate_poisson_arrivals(qps, total)
            params = SamplingParams(
                temperature=gen_params.get("temperature", 0.7),
                top_p=gen_params.get("top_p", 0.9),
                max_tokens=gen_params.get("max_tokens", 1024),
                min_tokens=gen_params.get("min_tokens", 512)
            )

            for i in range(total):
                prompt = prompts[i % len(prompts)]
                target = arrival_times[i]

                async def send(pmt, at, idx):
                    target_mono = self._benchmark_start_mono + at
                    to_sleep = max(0, target_mono - time.monotonic())
                    if to_sleep > 0:
                        await asyncio.sleep(to_sleep)
                    info = {"arrival_time": time.time(), "request_idx": idx}
                    await self._consume_stream(pmt, params, info, record_results=True)

                tasks.append(asyncio.create_task(send(prompt, target, i)))
                if duration and target > duration:
                    break

            await asyncio.gather(*tasks)

        elif trace_path:
            requests = self._parse_trace_file(trace_path)
            for req in requests:
                target_mono = self._benchmark_start_mono + req['arrived_at']
                to_sleep = max(0, target_mono - time.monotonic())
                if to_sleep > 0:
                    await asyncio.sleep(to_sleep)
                req['arrival_time'] = time.time()
                tasks.append(asyncio.create_task(self._consume_stream(req['prompt'], req['params'], req, True)))
            if tasks:
                await asyncio.gather(*tasks)

        elif prompts:
            params = SamplingParams(
                temperature=gen_params.get("temperature", 0.7),
                top_p=gen_params.get("top_p", 0.9),
                max_tokens=gen_params.get("max_tokens", 256)
            )
            for prompt in prompts:
                info = {"arrival_time": time.time()}
                tasks.append(asyncio.create_task(self._consume_stream(prompt, params, info, True)))
                if len(tasks) >= concurrency:
                    await asyncio.gather(*tasks)
                    tasks = []
            if tasks:
                await asyncio.gather(*tasks)
        else:
            raise ValueError("Either --prompts-file, --trace, or --qps-mode must be provided.")

    def _calculate_stats(self) -> Dict[str, Any]:
        if not self.results:
            return {}
        as_ms = lambda xs: [x * 1000.0 for x in xs]
        total_ms = as_ms([r["total_latency"] for r in self.results])
        ttft_ms = as_ms([r["time_to_first_token"] for r in self.results])
        e2e_ms = as_ms([r["e2e_latency"] for r in self.results])
        sched_ms = as_ms([r["schedule_delay"] for r in self.results])
        itl_ms = as_ms([lat for r in self.results for lat in r["token_latencies"]])

        def pct(v, p): return float(np.percentile(v, p)) if v else 0.0
        def stats(v, prefix):
            if not v: return {}
            return {
                f"{prefix}avg": float(np.mean(v)), f"{prefix}p50": pct(v, 50),
                f"{prefix}p90": pct(v, 90), f"{prefix}p95": pct(v, 95),
                f"{prefix}p99": pct(v, 99), f"{prefix}max": float(np.max(v))
            }

        latency = {
            "total_generation_latency": stats(total_ms, "total_"),
            "time_to_first_token":      stats(ttft_ms,  "first_token_"),
            "inter_token_latency":      stats(itl_ms,   "inter_token_"),
            "schedule_delay":           stats(sched_ms, "schedule_delay_"),
            "e2e_latency":              stats(e2e_ms,   "e2e_"),
        }

        total_in = sum(r["input_tokens"] for r in self.results)
        total_out = sum(r["output_tokens"] for r in self.results)
        wall = time.monotonic() - getattr(self, "_benchmark_start_mono", time.monotonic())
        tps = total_out / wall if wall > 0 else 0.0

        return {
            "summary": {
                "total_requests": len(self.results),
                "total_input_tokens": int(total_in),
                "total_output_tokens": int(total_out),
                "overall_throughput_tokens_per_sec": float(tps)
            },
            "latency_percentiles_ms": latency
        }

    def print_results(self):
        if not self.results:
            logger.warning("No results to show.")
            return
        stats = self._calculate_stats()
        s = stats["summary"]; L = stats["latency_percentiles_ms"]
        print("\n=== Benchmark Results (TensorRT-LLM) ===")
        print(f"Total requests: {s['total_requests']}")
        print(f"Total input tokens: {s['total_input_tokens']}")
        print(f"Total output tokens: {s['total_output_tokens']}")
        print(f"Overall Throughput: {s['overall_throughput_tokens_per_sec']:.2f} tokens/sec")
        def p(name, k): 
            d=L[name]; 
            print(f"\n{name.replace('_',' ').title()}:")
            print(f"  Avg: {d.get(k+'avg',0):.2f}  P50: {d.get(k+'p50',0):.2f}  "
                  f"P90: {d.get(k+'p90',0):.2f}  P95: {d.get(k+'p95',0):.2f}  "
                  f"P99: {d.get(k+'p99',0):.2f}  Max: {d.get(k+'max',0):.2f}")
        p("total_generation_latency","total_")
        p("time_to_first_token","first_token_")
        p("inter_token_latency","inter_token_")
        p("schedule_delay","schedule_delay_")
        p("e2e_latency","e2e_")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"\nGPU Memory Usage: {((total-free)/1024**3):.2f}GB / {total/1024**3:.2f}GB")

    def save_results_to_json(self, path: Path):
        if not self.results:
            logger.warning("No results to save.")
            return
        stats = self._calculate_stats()
        payload = {
            "benchmark_config": {
                "model_or_path": self.model_or_path,
                "tensor_parallel_size": self.tensor_parallel_size,
                "pipeline_parallel_size": self.pipeline_parallel_size,
                "data_parallel_size": self.data_parallel_size,
                "max_batch_size": self.max_batch_size,
                "max_seq_len": self.max_seq_len,
                "kv_cache_fraction": self.kv_cache_fraction,
                "warmup_requests": getattr(self, "_warmup_requests", 0),
            },
            "benchmark_stats": stats,
            "individual_requests": self.results
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=4)
        logger.info(f"Saved results to {path}")

async def main():
    ap = argparse.ArgumentParser("TensorRT-LLM Benchmarking Tool (vLLM-parity)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompts-file", type=str, help="Path to prompts, one per line.")
    src.add_argument("--trace", type=str, help="CSV trace with request_arrived_at, request_num_prefill_tokens, request_num_decode_tokens.")
    src.add_argument("--qps-mode", action="store_true", help="Enable QPS-mode with Poisson arrivals.")

    # Model / Runtime
    ap.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    help="HF model ID or local HF folder (LLM API).")
    ap.add_argument("--engine-dir", type=str, default=None,
                    help="(Optional) Prebuilt engine dir. NOTE: this script uses LLM API; engine_dir is not consumed here.")
    ap.add_argument("--tokenizer-dir", type=str, default=None,
                    help="Optional tokenizer directory (useful if loading from local/optimized weights).")

    # Parallelism (DP unused in single-process LLM API)
    ap.add_argument("--tp", "--tensor-parallel-size", dest="tensor_parallel_size", type=int, default=1)
    ap.add_argument("--pp", "--pipeline-parallel-size", dest="pipeline_parallel_size", type=int, default=1)
    ap.add_argument("--dp", "--data-parallel-size", dest="data_parallel_size", type=int, default=1)

    # Runtime capacity knobs (runtime; build-time max_num_tokens is separate)
    ap.add_argument("--max-batch-size", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--kv-cache-fraction", type=float, default=0.85, help="Fraction of free GPU mem allocated for KV cache.")

    # Concurrency / warmup
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--warmup-requests", type=int, default=5)

    # QPS-mode args
    ap.add_argument("--qps", type=float, default=1.0)
    ap.add_argument("--qps-prompts-file", type=str)
    ap.add_argument("--qu-prompts-file", type=str, help="CSV with 'input' column for QPS queries.")
    ap.add_argument("--num-requests", type=int)
    ap.add_argument("--duration", type=float)

    # Generation params (for prompts/QPS mode)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)

    # Profiling / output
    ap.add_argument("--nsys", action="store_true")
    ap.add_argument("--output-dir", type=str, default="./benchmark_runs_trtllm")

    args = ap.parse_args()

    # Output dirs & logging
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir_name = (args.model or "engine").replace("/", "_")
    run_dir = Path(args.output_dir) / model_dir_name / f"run_{ts}"
    log_path = run_dir / "benchmark.log"
    config_path = run_dir / "config.json"
    results_path = run_dir / "benchmark_results.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s [%(name)s:%(levelname)s] %(message)s")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path); fh.setFormatter(formatter); logger.addHandler(fh)
    ch = logging.StreamHandler(); ch.setFormatter(formatter); logger.addHandler(ch)

    # Save config
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"Config saved to {config_path}")

    bench = TRTLLMBenchmark(
        model_or_path=args.model if args.engine_dir is None else args.model,
        args=args,
        nsys=args.nsys,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        data_parallel_size=args.data_parallel_size,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        kv_cache_fraction=args.kv_cache_fraction,
        tokenizer_dir=args.tokenizer_dir
    )

    try:
        await bench.initialize_engine()
        await bench.run_warmup(args.warmup_requests)

        if args.trace:
            await bench.run_benchmark(trace_path=args.trace)
        elif args.qps_mode:
            if not args.qps_prompts_file and not args.qu_prompts_file:
                logger.error("Provide --qps-prompts-file or --qu-prompts-file for QPS mode.")
                return
            if args.qps_prompts_file and args.qu_prompts_file:
                logger.error("Choose either --qps-prompts-file or --qu-prompts-file, not both.")
                return
            if args.qu_prompts_file:
                df = pd.read_csv(args.qu_prompts_file)
                prompts = df['input'].dropna().astype(str).tolist()
            else:
                with open(args.qps_prompts_file, "r", encoding="utf-8") as f:
                    prompts = [ln.strip() for ln in f if ln.strip()]
            gen = {"temperature": args.temperature, "top_p": args.top_p,
                   "max_tokens": args.max_tokens, "min_tokens": int(0.7 * int(args.max_tokens))}
            await bench.run_benchmark(prompts=prompts, gen_params=gen, qps_mode=True,
                                      qps=args.qps, num_requests=args.num_requests, duration=args.duration)
        else:
            with open(args.prompts_file, "r", encoding="utf-8") as f:
                prompts = [ln.strip() for ln in f if ln.strip()]
            gen = {"temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens}
            await bench.run_benchmark(prompts=prompts, gen_params=gen, concurrency=args.concurrency)

        bench.print_results()
        bench.save_results_to_json(results_path)

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
