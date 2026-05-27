from dataclasses import dataclass
from collections import deque
from typing import Optional
import random
import statistics
import math


@dataclass
class Request:
    id: int
    arrival_time: int
    prompt_tokens: int
    output_tokens: int
    priority: int

    prefill_remaining_steps: int
    decode_remaining_tokens: int
    generated_tokens: int = 0

    allocated_kv_blocks: int = 0
    start_time: Optional[int] = None
    first_token_time: Optional[int] = None
    finish_time: Optional[int] = None


class KVCacheBlockManager:
    def __init__(self, total_blocks: int, block_size_tokens: int):
        self.total_blocks = total_blocks
        self.block_size_tokens = block_size_tokens
        self.free_blocks = total_blocks
        self.allocations: dict[int, int] = {}

    def blocks_needed(self, tokens: int) -> int:
        return max(1, math.ceil(tokens / self.block_size_tokens))

    def can_allocate_blocks(self, blocks: int) -> bool:
        return self.free_blocks >= blocks

    def allocate_initial_prompt(self, request: Request) -> bool:
        blocks = self.blocks_needed(request.prompt_tokens)

        if not self.can_allocate_blocks(blocks):
            return False

        self.free_blocks -= blocks
        self.allocations[request.id] = blocks
        request.allocated_kv_blocks = blocks
        return True

    def maybe_grow_for_decode(self, request: Request) -> bool:
        total_tokens_now = request.prompt_tokens + request.generated_tokens
        required_blocks = self.blocks_needed(total_tokens_now)

        if required_blocks <= request.allocated_kv_blocks:
            return True

        additional_blocks = required_blocks - request.allocated_kv_blocks

        if not self.can_allocate_blocks(additional_blocks):
            return False

        self.free_blocks -= additional_blocks
        self.allocations[request.id] = required_blocks
        request.allocated_kv_blocks = required_blocks
        return True

    def release(self, request: Request):
        blocks = self.allocations.pop(request.id, 0)
        self.free_blocks += blocks
        request.allocated_kv_blocks = 0

    @property
    def used_blocks(self) -> int:
        return self.total_blocks - self.free_blocks

    @property
    def utilization(self) -> float:
        return self.used_blocks / self.total_blocks


def generate_requests(n=300, seed=42):
    random.seed(seed)
    requests = []
    time = 0

    for i in range(n):
        time += random.randint(1, 5)

        prompt_tokens = random.choice([64, 128, 512, 1024, 2048, 4096])
        output_tokens = random.choice([32, 64, 128, 256])
        prefill_steps = max(1, prompt_tokens // 256)

        requests.append(
            Request(
                id=i,
                arrival_time=time,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                priority=random.choice([1, 2, 3]),
                prefill_remaining_steps=prefill_steps,
                decode_remaining_tokens=output_tokens,
            )
        )

    return requests


def pick_next_request(waiting: deque[Request], policy: str) -> Request:
    if policy == "fifo":
        return waiting.popleft()

    if policy == "shortest_job_first":
        best = min(waiting, key=lambda r: r.output_tokens)
        waiting.remove(best)
        return best

    if policy == "priority":
        best = max(waiting, key=lambda r: r.priority)
        waiting.remove(best)
        return best

    raise ValueError(f"Unknown policy: {policy}")


def simulate(
    total_kv_blocks: int,
    max_batch_size: int,
    policy: str = "fifo",
    block_size_tokens: int = 256,
    max_time: int = 100_000,
):
    pending = deque(generate_requests())
    waiting = deque()
    running = []
    finished = []

    block_manager = KVCacheBlockManager(
        total_blocks=total_kv_blocks,
        block_size_tokens=block_size_tokens,
    )

    time = 0
    memory_utilization_samples = []
    queue_depth_samples = []
    memory_blocked_events = 0

    while time < max_time and (pending or waiting or running):
        while pending and pending[0].arrival_time <= time:
            waiting.append(pending.popleft())

        next_waiting = deque()

        while waiting and len(running) < max_batch_size:
            req = pick_next_request(waiting, policy)

            if block_manager.allocate_initial_prompt(req):
                req.start_time = time
                running.append(req)
            else:
                memory_blocked_events += 1
                next_waiting.append(req)

        while waiting:
            next_waiting.append(waiting.popleft())

        waiting = next_waiting

        next_running = []

        for req in running:
            if req.prefill_remaining_steps > 0:
                req.prefill_remaining_steps -= 1

                if req.prefill_remaining_steps == 0:
                    req.first_token_time = time

                next_running.append(req)
                continue

            can_grow = block_manager.maybe_grow_for_decode(req)
            if not can_grow:
                memory_blocked_events += 1
                next_running.append(req)
                continue

            req.generated_tokens += 1
            req.decode_remaining_tokens -= 1

            if req.decode_remaining_tokens == 0:
                req.finish_time = time
                block_manager.release(req)
                finished.append(req)
            else:
                next_running.append(req)

        running = next_running

        memory_utilization_samples.append(block_manager.utilization)
        queue_depth_samples.append(len(waiting))
        time += 1

    if not finished:
        raise RuntimeError("No requests completed. Increase KV blocks or max_time.")

    latencies = [r.finish_time - r.arrival_time for r in finished]
    wait_times = [r.start_time - r.arrival_time for r in finished]
    ttfts = [r.first_token_time - r.arrival_time for r in finished]
    decode_times = [r.finish_time - r.first_token_time for r in finished]
    tpots = [
        decode_time / r.output_tokens
        for r, decode_time in zip(finished, decode_times)
    ]

    return {
        "policy": policy,
        "total_kv_blocks": total_kv_blocks,
        "max_batch_size": max_batch_size,
        "block_size_tokens": block_size_tokens,
        "completed": len(finished),
        "not_completed": len(pending) + len(waiting) + len(running),
        "avg_latency": round(statistics.mean(latencies), 2),
        "p95_latency": round(statistics.quantiles(latencies, n=20)[18], 2),
        "p99_latency": round(statistics.quantiles(latencies, n=100)[98], 2),
        "avg_wait_time": round(statistics.mean(wait_times), 2),
        "avg_ttft": round(statistics.mean(ttfts), 2),
        "p99_ttft": round(statistics.quantiles(ttfts, n=100)[98], 2),
        "avg_tpot": round(statistics.mean(tpots), 4),
        "avg_queue_depth": round(statistics.mean(queue_depth_samples), 2),
        "peak_queue_depth": max(queue_depth_samples),
        "avg_memory_utilization": round(statistics.mean(memory_utilization_samples), 4),
        "peak_memory_utilization": round(max(memory_utilization_samples), 4),
        "memory_blocked_events": memory_blocked_events,
        "throughput_reqs_per_step": round(len(finished) / time, 4),
        "total_time": time,
    }


if __name__ == "__main__":
    policies = ["fifo", "shortest_job_first", "priority"]

    for policy in policies:
        print(f"\nPolicy: {policy}")
        for total_blocks in [128, 256, 512]:
            print(
                simulate(
                    total_kv_blocks=total_blocks,
                    max_batch_size=16,
                    policy=policy,
                )
            )