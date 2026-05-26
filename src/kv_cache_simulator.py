from dataclasses import dataclass
from collections import deque
from typing import Optional
import random
import statistics


@dataclass
class Request:
    id: int
    arrival_time: int
    prompt_tokens: int
    output_tokens: int
    remaining_tokens: int
    kv_blocks_needed: int
    start_time: Optional[int] = None
    finish_time: Optional[int] = None


class KVCacheBlockManager:
    def __init__(self, total_blocks: int):
        self.total_blocks = total_blocks
        self.free_blocks = total_blocks
        self.allocations = {}

    def can_allocate(self, request: Request) -> bool:
        return self.free_blocks >= request.kv_blocks_needed

    def allocate(self, request: Request):
        if not self.can_allocate(request):
            raise RuntimeError("Not enough KV cache blocks")

        self.free_blocks -= request.kv_blocks_needed
        self.allocations[request.id] = request.kv_blocks_needed

    def release(self, request: Request):
        blocks = self.allocations.pop(request.id, 0)
        self.free_blocks += blocks

    @property
    def used_blocks(self):
        return self.total_blocks - self.free_blocks

    @property
    def utilization(self):
        return self.used_blocks / self.total_blocks


def generate_requests(n=300, seed=42):
    random.seed(seed)
    requests = []
    time = 0

    for i in range(n):
        time += random.randint(1, 4)

        prompt_tokens = random.choice([128, 512, 1024, 2048, 4096])
        output_tokens = random.choice([32, 64, 128, 256])

        # Simplified KV cache model:
        # KV cache grows with prompt + generated tokens.
        total_tokens = prompt_tokens + output_tokens

        # Assume each KV block stores 256 tokens.
        kv_blocks_needed = max(1, total_tokens // 256)

        requests.append(
            Request(
                id=i,
                arrival_time=time,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                remaining_tokens=output_tokens,
                kv_blocks_needed=kv_blocks_needed,
            )
        )

    return requests


def simulate(total_kv_blocks: int, max_batch_size: int):
    pending = deque(generate_requests())
    waiting = deque()
    running = []
    finished = []
    rejected = 0
    block_manager = KVCacheBlockManager(total_blocks=total_kv_blocks)

    time = 0
    memory_utilization_samples = []

    while pending or waiting or running:
        while pending and pending[0].arrival_time <= time:
            waiting.append(pending.popleft())

        next_waiting = deque()

        while waiting and len(running) < max_batch_size:
            req = waiting.popleft()

            if block_manager.can_allocate(req):
                block_manager.allocate(req)
                req.start_time = time
                running.append(req)
            else:
                next_waiting.append(req)

        while waiting:
            next_waiting.append(waiting.popleft())

        waiting = next_waiting

        next_running = []

        for req in running:
            req.remaining_tokens -= 1

            if req.remaining_tokens == 0:
                req.finish_time = time
                block_manager.release(req)
                finished.append(req)
            else:
                next_running.append(req)

        running = next_running
        memory_utilization_samples.append(block_manager.utilization)
        time += 1

        if time > 100000:
            rejected += len(waiting)
            break

    latencies = [r.finish_time - r.arrival_time for r in finished]
    wait_times = [r.start_time - r.arrival_time for r in finished]

    return {
        "total_kv_blocks": total_kv_blocks,
        "max_batch_size": max_batch_size,
        "completed": len(finished),
        "rejected_or_timed_out": rejected,
        "avg_latency": round(statistics.mean(latencies), 2),
        "p99_latency": round(statistics.quantiles(latencies, n=100)[98], 2),
        "avg_wait_time": round(statistics.mean(wait_times), 2),
        "avg_memory_utilization": round(statistics.mean(memory_utilization_samples), 4),
        "peak_memory_utilization": round(max(memory_utilization_samples), 4),
        "throughput_reqs_per_step": round(len(finished) / time, 4),
        "total_time": time,
    }


if __name__ == "__main__":
    for total_blocks in [64, 128, 256, 512]:
        result = simulate(total_kv_blocks=total_blocks, max_batch_size=16)
        print(result)