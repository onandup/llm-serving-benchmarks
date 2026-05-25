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
    start_time: Optional[int] = None
    finish_time: Optional[int] = None


def generate_requests(n=200, seed=42):
    random.seed(seed)
    requests = []
    time = 0

    for i in range(n):
        time += random.randint(1, 5)
        output_tokens = random.choice([32, 64, 128, 256])

        requests.append(
            Request(
                id=i,
                arrival_time=time,
                prompt_tokens=random.choice([64, 128, 512, 1024]),
                output_tokens=output_tokens,
                remaining_tokens=output_tokens,
            )
        )

    return requests


def simulate(max_batch_size: int):
    pending = deque(generate_requests())
    waiting = deque()
    running = []
    finished = []
    time = 0

    while pending or waiting or running:
        while pending and pending[0].arrival_time <= time:
            waiting.append(pending.popleft())

        while waiting and len(running) < max_batch_size:
            req = waiting.popleft()
            req.start_time = time
            running.append(req)

        next_running = []
        for req in running:
            req.remaining_tokens -= 1
            if req.remaining_tokens == 0:
                req.finish_time = time
                finished.append(req)
            else:
                next_running.append(req)

        running = next_running
        time += 1

    latencies = [r.finish_time - r.arrival_time for r in finished]

    return {
        "batch_size": max_batch_size,
        "completed": len(finished),
        "avg_latency": round(statistics.mean(latencies), 2),
        "p95_latency": round(statistics.quantiles(latencies, n=20)[18], 2),
        "p99_latency": round(statistics.quantiles(latencies, n=100)[98], 2),
        "total_time": time,
        "throughput_reqs_per_step": round(len(finished) / time, 4),
    }


if __name__ == "__main__":
    for batch_size in [1, 2, 4, 8, 16, 32]:
        print(simulate(batch_size))
