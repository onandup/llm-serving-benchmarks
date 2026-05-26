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
    priority: int

    prefill_remaining_steps: int
    decode_remaining_tokens: int

    start_time: Optional[int] = None
    first_token_time: Optional[int] = None
    finish_time: Optional[int] = None


def generate_requests(n=300, seed=42):
    random.seed(seed)
    requests = []
    time = 0

    for i in range(n):
        time += random.randint(1, 5)

        prompt_tokens = random.choice([64, 128, 512, 1024, 2048])
        output_tokens = random.choice([32, 64, 128, 256])

        # Simplified prefill model:
        # Larger prompts take more simulation steps before first token.
        prefill_steps = max(1, prompt_tokens // 256)

        requests.append(
            Request(
                id=i,
                arrival_time=time,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                priority=random.choice([1, 2, 3]),  # 3 = highest priority
                prefill_remaining_steps=prefill_steps,
                decode_remaining_tokens=output_tokens,
            )
        )

    return requests


def pick_next_request(waiting, policy):
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


def simulate(max_batch_size: int, policy: str = "fifo"):
    pending = deque(generate_requests())
    waiting = deque()
    running = []
    finished = []
    time = 0

    while pending or waiting or running:
        while pending and pending[0].arrival_time <= time:
            waiting.append(pending.popleft())

        while waiting and len(running) < max_batch_size:
            req = pick_next_request(waiting, policy)
            req.start_time = time
            running.append(req)

        next_running = []

        for req in running:
            if req.prefill_remaining_steps > 0:
                req.prefill_remaining_steps -= 1

                if req.prefill_remaining_steps == 0:
                    req.first_token_time = time

                next_running.append(req)
                continue

            req.decode_remaining_tokens -= 1

            if req.decode_remaining_tokens == 0:
                req.finish_time = time
                finished.append(req)
            else:
                next_running.append(req)

        running = next_running
        time += 1

    end_to_end_latencies = [r.finish_time - r.arrival_time for r in finished]
    wait_times = [r.start_time - r.arrival_time for r in finished]
    ttfts = [r.first_token_time - r.arrival_time for r in finished]
    decode_times = [r.finish_time - r.first_token_time for r in finished]
    tpots = [decode_time / r.output_tokens for r, decode_time in zip(finished, decode_times)]

    return {
        "policy": policy,
        "batch_size": max_batch_size,
        "completed": len(finished),
        "avg_latency": round(statistics.mean(end_to_end_latencies), 2),
        "p95_latency": round(statistics.quantiles(end_to_end_latencies, n=20)[18], 2),
        "p99_latency": round(statistics.quantiles(end_to_end_latencies, n=100)[98], 2),
        "avg_wait_time": round(statistics.mean(wait_times), 2),
        "avg_ttft": round(statistics.mean(ttfts), 2),
        "p95_ttft": round(statistics.quantiles(ttfts, n=20)[18], 2),
        "p99_ttft": round(statistics.quantiles(ttfts, n=100)[98], 2),
        "avg_decode_time": round(statistics.mean(decode_times), 2),
        "avg_tpot": round(statistics.mean(tpots), 4),
        "total_time": time,
        "throughput_reqs_per_step": round(len(finished) / time, 4),
    }


if __name__ == "__main__":
    policies = ["fifo", "shortest_job_first", "priority"]

    for policy in policies:
        print(f"\nPolicy: {policy}")
        for batch_size in [1, 2, 4, 8, 16, 32]:
            print(simulate(batch_size, policy))