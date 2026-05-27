import pandas as pd
import matplotlib.pyplot as plt

from unified_serving_simulator import simulate

policies = ["fifo", "shortest_job_first", "priority"]
kv_block_sizes = [128, 256, 512]
batch_sizes = [4, 8, 16, 32]

rows = []

for policy in policies:
    for total_blocks in kv_block_sizes:
        for batch_size in batch_sizes:
            rows.append(
                simulate(
                    total_kv_blocks=total_blocks,
                    max_batch_size=batch_size,
                    policy=policy,
                )
            )

df = pd.DataFrame(rows)
print(df)

df.to_csv("results/unified_results.csv", index=False)


def plot_by_kv_blocks(metric, ylabel, title, filename, policy="fifo"):
    plt.figure()

    subset_policy = df[df["policy"] == policy]

    for total_blocks in kv_block_sizes:
        subset = subset_policy[subset_policy["total_kv_blocks"] == total_blocks]
        plt.plot(
            subset["max_batch_size"],
            subset[metric],
            marker="o",
            label=f"{total_blocks} KV blocks",
        )

    plt.xlabel("Max Batch Size")
    plt.ylabel(ylabel)
    plt.title(f"{title} ({policy})")
    plt.legend()
    plt.savefig(f"results/charts/{filename}", bbox_inches="tight")


def plot_by_policy(metric, ylabel, title, filename, total_blocks=256):
    plt.figure()

    subset_blocks = df[df["total_kv_blocks"] == total_blocks]

    for policy in policies:
        subset = subset_blocks[subset_blocks["policy"] == policy]
        plt.plot(
            subset["max_batch_size"],
            subset[metric],
            marker="o",
            label=policy,
        )

    plt.xlabel("Max Batch Size")
    plt.ylabel(ylabel)
    plt.title(f"{title} ({total_blocks} KV blocks)")
    plt.legend()
    plt.savefig(f"results/charts/{filename}", bbox_inches="tight")


plot_by_kv_blocks(
    "throughput_reqs_per_step",
    "Throughput: Requests / Step",
    "Unified Simulator Throughput vs Batch Size",
    "unified_throughput_by_kv_blocks.png",
)

plot_by_kv_blocks(
    "p99_latency",
    "P99 End-to-End Latency",
    "Unified Simulator P99 Latency vs Batch Size",
    "unified_p99_latency_by_kv_blocks.png",
)

plot_by_kv_blocks(
    "avg_memory_utilization",
    "Average KV Cache Utilization",
    "Unified Simulator KV Utilization vs Batch Size",
    "unified_memory_utilization_by_kv_blocks.png",
)

plot_by_policy(
    "avg_ttft",
    "Average TTFT",
    "Unified Simulator Average TTFT by Policy",
    "unified_avg_ttft_by_policy.png",
)

plot_by_policy(
    "memory_blocked_events",
    "Memory Blocked Events",
    "Unified Simulator Memory Blocking by Policy",
    "unified_memory_blocking_by_policy.png",
)

plot_by_policy(
    "avg_queue_depth",
    "Average Queue Depth",
    "Unified Simulator Queue Depth by Policy",
    "unified_queue_depth_by_policy.png",
)