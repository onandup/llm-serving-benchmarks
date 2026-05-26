import pandas as pd
import matplotlib.pyplot as plt

from kv_cache_simulator import simulate


kv_block_sizes = [64, 128, 256, 512]
batch_sizes = [4, 8, 16, 32]

rows = []

for total_blocks in kv_block_sizes:
    for batch_size in batch_sizes:
        rows.append(
            simulate(
                total_kv_blocks=total_blocks,
                max_batch_size=batch_size,
            )
        )

df = pd.DataFrame(rows)
print(df)

df.to_csv("results/kv_cache_results.csv", index=False)


def plot_metric(metric, ylabel, title, filename):
    plt.figure()

    for total_blocks in kv_block_sizes:
        subset = df[df["total_kv_blocks"] == total_blocks]
        plt.plot(
            subset["max_batch_size"],
            subset[metric],
            marker="o",
            label=f"{total_blocks} KV blocks",
        )

    plt.xlabel("Max Batch Size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(f"results/charts/{filename}", bbox_inches="tight")


plot_metric(
    "throughput_reqs_per_step",
    "Throughput: Requests / Step",
    "Throughput vs Batch Size Under KV Cache Limits",
    "kv_throughput_vs_batch_size.png",
)

plot_metric(
    "p99_latency",
    "P99 Latency",
    "P99 Latency vs Batch Size Under KV Cache Limits",
    "kv_p99_latency_vs_batch_size.png",
)

plot_metric(
    "avg_memory_utilization",
    "Average KV Cache Utilization",
    "Average KV Cache Utilization vs Batch Size",
    "kv_avg_memory_utilization.png",
)

plot_metric(
    "peak_memory_utilization",
    "Peak KV Cache Utilization",
    "Peak KV Cache Utilization vs Batch Size",
    "kv_peak_memory_utilization.png",
)