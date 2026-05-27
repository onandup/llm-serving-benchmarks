import pandas as pd
import matplotlib.pyplot as plt

from fragmentation_simulator import run_simulation


total_block_sizes = [64, 128, 256, 512]
allocator_types = ["contiguous", "paged"]

rows = []

for total_blocks in total_block_sizes:
    for allocator_type in allocator_types:
        result = run_simulation(
            total_blocks=total_blocks,
            allocator_type=allocator_type,
        )
        rows.append(result.__dict__)

df = pd.DataFrame(rows)
print(df)

df.to_csv("results/fragmentation_results.csv", index=False)


def plot_metric(metric, ylabel, title, filename):
    plt.figure()

    for allocator_type in allocator_types:
        subset = df[df["allocator_type"] == allocator_type]
        plt.plot(
            subset["total_blocks"],
            subset[metric],
            marker="o",
            label=allocator_type,
        )

    plt.xlabel("Total KV Blocks")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(f"results/charts/{filename}", bbox_inches="tight")


plot_metric(
    "successful_allocations",
    "Successful Allocations",
    "Successful Allocations: Contiguous vs Paged",
    "fragmentation_successful_allocations.png",
)

plot_metric(
    "failed_allocations",
    "Failed Allocations",
    "Failed Allocations: Contiguous vs Paged",
    "fragmentation_failed_allocations.png",
)

plot_metric(
    "fragmentation_events",
    "Fragmentation Events",
    "Fragmentation Events in Contiguous Allocation",
    "fragmentation_events.png",
)

plot_metric(
    "peak_used_blocks",
    "Peak Used Blocks",
    "Peak Used Blocks: Contiguous vs Paged",
    "fragmentation_peak_used_blocks.png",
)