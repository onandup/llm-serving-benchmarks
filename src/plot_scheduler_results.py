import pandas as pd
import matplotlib.pyplot as plt

from scheduler_simulator import simulate


policies = ["fifo", "shortest_job_first", "priority"]
batch_sizes = [1, 2, 4, 8, 16, 32]

rows = []
for policy in policies:
    for batch_size in batch_sizes:
        rows.append(simulate(max_batch_size=batch_size, policy=policy))

df = pd.DataFrame(rows)
print(df)

df.to_csv("results/scheduler_results.csv", index=False)


def plot_metric(metric, ylabel, title, filename):
    plt.figure()
    for policy in policies:
        subset = df[df["policy"] == policy]
        plt.plot(
            subset["batch_size"],
            subset[metric],
            marker="o",
            label=policy,
        )

    plt.xlabel("Max Batch Size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(f"results/charts/{filename}", bbox_inches="tight")


plot_metric(
    "throughput_reqs_per_step",
    "Throughput: Requests / Step",
    "Throughput vs Batch Size by Scheduling Policy",
    "throughput_by_policy.png",
)

plot_metric(
    "p99_latency",
    "P99 End-to-End Latency",
    "P99 End-to-End Latency vs Batch Size by Scheduling Policy",
    "p99_latency_by_policy.png",
)

plot_metric(
    "avg_wait_time",
    "Average Wait Time",
    "Average Wait Time vs Batch Size by Scheduling Policy",
    "wait_time_by_policy.png",
)

plot_metric(
    "avg_ttft",
    "Average TTFT",
    "Average Time to First Token vs Batch Size",
    "avg_ttft_by_policy.png",
)

plot_metric(
    "p99_ttft",
    "P99 TTFT",
    "P99 Time to First Token vs Batch Size",
    "p99_ttft_by_policy.png",
)

plot_metric(
    "avg_tpot",
    "Average TPOT",
    "Average Time Per Output Token vs Batch Size",
    "avg_tpot_by_policy.png",
)