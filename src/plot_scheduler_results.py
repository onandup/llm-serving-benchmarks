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

plt.figure()
for policy in policies:
    subset = df[df["policy"] == policy]
    plt.plot(
        subset["batch_size"],
        subset["throughput_reqs_per_step"],
        marker="o",
        label=policy,
    )

plt.xlabel("Max Batch Size")
plt.ylabel("Throughput: Requests / Step")
plt.title("Throughput vs Batch Size by Scheduling Policy")
plt.legend()
plt.savefig("results/charts/throughput_by_policy.png", bbox_inches="tight")

plt.figure()
for policy in policies:
    subset = df[df["policy"] == policy]
    plt.plot(
        subset["batch_size"],
        subset["p99_latency"],
        marker="o",
        label=policy,
    )

plt.xlabel("Max Batch Size")
plt.ylabel("P99 Latency")
plt.title("P99 Latency vs Batch Size by Scheduling Policy")
plt.legend()
plt.savefig("results/charts/p99_latency_by_policy.png", bbox_inches="tight")

plt.figure()
for policy in policies:
    subset = df[df["policy"] == policy]
    plt.plot(
        subset["batch_size"],
        subset["avg_wait_time"],
        marker="o",
        label=policy,
    )

plt.xlabel("Max Batch Size")
plt.ylabel("Average Wait Time")
plt.title("Average Wait Time vs Batch Size by Scheduling Policy")
plt.legend()
plt.savefig("results/charts/wait_time_by_policy.png", bbox_inches="tight")