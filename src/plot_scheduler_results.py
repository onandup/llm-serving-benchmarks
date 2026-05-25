import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scheduler_simulator import simulate


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
CHARTS_DIR = RESULTS_DIR / "charts"

CHARTS_DIR.mkdir(parents=True, exist_ok=True)

rows = [simulate(batch_size) for batch_size in [1, 2, 4, 8, 16, 32]]
df = pd.DataFrame(rows)

print(df)

df.to_csv(RESULTS_DIR / "scheduler_results.csv", index=False)

plt.figure()
plt.plot(df["batch_size"], df["throughput_reqs_per_step"], marker="o")
plt.xlabel("Max Batch Size")
plt.ylabel("Throughput: Requests / Step")
plt.title("Throughput vs Batch Size")
plt.savefig(CHARTS_DIR / "throughput_vs_batch_size.png", bbox_inches="tight")

plt.figure()
plt.plot(df["batch_size"], df["p99_latency"], marker="o")
plt.xlabel("Max Batch Size")
plt.ylabel("P99 Latency")
plt.title("P99 Latency vs Batch Size")
plt.savefig(CHARTS_DIR / "p99_latency_vs_batch_size.png", bbox_inches="tight")
