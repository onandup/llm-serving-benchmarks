# LLM Serving Benchmarks

## Overview
This project explores LLM serving tradeoffs: continuous batching, throughput, p99 latency, and scheduler behavior.

## Why this matters
Production LLM inference is constrained by GPU utilization, memory efficiency, request scheduling, and tail latency.

## Current Features
- Scheduler simulator
- Batch size sweep
- Throughput and p99 latency charts

## How to run
pip install -r requirements.txt
python src/scheduler_simulator.py
python src/plot_scheduler_results.py

## Results
Include generated charts here.

## Next Steps
- Add FIFO vs shortest-job-first scheduling
- Add prefill/decode modeling
- Add workload classes
- Add KV cache visualizer