# LLM Serving Benchmarks

This project explores production LLM serving tradeoffs: scheduler policy, batching, throughput, and tail latency.

## Why this matters

LLM inference performance depends on:
- GPU utilization
- memory efficiency
- batching strategy
- request scheduling
- p95/p99 latency

This repo starts with a scheduler simulator inspired by continuous batching systems like vLLM.

## Current features

- Request arrival simulation
- Batch size sweep
- Throughput measurement
- p95 / p99 latency measurement
- Charts for throughput and tail latency

## How to run

```bash
pip install -r requirements.txt
python src/scheduler_simulator.py
python src/plot_scheduler_results.py