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
- Scheduling policies and its effects of throughput/latency for different batch sizes

# Findings

## 1. Larger batches improve throughput

As batch size increases, throughput improves significantly because more requests are processed concurrently.

This mirrors production LLM serving systems such as vLLM, where continuous batching improves GPU utilization.

## 2. Tail latency increases with aggressive batching

While throughput improves, p99 latency and wait time can also increase under larger batch sizes.

This demonstrates the core production tradeoff in LLM inference systems:

- maximizing GPU utilization
vs
- maintaining low tail latency

## 3. Scheduling policy impacts fairness and latency

Different scheduling policies produce different latency characteristics.

### FIFO
- simple and fair
- can suffer from head-of-line blocking

### Shortest Job First
- improves average latency
- risks starving large requests

### Priority Scheduling
- useful for latency-sensitive workloads
- lower-priority requests may wait longer

## 4. Continuous batching requires careful scheduler policy

The simulator demonstrates why scheduler design is a critical component of large-scale inference systems.

Aggressive batching policies improve utilization but may negatively impact p95/p99 latency under bursty workloads.

## How to run

```bash
pip install -r requirements.txt
python src/scheduler_simulator.py
python src/plot_scheduler_results.py