# LLM Serving Benchmarks

A systems-focused exploration of production LLM inference tradeoffs including:

* continuous batching
* scheduler policy
* throughput vs latency
* prefill vs decode behavior
* KV cache memory pressure
* TTFT (Time To First Token)
* TPOT (Time Per Output Token)

This project is inspired by production serving systems such as vLLM and explores the systems challenges involved in large-scale LLM inference.

---

# Why This Matters

LLM serving performance depends on:

* GPU utilization
* batching efficiency
* memory efficiency
* scheduler policy
* request fairness
* tail latency (p95 / p99)
* KV cache management

Modern serving systems like vLLM improve inference performance through:

* continuous batching
* efficient KV cache management
* dynamic scheduling
* paged KV allocation

This project builds intuition for those tradeoffs through simulation and visualization.

---

# Current Features

## Scheduler Simulator

Simulates:

* request arrival over time
* concurrent request execution
* dynamic batching
* scheduler policies

Supported scheduling policies:

* FIFO
* Shortest Job First
* Priority Scheduling

---

## Prefill vs Decode Modeling

The simulator separately models:

### Prefill Phase

* prompt processing
* TTFT impact
* prompt-length-dependent latency

### Decode Phase

* token-by-token generation
* TPOT impact
* sequential decode bottlenecks

This mirrors real-world LLM inference systems.

---

## KV Cache Memory Simulation

The simulator models:

* KV cache block allocation
* GPU memory pressure
* request admission constraints
* KV cache utilization
* concurrency limits imposed by memory

This simulates one of the central bottlenecks in production LLM serving systems:
GPU memory exhaustion caused by KV cache growth.

---

# Metrics Collected

The simulator tracks:

* Throughput
* End-to-end latency
* Average wait time
* P95 latency
* P99 latency
* TTFT (Time To First Token)
* TPOT (Time Per Output Token)
* Average KV cache utilization
* Peak KV cache utilization
* Request rejection / admission pressure

---

# Repository Structure

```text
src/
  scheduler_simulator.py
  plot_scheduler_results.py
  kv_cache_simulator.py
  plot_kv_cache_results.py

results/
  charts/
  scheduler_results.csv
  kv_cache_results.csv
```

---

# How To Run

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run scheduler simulator

```bash
python src/scheduler_simulator.py
```

## Generate scheduler charts

```bash
python src/plot_scheduler_results.py
```

## Run KV cache simulator

```bash
python src/kv_cache_simulator.py
```

## Generate KV cache charts

```bash
python src/plot_kv_cache_results.py
```

Charts are generated under:

```text
results/charts/
```

---

# Experimental Findings

## 1. Larger batches improve throughput

Increasing batch size improves throughput because more requests are processed concurrently.

This mirrors production serving systems where continuous batching improves GPU utilization.

---

## 2. Aggressive batching can increase tail latency

Larger batches improve utilization but can negatively impact:

* p95 latency
* p99 latency
* fairness

This is one of the key tradeoffs in production inference systems.

---

## 3. Scheduling policy matters significantly

### FIFO

* simple and fair
* vulnerable to head-of-line blocking

### Shortest Job First

* improves average latency
* may starve large requests

### Priority Scheduling

* improves responsiveness for critical requests
* lower-priority requests may wait significantly longer

---

## 4. TTFT grows with prompt size

Longer prompts require more prefill work before generation can begin.

This reflects real-world inference behavior where long context windows increase:

* attention compute
* KV cache usage
* first-token latency

---

## 5. KV cache capacity directly limits concurrency

The KV cache simulation demonstrates that GPU memory becomes a hard upper bound on concurrent request execution.

As KV cache utilization approaches capacity:

* request admission becomes constrained
* wait times increase
* throughput gains diminish

This mirrors real-world serving systems where KV cache memory pressure is often the dominant serving bottleneck.

---

# Engineering Analysis & Learnings

## 1. Continuous batching improves throughput significantly

The simulator demonstrates that larger dynamic batches substantially improve request throughput because the serving system keeps compute resources utilized more consistently.

This mirrors real-world serving systems such as vLLM where continuous batching is one of the primary throughput optimizations.

Key insight:

* GPU utilization improves when idle gaps between requests are minimized.

However, maximizing throughput alone is insufficient because latency tradeoffs emerge quickly under bursty workloads.

---

## 2. Tail latency becomes the critical production challenge

As batch size increases:

* average throughput improves
* p95/p99 latency can worsen

This reflects one of the most important real-world serving tradeoffs:

```text
maximize utilization
vs
maintain low tail latency
```

In production environments, p99 latency often matters more than average latency because:

* user-facing responsiveness degrades
* timeout risk increases
* fairness issues emerge across workloads

This is particularly important for recommendation systems and interactive AI applications.

---

## 3. Scheduler policy has major impact on fairness and responsiveness

The simulator demonstrates that scheduler policy materially affects:

* wait time
* tail latency
* request fairness

### FIFO

FIFO scheduling is simple and fair but can suffer from head-of-line blocking where long-running requests delay smaller latency-sensitive requests.

### Shortest Job First

Shortest-job-first improves average latency and throughput efficiency for smaller requests.

However:

* large requests may experience starvation
* fairness degrades under skewed workloads

This mirrors tradeoffs seen in real serving systems where latency optimization can negatively impact workload fairness.

### Priority Scheduling

Priority scheduling improves responsiveness for critical workloads but introduces the possibility of sustained delays for lower-priority requests.

This resembles production scenarios where:

* premium traffic
* ranking workloads
* interactive requests

may receive preferential treatment over batch inference jobs.

---

## 4. Prefill and decode exhibit fundamentally different behavior

One of the most important learnings from this project is that prefill and decode phases behave very differently.

### Prefill

Prefill:

* processes the entire prompt
* scales with prompt length
* is more compute-heavy and parallelizable

Long prompts significantly increase TTFT because the model must process the entire context before generation begins.

### Decode

Decode:

* generates tokens sequentially
* is memory-bandwidth sensitive
* is harder to parallelize efficiently

This explains why decode often becomes the dominant bottleneck in production serving systems.

---

## 5. KV cache memory pressure is a first-class serving constraint

The KV cache simulation highlighted that memory management is just as important as compute scheduling in large-scale inference systems.

Even when compute capacity exists:

* requests may still be blocked due to KV cache exhaustion
* concurrency can collapse under large-context workloads
* memory pressure can dominate serving efficiency

This explains why systems such as vLLM introduced paged KV cache allocation mechanisms like PagedAttention.

---

## 6. Prompt tokens and generated tokens behave differently in KV cache growth

One important modeling insight from this project was understanding that:

* prompt token KV cache exists immediately after prefill
* generated tokens incrementally grow KV cache during decode

This distinction is important because decode-phase memory growth can progressively reduce concurrency over time.

This is one of the central scalability challenges in modern LLM serving systems.

---

## 7. Throughput optimization alone is not sufficient

An important systems insight from this project is that the “best” scheduler depends on workload characteristics.

Production serving systems must balance:

* throughput
* latency
* fairness
* memory pressure
* workload prioritization

This is why modern inference systems require sophisticated schedulers rather than static batching approaches.

---

## 8. Why vLLM-style systems matter

This project helped build intuition for why systems such as vLLM introduced:

* continuous batching
* dynamic scheduling
* efficient KV cache management
* paged KV allocation

Traditional static batching approaches leave significant compute capacity underutilized and struggle under heterogeneous workloads.

The simulator demonstrates how scheduler efficiency and memory efficiency become first-class concerns in large-scale inference serving systems.

---

# Example Charts

## Throughput by Scheduling Policy

![Throughput](results/charts/throughput_by_policy.png)

---

## P99 Latency by Scheduling Policy

![P99 Latency](results/charts/p99_latency_by_policy.png)

---

## Average Wait Time by Scheduling Policy

![Wait Time](results/charts/wait_time_by_policy.png)

---

## Average TTFT by Scheduling Policy

![TTFT](results/charts/avg_ttft_by_policy.png)

---

## P99 TTFT by Scheduling Policy

![P99 TTFT](results/charts/p99_ttft_by_policy.png)

---

## Average TPOT by Scheduling Policy

![TPOT](results/charts/avg_tpot_by_policy.png)

---

## KV Throughput vs Batch Size

![KV Throughput](results/charts/kv_throughput_vs_batch_size.png)

---

## KV P99 Latency vs Batch Size

![KV P99](results/charts/kv_p99_latency_vs_batch_size.png)

---

## Average KV Cache Utilization

![KV Avg Utilization](results/charts/kv_avg_memory_utilization.png)

---

## Peak KV Cache Utilization

![KV Peak Utilization](results/charts/kv_peak_memory_utilization.png)

---

# Unified Serving Simulator

The repository now includes a unified serving simulator that combines:

* scheduler policy
* prefill vs decode behavior
* KV cache allocation
* decode-phase KV cache growth
* memory-aware admission control
* queueing behavior
* throughput and latency analysis

This models a simplified production-style LLM serving stack inspired by systems such as vLLM.

---

## Unified Simulator Features

The unified simulator models:

### Request Lifecycle

Each request includes:

* prompt tokens
* generated output tokens
* scheduler priority
* prefill phase
* decode phase
* KV cache growth over time

---

### Scheduler Policies

Supported policies:

* FIFO
* Shortest Job First
* Priority Scheduling

---

### KV Cache Memory Management

The simulator models:

* KV block allocation
* memory-aware request admission
* decode-time KV growth
* KV block release on completion
* GPU memory pressure

---

### Decode-Phase KV Growth

One important improvement in the unified simulator is modeling incremental KV cache growth during decode.

The simulator now distinguishes:

* prompt token KV cache allocated during prefill
* generated token KV cache added gradually during decode

This better reflects real-world inference behavior.

---

# Unified Simulator Metrics

The unified simulator tracks:

* throughput
* end-to-end latency
* p95/p99 latency
* TTFT
* TPOT
* queue depth
* average KV utilization
* peak KV utilization
* memory-blocked events
* request admission pressure

---

# Running the Unified Simulator

## Run unified simulator

```bash
python src/unified_serving_simulator.py
```

## Generate unified simulator charts

```bash
python src/plot_unified_results.py
```

---

# Unified Simulator Findings

## 1. Memory pressure and scheduler behavior are tightly coupled

One of the most important learnings from the unified simulator is that scheduling efficiency cannot be analyzed independently from memory behavior.

As KV cache utilization increases:

* request admission slows
* queue depth grows
* tail latency increases

This demonstrates why modern inference systems require memory-aware schedulers.

---

## 2. Long-lived decode workloads reduce concurrency over time

The unified simulator models decode-phase KV cache growth, showing that requests progressively consume more memory during generation.

This creates an important serving effect:

* concurrency degrades dynamically during long generations
* admission pressure increases over time
* throughput eventually plateaus under memory pressure

This is a major challenge in production LLM serving systems.

---

## 3. Memory residency time becomes a critical scalability constraint

The simulator demonstrates that concurrency is constrained not only by compute capacity but also by:

* how long requests occupy KV cache memory
* how quickly memory can be reclaimed
* how efficiently memory is allocated

This helps explain why modern serving systems emphasize:

* paging
* memory reuse
* continuous batching
* efficient KV allocation

---

## 4. Scheduler policy influences memory pressure indirectly

Different scheduling policies affect:

* queue growth
* request completion order
* memory residency duration
* decode overlap

This means scheduler policy impacts both:

* latency behavior
* memory efficiency

An important insight is that scheduler optimization and memory optimization are deeply interconnected in large-scale inference systems.

---

## 5. Decode-phase memory growth is operationally dangerous

The simulator demonstrates that generated tokens progressively increase memory usage over time.

This means:

* long generations can unexpectedly reduce available concurrency
* memory utilization may spike even after admission succeeds
* systems may become unstable under long-context workloads

This is one of the reasons long-context serving remains operationally challenging at scale.

---

## 6. Why PagedAttention-style systems matter

The unified simulator helped build intuition for why systems such as vLLM introduced:

* paged KV cache allocation
* dynamic block management
* memory-efficient continuous batching

Without efficient paging strategies:

* fragmentation increases
* memory reuse becomes inefficient
* concurrency collapses under heterogeneous workloads

The simulator demonstrates that memory efficiency becomes a first-class concern in modern inference systems.

---

# Unified Simulator Charts

## Unified Throughput vs Batch Size

![Unified Throughput](results/charts/unified_throughput_by_kv_blocks.png)

---

## Unified P99 Latency vs Batch Size

![Unified P99 Latency](results/charts/unified_p99_latency_by_kv_blocks.png)

---

## Unified KV Utilization vs Batch Size

![Unified KV Utilization](results/charts/unified_memory_utilization_by_kv_blocks.png)

---

## Unified Average TTFT by Policy

![Unified TTFT](results/charts/unified_avg_ttft_by_policy.png)

---

## Unified Memory Blocking by Policy

![Unified Memory Blocking](results/charts/unified_memory_blocking_by_policy.png)

---

## Unified Queue Depth by Policy

![Unified Queue Depth](results/charts/unified_queue_depth_by_policy.png)

# KV Cache Fragmentation Simulation

The repository now includes a fragmentation simulator comparing:

* contiguous KV cache allocation
* paged/block-based KV allocation

This models one of the key architectural insights behind systems such as vLLM and PagedAttention.

---

## Why Fragmentation Matters

Traditional contiguous memory allocation strategies can fail even when sufficient total free memory exists.

Example:

```text id="0pnrkc"
Free blocks:
[ ][X][ ][X][ ][X]

Need:
3 contiguous blocks
```

Even though 3 total blocks are free, allocation fails because the memory is fragmented.

This becomes increasingly problematic under:

* heterogeneous request sizes
* long-lived decode workloads
* dynamic request arrival patterns

---

## Paged Allocation

Paged allocation removes the requirement for contiguous memory.

Instead:

```text id="13gh9y"
logical KV layout
!=
physical memory layout
```

Requests are allocated arbitrary free blocks/pages.

This significantly improves:

* memory reuse
* allocation success rate
* concurrency
* GPU utilization

This is one of the core ideas behind PagedAttention.

---

# Fragmentation Simulator Features

The simulator models:

* contiguous memory allocation
* paged/block allocation
* allocation failures
* fragmentation-induced failures
* random request arrivals
* heterogeneous request sizes
* memory release and reuse

---

# Running the Fragmentation Simulator

## Run fragmentation simulator

```bash id="6hql2o"
python src/fragmentation_simulator.py
```

## Generate fragmentation charts

```bash id="ye6jz8"
python src/plot_fragmentation_results.py
```

---

# Fragmentation Findings

## 1. Contiguous allocation suffers from fragmentation collapse

The simulator demonstrates that contiguous allocation strategies experience increasing allocation failures over time due to fragmentation.

Importantly:

* failures occur even when sufficient total free memory exists
* memory becomes unusable because free space is non-contiguous

This is one of the core scalability problems in traditional KV cache allocation approaches.

---

## 2. Paged allocation significantly improves allocation efficiency

Paged allocation avoids fragmentation-induced failures because requests can occupy arbitrary free blocks.

This leads to:

* higher allocation success rates
* improved memory reuse
* more stable concurrency
* lower memory waste

This demonstrates why block-based KV allocation became foundational in modern serving systems.

---

## 3. Fragmentation is workload dependent

The simulator shows fragmentation worsening under:

* heterogeneous request sizes
* random request release timing
* bursty workloads

Large long-lived requests create allocation holes that progressively reduce allocator efficiency.

This mirrors production inference serving behavior.

---

## 4. Memory efficiency directly impacts throughput

As fragmentation increases:

* allocation failures rise
* concurrency decreases
* throughput degrades

This demonstrates that serving throughput is strongly influenced by memory allocator efficiency, not just raw GPU compute.

---

## 5. Why PagedAttention matters

The fragmentation simulation helped build intuition for why systems such as vLLM introduced:

* paged KV allocation
* block-based memory management
* logical-to-physical KV indirection

Without paging:

* fragmentation accumulates
* memory reuse deteriorates
* serving efficiency collapses under heterogeneous workloads

PagedAttention fundamentally improves allocator stability and scalability.

---

# Fragmentation Charts

## Successful Allocations

![Fragmentation Success](results/charts/fragmentation_successful_allocations.png)

---

## Failed Allocations

![Fragmentation Failures](results/charts/fragmentation_failed_allocations.png)

---

## Fragmentation Events

![Fragmentation Events](results/charts/fragmentation_events.png)

---

## Peak Used Blocks

![Peak Used Blocks](results/charts/fragmentation_peak_used_blocks.png)

# Future Improvements

Planned additions:

* dynamic token-level scheduling
* speculative decoding simulation
* multi-tenant serving workloads
* long-context workload simulation
* decode prioritization policies
* realistic decode-phase KV cache growth

---

# Inspiration

This project is heavily inspired by:

* vLLM
* PagedAttention
* modern continuous batching inference systems
* large-scale production LLM serving architectures

