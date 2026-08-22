# KVShrink Design

This document describes the attention-only connector. Models that also
have GDN/Mamba layers (Qwen3.5 and similar) take a separate path with
different restore rules; see
[KVShrink Hybrid Design](kvshrink-hybrid.md).

## Purpose

KVShrink is a reference vLLM V1 KV-cache connector built on IAXL. It shows how a serving runtime can reuse prompt KV blocks from host memory or persistent storage while Intel Data Streaming Accelerator (DSA) and Intel QuickAssist Technology (QAT) optimize the data path.

Within this Intel reference solution, KVShrink is the framework integration and IAXL is the reusable acceleration layer. KVShrink owns vLLM request and layer semantics; IAXL owns block transfer, compression, and storage.

## Architecture

```mermaid
flowchart TD
    Client["OpenAI-compatible client"] --> Engine["vLLM V1 engine"]

    subgraph Scheduler["vLLM scheduler process"]
        Policy["Scheduling and load budget"] --> SConn["KVShrink scheduler connector"]
        SConn --> Hash["Block hashing and prefix lookup"]
        Hash --> Index["IAXL metadata-only KVStore"]
    end

    Engine --> Policy
    SConn --> Meta["Per-request connector metadata"]

    subgraph Workers["Tensor-parallel workers"]
        Meta --> WConn["KVShrink worker connector"]

        subgraph Store["IAXL KVStore per rank"]
            Core["Block cache pipeline"]
            DSA["Transfer feature: Intel DSA"]
            QAT["Compression feature: Intel QAT"]
            Core --- DSA
            Core --- QAT
        end

        WConn --> Core
        Device["vLLM KV tensors"] <--> DSA
        QAT <--> Host["Compressed DDR / persistent cache"]
    end
```

The scheduler-side connector runs `KVStore` in metadata-only mode. It hashes complete prompt blocks, finds the longest contiguous external-cache prefix, allocates vLLM KV blocks, and sends load/save metadata to workers. Each worker binds its rank-local vLLM KV tensors to a full IAXL `KVStore`.

## Request Flow

```mermaid
flowchart TD
    Request["New vLLM request"] --> Hash["Hash complete prompt blocks"]
    Hash --> Lookup["Lookup contiguous cached prefix"]
    Lookup --> Hit{"External KV hit?"}

    Hit -->|No| Prefill["Compute prefill"]
    Hit -->|Yes| Allocate["Allocate vLLM KV blocks"]
    Allocate --> Mode{"Async load enabled?"}

    Mode -->|No| Sync["Load and wait per layer"]
    Mode -->|Yes| Background["Submit per-request background load"]
    Background --> Gate["All layers or first N layers ready"]
    Gate --> Resume["Notify scheduler and resume request"]
    Resume --> OnDemand["Wait for remaining layers on demand"]

    Sync --> Prefill
    OnDemand --> Prefill
    Prefill --> Decode["Decode tokens"]
    Prefill --> Save["Save newly computed missing blocks"]
    Save --> IAXL["IAXL: DSA transfer, QAT compression, cache"]
```

The scheduler returns both the number of externally cached tokens and whether the request should load asynchronously. Synchronous requests are merged into a load batch. Asynchronous requests retain separate task state so each request can be resumed independently.

With early-layer promotion, the worker reports a load complete after the first configured layers are ready. vLLM can then begin prefill while IAXL continues loading later layers; `wait_for_layer_load()` enforces the dependency immediately before each layer consumes its KV data.

## Intel Accelerator Optimizations

- **Intel DSA accelerates KV movement.** IAXL batches fragmented KV regions and uses DSA with GDRCopy for H2D and D2H transfers on supported CUDA systems. This reduces CPU-driven copy overhead and makes host-backed cache reuse more practical.
- **Intel QAT reduces cache size.** KV blocks are compressed before entering the host cache. The smaller representation raises effective DDR capacity and lowers persistence bandwidth.
- **Minimal compute interference.** When QAT is enabled, compression and decompression run asynchronously on dedicated hardware, so cache-size reduction adds little pressure to inference CPU resources and is designed to have minimal impact on model execution performance. Compatible CPU DEFLATE workers remain available when software capacity is needed.

## Data-Path Optimizations

- **Decompression-inference overlap:** KVShrink starts prefill after the configured leading KV layers are ready, while IAXL continues decompressing and loading later layers in parallel with inference.
- **Adaptive asynchronous loading:** `KVSHRINK_VLLM_KV_ASYNC_LOAD_ENABLED` enables background loading, while `KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS_DYNAMIC_MAP` selects synchronous or layer-gated asynchronous loading for each request-concurrency range.
- **Layer-level overlap:** `KVSHRINK_VLLM_KV_ASYNC_LOAD_LAYERS` controls when a partially loaded request may resume.
- **Rank-local affinity:** CPU cores, QAT devices, and DSA work queues are assigned per tensor-parallel rank to preserve NUMA locality.

The dynamic layer map uses contiguous inclusive concurrency ranges. It must
start at zero and end with an open range. For example,
`0-3:0,4-6:4,7-:8` selects synchronous loading for concurrency 0–3, waits for
four layers at concurrency 4–6, and waits for eight layers at concurrency 7 or
higher. A layer value of zero means synchronous loading for that range.
