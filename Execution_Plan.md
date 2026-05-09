# Execution Plan: Transformer Block - From Scratch to torch.compile

**Goal:** Build three variants of a Transformer block, benchmark them, and export the compute graph.  
**Timeline:** Week 1 (5-7 days)  
**Outcome:** A working project + hands-on understanding of torch.compile, its modes, and its limitations.

---

## Project Structure

```
transformer_compile_bench/
├── src/
│   ├── numpy_transformer.py       # Variant 1: NumPy/plain Python
│   ├── torch_transformer.py       # Variant 2: Vanilla PyTorch
│   ├── compiled_transformer.py    # Variant 3: torch.compile
│   └── export_graph.py            # Graph export
├── benchmark.py                   # Benchmarking runner
├── results/
│   └── benchmark_results.csv      # Output metrics
├── requirements.txt
└── README.md
```

---

## Phase 1: Build the NumPy Transformer Block (Day 1)

**What to build:**  
A single Transformer block with the following components, written using only NumPy and plain Python:

- Scaled Dot-Product Attention (single head is fine)
- Layer Normalization
- Feed-Forward Network (two linear layers with ReLU)
- Residual connections

**Scope:**  
No training. Forward pass only. Use random weight matrices. Batch size of 1 is acceptable.

**Goal:**  
Understand every operation at the math level before any framework abstracts it away.

**Reference:**  
"Attention Is All You Need" (arXiv:1706.03762) - you already know this from Project 1.

---

## Phase 2: Rewrite in Vanilla PyTorch (Day 2)

**What to build:**  
Reproduce the exact same Transformer block using PyTorch primitives:

- Use `torch.nn.Module` as the base class
- Use `torch.nn.Linear`, `torch.nn.LayerNorm`
- Implement attention manually using `torch.matmul` and `torch.softmax`
- Do NOT use `torch.nn.MultiheadAttention` - write it yourself

**Goal:**  
Validate your NumPy implementation by checking that outputs are numerically close between the two.  
Use `torch.allclose()` or `numpy.allclose()` to confirm.

**Key habit to build:**  
Always check shapes at every step. Print `tensor.shape` liberally.

---

## Phase 3: Add torch.compile (Day 3)

**What to build:**  
Wrap your PyTorch module with `torch.compile` and run the same forward pass.

**The three modes to test:**
```python
torch.compile(model, mode="default")
torch.compile(model, mode="reduce-overhead")
torch.compile(model, mode="max-autotune")
```

**For each mode, understand:**
- What is the compiler optimizing for?
- What is the tradeoff (compile time vs. runtime)?
- When would you use each in production?

**Read before writing code:**
- PyTorch docs on `torch.compile`: https://pytorch.org/docs/stable/generated/torch.compile.html
- TorchDynamo overview: https://pytorch.org/docs/stable/dynamo/index.html

**Known limitations to encounter and document:**
- Graph breaks (what causes them, how to detect them)
- Dynamic shapes and how to handle them
- Unsupported Python constructs inside compiled functions

**How to detect graph breaks:**
```python
import torch._dynamo
torch._dynamo.explain(model)(input)
```

Use this. It will teach you more than any blog post.

---

## Phase 4: Benchmark All Three Variants (Day 4)

**What to measure:**
- Inference latency (milliseconds per forward pass)
- Use wall-clock time with `time.perf_counter()`
- Warm up the compiled model before measuring (important - look up why)

**Benchmark structure:**
- Fixed input: batch_size=8, seq_len=128, d_model=256
- Runs: 100 iterations after a warmup of 10 iterations
- Report: mean, min, max latency across runs

**What to record per variant:**
| Variant | Mean Latency (ms) | Min | Max |
|---|---|---|---|
| NumPy | | | |
| PyTorch (eager) | | | |
| torch.compile (default) | | | |
| torch.compile (reduce-overhead) | | | |
| torch.compile (max-autotune) | | | |

Save results to `results/benchmark_results.csv`.

**Note on NumPy benchmarking:**  
NumPy will be slower. That is expected and the point. It shows what the framework buys you.

---

## Phase 5: Export the Compute Graph (Day 5)

**What to build:**  
Export the compiled model's compute graph using `torch.export`.

```python
import torch

exported = torch.export.export(model, args=(example_input,))
print(exported.graph)
```

**What to explore:**
- What does the exported graph look like?
- Can you identify your attention operations in the graph nodes?
- What is an `ExportedProgram` and what does it contain?

**Also try `torch.fx` tracing:**
```python
from torch.fx import symbolic_trace
traced = symbolic_trace(model)
traced.graph.print_tabular()
```

Compare what `torch.fx` gives you vs `torch.export`. They serve different purposes - understand the difference.

**Reference:**
- torch.export docs: https://pytorch.org/docs/stable/export.html
- torch.fx docs: https://pytorch.org/docs/stable/fx.html

---

## Phase 6: Document and Wrap Up (Day 6-7)

**Write a README.md that covers:**
- What you built and why
- How to run each variant
- Your benchmark results (include the table)
- What you learned about torch.compile modes
- Any graph breaks you encountered and how you resolved them

**Reflect on these questions in writing:**
- Where did graph breaks occur in your model, if any?
- Which compile mode gave the best speedup for this workload?
- What would need to change to run this on different hardware?

This reflection is not busywork. At your internship, you will be explaining these tradeoffs to engineers.

---

## Checkpoints

| Day | Deliverable |
|---|---|
| 1 | NumPy forward pass working, outputs make sense |
| 2 | PyTorch variant matches NumPy outputs numerically |
| 3 | All three compile modes running, graph breaks documented |
| 4 | Benchmark script complete, results CSV saved |
| 5 | Graph exported, graph nodes explored |
| 6-7 | README written, repo clean |

---

## Constraints and Scope Guards

- Single Transformer block only. Not a full model. Specifically, one Encoder block (self-attention + residual + LayerNorm + FFN + residual + LayerNorm) — the repeating unit inside a Transformer Encoder. No Decoder, no cross-attention, no embedding layers.
- Single-head attention only. Multi-head adds complexity without adding learning value here.
- No training loop. Forward pass only.
- No GPU required. CPU is fine for this scale. If you have a GPU, run both and note the difference.
- d_model = 256, seq_len = 128, batch_size = 8. Keep it fixed throughout.

---

## What This Project Teaches You

By the end of this, you will have:

- Rebuilt your intuition for Transformer internals at the operation level
- Written and used torch.compile in all three modes with real understanding of each
- Detected and understood graph breaks
- Exported a compute graph and read it
- Built a benchmarking setup you can reuse for your internship work