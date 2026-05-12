# gnn-arxiv-benchmarking
GNN
# GNN Architecture Benchmarking on OGBN-Arxiv

A comparative study of Graph Neural Network architectures for large-scale transductive node classification.

**Course:** CCS4354 — Tensors and Graph Neural Networks  
**Group:** 05 — ·W.A.P.M. Weerakkody 
                · S.S. Ellawala   
                · Devanga Wettasinghe  
**Dataset:** [OGBN-Arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) · 169,343 nodes · 1,166,243 edges

---

## Results

| Model | Val Accuracy | Test Accuracy | Parameters | Speed |
|---|---|---|---|---|
| GCN | 60.68% | 54.42% | 0.11 M | Fastest |
| **GraphSAGE** | **63.36%** | **56.59%** | 0.22 M | Moderate |
| GATv2 | 60.48% | 54.17% | 0.22 M | Slowest |

---

## Overview

This project implements and benchmarks three foundational GNN architectures on the OGB citation network benchmark, evaluating the trade-off between accuracy, parameter count, and computational cost.

**Key finding:** GraphSAGE outperforms GCN by +2.17% test accuracy. The decisive factor is its use of **concatenation** to preserve a node's own identity alongside aggregated neighbourhood features, rather than blending them symmetrically as GCN does.

---

## Models

### GCN — Graph Convolutional Network
Spectral-spatial hybrid using symmetric normalisation. The most parameter-efficient model (0.11 M), best suited for memory-constrained environments.

### GraphSAGE — Sample and aggreGAtE
Spatial architecture designed for inductive learning. Uses neighbourhood sampling with mean aggregation and concatenation-based identity preservation.

### GATv2 — Graph Attention Network v2
Dynamic multi-head attention (8 heads) that learns per-edge importance scores. Resolves the static attention limitation of original GAT (Brody et al., 2022), but incurs the highest computational cost.

---

## Dataset

The `ogbn-arxiv` benchmark is a directed citation graph of Computer Science arXiv papers with a **temporal split** — making this a generalisation task across time, not just topology.

| Split | Papers | Years |
|---|---|---|
| Train | ~90,941 | up to 2017 |
| Validation | ~29,799 | 2018 |
| Test | ~48,603 | 2019+ |

- **Node features:** 128-dim word2vec embeddings from titles and abstracts  
- **Classes:** 40 CS subject categories  
- **Preprocessing:** Z-score normalisation computed strictly on the training partition to prevent data leakage

---

## Setup

```bash
pip install torch torch-geometric ogb
```

```python
from ogb.nodeproppred import PygNodePropPredDataset
dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='data/')
```

---

## Training

All models trained with:
- **Optimizer:** Adam
- **Learning rate:** 1e-2 (GCN/SAGE), 5e-3 (GATv2)
- **Regularisation:** Layer Normalisation + Dropout (p=0.5)
- **Epochs:** 100
- **Hardware:** NVIDIA T4 GPU

---

## Key Insights

- **Identity preservation wins.** Concatenating a node's own features with its aggregated neighbourhood (GraphSAGE) consistently outperforms symmetric blending (GCN).
- **Attention ≠ accuracy here.** GATv2's edge-wise attention scoring adds ~2.5× training time with no accuracy gain, suggesting topology is a stronger signal than learned edge weights for this graph.
- **The transductive gap is real.** A consistent ~7% drop from validation to test accuracy reflects the difficulty of generalising from historical (≤2017) citation patterns to future (2019+) research categories.
- **Over-smoothing limits depth.** Beyond 3 layers, test accuracy degrades as node representations become indistinguishable from their neighbours.

---

## Limitations

- Cold-start: zero-citation papers have no neighbourhood context
- Static embeddings: word2vec features cannot adapt to abstract content
- Over-smoothing bottleneck at 3 layers limits receptive field

## Future Work

- Temporal GNNs weighting recent citations more heavily
- Label propagation using known neighbour labels as auxiliary features
- Mini-batch neighbour sampling for scaling to `ogb-papers100M`
- Graph Transformers to close the ~15% gap to SOTA (~71%)

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-latest-red)
![OGB](https://img.shields.io/badge/OGB-benchmark-green)

---

## References

1. Kipf & Welling (2017) — [Semi-supervised classification with GCNs](https://arxiv.org/abs/1609.02907)
2. Hamilton et al. (2017) — Inductive Representation Learning on Large Graphs (NeurIPS)
3. Veličković et al. (2018) — Graph Attention Networks (ICLR)
4. Brody et al. (2022) — How Attentive Are Graph Attention Networks? (ICLR)
5. Hu et al. (2020) — Open Graph Benchmark (NeurIPS)
