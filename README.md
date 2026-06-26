# AI Music Generation with Recurrent Neural Networks

**Course:** CSE 153 — Introduction to Music Information Retrieval, UC San Diego  
**Authors:** Jay Gao, Billy Dang, Ryan Park, Muddassir Ahmadzada

---

## Overview

This project implements symbolic music generation using recurrent neural networks trained on classical piano MIDI data. We explore both **unconditional generation** (generating music from a random seed) and **conditional generation** (generating music based on a given musical context), comparing RNN, GRU, and LSTM architectures.

---

## Dataset

- **MAESTRO** (MIDI and Audio Edited for Synchronous TRacks and Organizations) — ~200 hours of classical piano performances from the International Piano-e-Competition, recorded using Yamaha Disklaviers with high-resolution MIDI capture
- Supplementary tracks provided by course instructor
- Subset used: performances from the "2004" folder, 90/10 train/test split

**Tokenization:** REMI (Revamped MIDI-derived events) via the `MidiTok` library — represents notes as successions of Pitch, Velocity, and Duration tokens with Bar and Position tokens for timing.

---

## Models

We implemented and compared three architectures:

| Model | Notes |
|-------|-------|
| **RNN** | Fastest to train; struggled with long-term musical dependencies |
| **GRU** | Fewer parameters than LSTM; did not learn musical structure effectively |
| **LSTM** | Best performance; captured long-term dependencies with Forget, Input, and Output Gates |

Key hyperparameters tuned: `embedding_dim`, `hidden_dim`, `num_layers`, `dropout`, `learning_rate`, `batch_size`, `max_seq_len`

Training was accelerated using CUDA GPU computation; CPU-only training exceeded 30 minutes on small subsets.

---

## Evaluation

- **Perplexity** measured against Uniform and Unigram baselines — our LSTM significantly outperformed both, confirming the model learned meaningful musical structure
- Evaluated pitch transition smoothness, note sequence quality, and token frequency distributions
- Training and validation loss monitored across epochs

---

## Results

Generated MIDI outputs are included in this repository:
- `symbolic_unconditioned.mid` — music generated without any conditioning input
- `symbolic_conditioned.mid` — music generated with a musical context seed

---

## Files

| File | Description |
|------|-------------|
| `workbook.html` | Full project notebook with code, analysis, and visualizations |
| `CSE153 - Final Project.pdf` | Project report / presentation slides |
| `symbolic_unconditioned.mid` | Generated MIDI: unconditional model output |
| `symbolic_conditioned.mid` | Generated MIDI: conditional model output |

---

## Tools & Libraries

- **Language:** Python
- **Libraries:** PyTorch, MidiTok, pretty_midi
- **Hardware:** CUDA GPU
- **Methods:** RNN, GRU, LSTM, REMI tokenization, cross-entropy loss, Adam optimizer
