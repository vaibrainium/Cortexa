# Cortexa

**A domain-tuned LLM research assistant for decision neuroscience.**

Decision neuroscience is fragmented across subfields that rarely cite each other — perceptual decision-making, reinforcement learning, value-based decisions, computational modeling, and subcortical systems each have their own vocabulary, landmark papers, and theoretical frameworks. Researchers working at the intersections of these fields spend significant time manually tracking connections, gaps, and contradictions across a literature that no single review covers.

Cortexa is built to close that gap. It combines a fine-tuned language model with retrieval-augmented generation (RAG) over a curated corpus of ~45 anchor papers spanning 8 subfields. Every output is grounded in retrieved source chunks, so claims are traceable rather than hallucinated. The system operates in two modes: a conversational brainstorming partner for exploring cross-subfield connections, and a structured literature synthesizer that surfaces agreements, tensions, and testable hypotheses across communities.

Built and used in active PhD research on Bayesian priors and Superior Colliculus function (Basso Lab and Walker Lab, UCLA/UW).

---

## What it does

- Answers cross-subfield questions that no single paper or review addresses directly (e.g., how does the drift-diffusion model relate to dopamine-based reinforcement learning? what does the Superior Colliculus contribute to evidence accumulation?)
- Surfaces gaps in the literature across subfields, including connections that are biologically motivated but rarely cited together
- Generates testable, specific hypotheses grounded in retrieved evidence
- Cites source chunks for every claim so outputs can be verified against the original papers
- Falls back to live PubMed / Semantic Scholar search when a query exceeds the local corpus (gated, opt-in)

---

## Stack

| Component | Choice | Notes |
|---|---|---|
| Base model | Mistral 7B / Llama 3.1 8B | Quality/cost tradeoff for HF Spaces hosting |
| Fine-tuning | QLoRA via unsloth | Single A100, overnight runs |
| Alignment | ORPO | Single-pass preference optimization |
| Embeddings | BGE-M3 (dense, 1024-dim) | Outperforms OpenAI Ada on scientific text |
| Vector store | Qdrant | Native dense + sparse support, disk-persistent |
| Retrieval (MVP) | Dense-only BGE-M3 retrieval | Covers most brainstorming queries |
| Retrieval (v2) | Hybrid BM25 + dense + RRF + BGE reranker + HyDE | Handles exact references and abstract queries |
| Orchestration | LangChain | RAG chain + conversation memory |
| UI | Gradio | Three-tab interface, source chunk display |
| Deployment | Hugging Face Spaces | Free GPU, public URL |
| Eval | BERTScore (MVP), G-Eval LLM-as-judge (v2) | Quantitative scores on held-out Q&A pairs |
| Retraining | GitHub Actions + HF Datasets | Monthly pipeline on new PubMed/arXiv papers |

---

## Corpus

8 subfields, ~45 anchor papers:

1. Value-based decision making
2. Perceptual decision making
3. Reinforcement learning in the brain
4. Drift-diffusion and computational models
5. Neuroeconomics
6. Cognitive control and executive function
7. Confidence and metacognition
8. Superior Colliculus and subcortical decision making

Bridge papers — those that already reason across subfield boundaries — are weighted most heavily in fine-tuning data, since they provide the best signal for cross-subfield synthesis.

---

## Project status

- [x] S1: Docker environment, paper corpus download, PDF ingestion pipeline (PyMuPDF + tiktoken chunking, 25,817 chunks)
- [x] S2: BGE-M3 embeddings, Qdrant vector store, dense retrieval MVP
- [ ] S3: Instruction pairs dataset (cross-subfield Q&A, gap-finding, hypothesis generation)
- [ ] S4: QLoRA fine-tuning config, SLURM cluster job
- [ ] S5-8: Training eval, LangChain RAG chain, Gradio UI, HF Spaces deploy
- [ ] S9-12: Hybrid retrieval (BM25 + RRF + reranker + HyDE + gated CRAG), ORPO alignment
- [ ] S13+: Multi-query chain-of-papers retrieval, automated retraining pipeline, G-Eval
