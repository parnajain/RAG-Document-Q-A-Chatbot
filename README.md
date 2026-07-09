# RAG Document Q&A Chatbot

A fully local, privacy-preserving Retrieval-Augmented Generation (RAG) system that lets you query your own PDF documents in natural language — no data ever leaves your machine.

Built with **LangChain**, **ChromaDB**, **HuggingFace embeddings**, and **Ollama (Llama 3.2)** for generation, with a **Streamlit** front-end. All retrieval, embedding, and generation happens locally, making it suitable for use cases involving sensitive or confidential documents where cloud-based AI tools aren't an option.

---

## Features

- **Upload PDFs and ask questions** in natural language through a simple Streamlit UI
- **Two selectable retrieval strategies**, switchable at runtime:
  - Chroma MMR (Maximal Marginal Relevance)
  - BM25 + Chroma Ensemble Retriever
- **Two selectable embedding models**: `sentence-transformers/all-MiniLM-L6-v2` and `BAAI/bge-base-en-v1.5`
- **Out-of-scope detection** — if the best retrieval score falls below a similarity threshold, the app returns "not present in the document" instead of letting the LLM hallucinate an answer
- **Summary-aware querying** — detects summary-type questions (e.g. "summarize this document") and routes them differently from the out-of-scope check
- **Custom local evaluation harness** scoring Faithfulness, Answer Relevancy, Context Quality, and Accuracy — no external API calls, even for evaluation

---

## How It Works

1. User uploads one or more PDFs through Streamlit
2. Documents are split into chunks (`RecursiveCharacterTextSplitter`, 1500 chars, 200 overlap)
3. Chunks are embedded using a HuggingFace sentence-transformer model and stored in a **Chroma** vector store (a separate collection per embedding model, to avoid dimension mismatches)
4. On a query, the selected retriever fetches the top-k relevant chunks
5. Retrieved context + question are passed to a local **Llama 3.2** model (via Ollama) using a strict prompt template that forces the model to answer only from context
6. If the top similarity score indicates the query is unrelated to the document, the app short-circuits and returns an out-of-scope message instead of calling the LLM

## Retrieval Strategies

| Strategy | How it works |
|---|---|
| **Chroma MMR** | Retrieves the top `fetch_k=30` chunks by similarity, then re-ranks to select `k=8` that balance relevance with diversity — reduces redundant, near-duplicate chunks in context |
| **BM25 + Chroma Ensemble** | Combines sparse keyword-based search (BM25, weight 0.4) with dense vector search (Chroma, weight 0.6) via LangChain's `EnsembleRetriever`, so exact keyword matches and semantic matches both contribute |

---

## Evaluation

Standard RAGAS tooling ran into dependency conflicts with the LangChain version used here, so a **custom RAGAS-style evaluator** was built from scratch — using the local Llama 3.2 model itself as an LLM judge, keeping the entire evaluation loop offline.

Each (embedding model × retrieval strategy) combination was benchmarked against a 30-question test set derived from an OS coursework document (disk scheduling algorithms, RAID), scoring:

- **Faithfulness** — is every claim in the answer grounded in the retrieved context?
- **Answer Relevancy** — does the answer address the question asked?
- **Context Quality** — was the retrieved context sufficient to answer the question at all?
- **Accuracy** — is the generated answer semantically equivalent to the ground-truth answer?

### Results

| Embedding Model | Retriever | Correct/Total | Accuracy | Faithfulness | Answer Relevancy | Context Quality | Avg. Latency (s) |
|---|---|---|---|---|---|---|---|
| BAAI/bge-base-en-v1.5 | Chroma MMR | 23.6/30 | 0.787 | 0.790 | 0.717 | 0.733 | 3.327 |
| BAAI/bge-base-en-v1.5 | Ensemble BM25 + Chroma | **24.0/30** | **0.800** | **0.800** | **0.750** | **0.750** | 2.649 |
| sentence-transformers/all-MiniLM-L6-v2 | Chroma MMR | 23.8/30 | 0.793 | **0.800** | 0.717 | 0.717 | **1.860** |
| sentence-transformers/all-MiniLM-L6-v2 | Ensemble BM25 + Chroma | **24.0/30** | **0.800** | **0.800** | 0.733 | 0.717 | 5.086 |

**Takeaway:** BAAI/bge-base-en-v1.5 + Ensemble (BM25 + Chroma) remains the strongest overall configuration — it ties for the best accuracy and faithfulness (with MiniLM + Ensemble) but leads outright on answer relevancy and context quality, at less than half the latency of its MiniLM counterpart. More broadly, hybrid retrieval (Ensemble) matched or outperformed Chroma MMR alone on accuracy for both embedding models, suggesting BM25's keyword matching complements dense semantic search on this document set. sentence-transformers/all-MiniLM-L6-v2 + Chroma MMR is now the fastest configuration at 1.86s, though it trails on accuracy and relevancy. Across all four configurations, Faithfulness consistently scores higher than Answer Relevancy and Context Quality (by roughly 5–8 points) — indicating that generation stays well-grounded in whatever context it receives, but retrieval isn't always surfacing the most relevant chunks.

---

## Shortcomings

- **Small evaluation set.** With 30 questions, each individual question is worth ~3.3 percentage points — differences like 0.780 vs. 0.800 accuracy represent a single question and aren't necessarily statistically significant.
- **Single-domain test set.** All benchmark questions are derived from one OS coursework document (disk scheduling, RAID). Results may not generalize to documents with different structure, length, or vocabulary.
- **Self-graded evaluation.** Faithfulness, Answer Relevancy, and Context Quality are all scored by the same local Llama 3.2 model used for generation, rather than an independent judge model — this can introduce self-consistency bias.
- **Latency is highly run-to-run variable.** Across evaluation runs, the latency ranking between the two embedding models flipped depending on retrieval strategy — `all-MiniLM-L6-v2` was faster than `bge-base-en-v1.5` under Chroma MMR (1.86s vs. 3.33s) but slower under BM25+Chroma Ensemble (5.09s vs. 2.65s). This inconsistency suggests latency here is more sensitive to system load, Ollama warm-up state, or caching effects than to the embedding model or retrieval strategy itself, and hasn't been isolated via component-level profiling (embedding vs. retrieval vs. generation time).
## Future Scope

- Expand the evaluation set beyond 30 questions and across multiple document domains for more statistically defensible, generalizable comparisons.
- Profile embedding vs. retrieval vs. generation latency separately to explain the MiniLM latency anomaly.
- Evaluate against an external, stronger judge model to reduce self-grading bias in the evaluation harness.
- Add a larger embedding model (e.g. `bge-large-en-v1.5`) to test whether embedding capacity is a bigger lever than retrieval strategy choice.
- Support additional file formats beyond PDF (e.g. `.docx`, `.txt`, `.md`).
- Add conversational memory for multi-turn follow-up questions.

---

## Tech Stack

`Python` · `LangChain` · `LangChain Community` · `ChromaDB` · `HuggingFace Embeddings` · `Ollama` · `Llama 3.2` · `Streamlit` · `BM25 (rank_bm25)` · `pandas`

---

## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed locally, with the Llama 3.2 model pulled:
  ```bash
  ollama pull llama3.2
  ```

### Installation

```bash
git clone https://github.com/parnajain/RAG-Document-Q-A-Chatbot.git
cd RAG-Document-Q-A-Chatbot

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

Upload a PDF, pick a retrieval strategy and embedding model, and start asking questions.

### Run the evaluation suite

```bash
python eval.py
```

This runs every (embedding × retriever) combination against `tests.csv` and writes results to `evaluation_results.csv` and `summary_results.csv`.

---

## Project Structure

```
.
├── app.py                  # Streamlit UI — upload, strategy selection, Q&A
├── rag_pipeline.py         # Core RAG logic: embeddings, vector store, retrievers, chain
├── eval.py                 # Custom evaluation harness (Faithfulness, Relevancy, Context Quality, Accuracy)
├── tests.csv               # Benchmark question / ground-truth pairs
├── evaluation_results.csv  # Per-question evaluation output
├── summary_results.csv     # Aggregated scores per (embedding, retriever) combination
├── requirements.txt
└── .gitignore
```

---

## Engineering Challenges Solved

- **Embedding-dimension mismatches** when switching between embedding models — resolved by giving each model its own Chroma collection/persist path
- **Windows-specific `asyncio`/Streamlit hang** — resolved via `WindowsSelectorEventLoopPolicy`
- **RAGAS dependency conflicts** with the installed LangChain version — resolved by building a custom, RAGAS-inspired local evaluator using the LLM itself as judge
- **OneDrive storage bloat** from a synced virtual environment — resolved by relocating `venv` outside the OneDrive-synced folder

---

## Notes

This project runs entirely offline after setup — no OpenAI/Anthropic/cloud API keys required. All embedding, retrieval, and generation happen on-device via Ollama and local HuggingFace models.
