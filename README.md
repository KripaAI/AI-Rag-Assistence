# Local Multimodal RAG (OpenAI + Pinecone + Gemini)

A high-performance, local-first Retrieval-Augmented Generation (RAG) system with multimodal support for PDF text, tables, and images.

---

## 🚀 Key Features

*   **Multimodal Extraction**: Automatically parses PDFs to extract text chunks, identifies tables (converted to Markdown), and embedded images.
*   **Vision-Enhanced Retrieval**: Uses OpenAI Vision (GPT-4o-mini) to generate technical captions for images, making them searchable via text queries.
*   **Hybrid Retrieval (RRF)**: Employs Reciprocal Rank Fusion (RRF) to combine results from text, table, and image modalities.
*   **Sophisticated Reranking**: Applies a weighted rerank score based on vector similarity, keyword overlap, and "source routing" (prioritizing specific documents based on the query).
*   **Grounded Generation**: Enforces absolute faithfulness to the provided context using GPT-4o with a "Closed World Assumption" and mandatory inline citations (e.g., `[C1]`).
*   **Automated Benchmarking**: Includes a full RAGAS evaluation suite (powered by Gemini) to calculate Faithfulness, Answer Relevancy, and Context Precision.
*   **Modern UI & CLI**: Choose between a robust CLI for headless operations or a modern Streamlit dashboard for visual evidence side-by-side with answers.

---

## 🛠️ Architecture Overview

The system is organized into four distinct layers:

1.  **Ingestion (`src/ingestion/`)**: PDF parsing, image captioning, and Pinecone vectorization.
2.  **Retrieval (`src/retrieval/`)**: Multi-stage RRF and semantic reranking.
3.  **Generation (`src/generation/`)**: Grounded answering and citation validation.
4.  **Evaluation (`src/evaluation/`)**: Automated RAGAS benchmarking and log generation.

---

## 🏁 Quick Start

### 1) Prerequisites

*   Python 3.10+
*   OpenAI API Key
*   Pinecone API Key (Serverless Index)
*   Gemini API Key (Required for `evaluate`)

### 2) Installation

```powershell
# Clone the repository and navigate to the project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3) Configuration

Copy the example environment file and provide your credentials:

```powershell
cp .env.example .env
```

Edit `.env` to set your API keys and preferred model parameters.

### 4) Ingest Documents

Place your PDFs in the `data/` folder and run the ingestion pipeline:

```powershell
python -m src.main ingest
```

### 5) Ask a Question (CLI)

```powershell
python -m src.main query "How does Reciprocal Rank Fusion work in this system?"
```

### 6) Launch the UI

```powershell
streamlit run src/app.py
```

---

## 📊 Evaluation & Benchmarking

To run the automated RAGAS benchmark on your dataset:

```powershell
python -m src.main evaluate --dataset eval_dataset.jsonl
```

Detailed reports (CSV and JSON) will be generated in the `logs/evaluation/` directory.

---

## 🧪 Testing

The project includes a robust testing suite. Run tests using `pytest`:

```powershell
python -m pytest -v
```

---

## 🏗️ Project Structure

```text
├── data/                    # Input PDFs
├── data/processed/          # Extracted assets and manifests
├── src/
│   ├── ingestion/           # ETL and Indexing
│   ├── retrieval/           # RRF and Reranking
│   ├── generation/          # Grounded Answer Production
│   └── evaluation/          # RAGAS Metrics
├── tests/                   # Pytest suite
├── app.py                   # Streamlit UI
├── main.py                  # CLI Orchestrator
└── README.md                # You are here!
```

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🐞 Bugs & Feedback

Found a bug or have a suggestion? Open an issue or use the `/bug` command in the Gemini CLI!
