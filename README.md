# AmbedkarGPT – SemRAG-based RAG System

A **Retrieval-Augmented Generation (RAG)** system built by strictly following the **SemRAG (Semantic Knowledge-Augmented RAG)** research paper. This project answers questions about **Dr. B. R. Ambedkar’s works** using semantic chunking, knowledge graphs, and local LLMs.

This repository is prepared as part of the **AI Engineering Intern – Technical Assignment** and is fully runnable **locally for live demonstration**.

---

## 🚀 Features

* ✅ Semantic Chunking using **cosine similarity** (Algorithm 1 – SemRAG)
* ✅ Buffer merging to preserve contextual continuity
* ✅ Token-aware chunking (1024 max tokens, 128 overlap)
* ✅ Knowledge Graph construction (entities + relationships)
* ✅ Community detection (Louvain / Leiden)
* ✅ Local GraphRAG Search (Equation 4)
* ✅ Global GraphRAG Search (Equation 5)
* ✅ Local LLM integration (Llama3 / Mistral via Ollama)
* ✅ End-to-end Q&A pipeline with citations

---

## 🧠 Architecture Overview (SemRAG)

```text
PDF → Semantic Chunking → Entity & Relation Extraction
   → Knowledge Graph → Community Detection
   → Local Search (Eq. 4) + Global Search (Eq. 5)
   → Prompt Construction → Local LLM → Answer
```

The implementation closely follows **Sections 3.2.1 – 3.2.3** of the SemRAG paper.

---

## 📁 Project Structure

```text
ambedkargpt/
├── data/
│   ├── Ambedkar_works.pdf
│   └── processed/
│       ├── chunks.json
│       └── knowledge_graph.pkl
├── src/
│   ├── chunking/
│   │   ├── semantic_chunker.py    # Algorithm 1
│   │   └── buffer_merger.py
│   ├── graph/
│   │   ├── entity_extractor.py
│   │   ├── graph_builder.py
│   │   ├── community_detector.py
│   │   └── summarizer.py
│   ├── retrieval/
│   │   ├── local_search.py        # Equation 4
│   │   ├── global_search.py       # Equation 5
│   │   └── ranker.py
│   ├── llm/
│   │   ├── llm_client.py
│   │   ├── prompt_templates.py
│   │   └── answer_generator.py
│   └── pipeline/
│       └── ambedkargpt.py         # Main pipeline
├── tests/
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_integration.py
├── config.yaml
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🛠️ Tech Stack

* **Python** 3.9+
* **sentence-transformers** (all-MiniLM-L6-v2)
* **spaCy** (NER + dependency parsing)
* **networkx** (knowledge graph)
* **python-louvain / leidenalg** (community detection)
* **Ollama** (local LLM runtime)
* **Llama3-8B / Mistral-7B** (local inference)

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone <private-repo-url>
cd ambedkargpt
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 5️⃣ Setup Local LLM (Ollama)

Install Ollama: [https://ollama.ai](https://ollama.ai)

Pull model:

```bash
ollama pull llama3:8b
# or
ollama pull mistral:7b
```

---

## ▶️ Running the System

### End-to-End Pipeline

```bash
python src/pipeline/ambedkargpt.py
```

Example query:

```text
What were Dr. B. R. Ambedkar's views on social justice?
```

---

## 🔍 Retrieval Methods Implemented

### 🔹 Local GraphRAG Search (Equation 4)

* Matches query → entities → related chunks
* Filters using similarity thresholds τₑ and τ𝒹

### 🔹 Global GraphRAG Search (Equation 5)

* Retrieves top-K community summaries
* Scores sub-points within communities

Both methods are combined during answer generation.

---

## 🧪 Testing

Run all tests:

```bash
pytest tests/
```

---

## 📌 Configuration

Key parameters are configurable in `config.yaml`:

* Buffer size
* Similarity thresholds
* Top-K retrieval limits
* LLM model selection

---

## 🧾 Notes for Live Interview Demo

* System runs **fully offline**
* No external APIs used
* Knowledge graph is built locally
* LLM inference is local (Ollama)

Demo flow:

1. Load PDF
2. Perform semantic chunking
3. Build knowledge graph
4. Ask 3–5 questions
5. Show retrieved context + answer

---

## 📚 Reference

* **SemRAG: Semantic Knowledge-Augmented RAG for Improved Question-Answering**
* Zhong et al., 2025

---

## 👤 Author

**Prajwal Patil**
B.E. Computer Science and Engineering
AI Engineering Intern Candidate
