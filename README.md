# Swiggy Annual Report QA System

RAG-based question answering system for the **Swiggy Annual Report FY 2023-24**. Extracts text from a 170-page PDF (including scanned pages via OCR), builds a vector search index, and uses Google Gemini to answer questions with page citations.

## How it Works

```
PDF (170 pages)
    |
    v
PDF Loader (PyMuPDF + Tesseract OCR)
    |
    v
Text Chunker (overlapping word chunks)
    |
    v
Embeddings (sentence-transformers, all-MiniLM-L6-v2, 384-dim)
    |
    v
FAISS Index (saved to disk)
    |
    v  (query time)
Semantic Search -> Top-K chunks
    |
    v
Prompt + Anti-hallucination instructions
    |
    v
Google Gemini -> Answer with page citations
```

## Tech Stack

- **PDF extraction:** PyMuPDF + Tesseract OCR (for scanned pages)
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector DB:** FAISS (L2 distance)
- **LLM:** Google Gemini (with model fallback)
- **Frontend:** Streamlit
- **Language:** Python 3.10+

## Files

```
project/
├── data/
│   └── Swiggy Annual-Report-FY-2023-24.pdf
├── src/
│   ├── pdf_loader.py      - text extraction + OCR
│   ├── text_chunker.py    - cleaning & chunking
│   ├── embeddings.py      - vector embeddings
│   ├── vector_store.py    - FAISS index ops
│   ├── rag_pipeline.py    - pipeline orchestration
│   └── query_engine.py    - QA with Gemini
├── vector_db/             - saved FAISS index (auto-generated)
├── app.py                 - Streamlit web UI
├── main.py                - CLI entry point
├── evaluation.ipynb       - step-by-step pipeline demo
├── requirements.txt
└── README.md
```

## Setup

**Prerequisites:**
- Python 3.10+
- Tesseract OCR ([download here](https://github.com/UB-Mannheim/tesseract/wiki) for Windows)
- Google Gemini API key ([get one here](https://aistudio.google.com/apikey))

**Install:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Put the PDF in `data/` folder.

**Run CLI:**
```bash
python main.py
```

**Run Web UI:**
```bash
streamlit run app.py
```

First run takes a few minutes (OCR on ~140 scanned pages). After that the index is cached and startup is fast.

## Anti-Hallucination

The system is designed to not make stuff up:
- The prompt tells Gemini to ONLY use the retrieved context
- If the answer isn't in the document, it says "I don't know based on the document."
- Answers include page numbers so you can verify
- Only top-K relevant chunks go to the LLM

## Example Outputs

**Q: What was Swiggy's total revenue in FY 2023-24?**
> Swiggy's total revenue from operations in FY 2023-24 was **₹1,12,473.90 Million** (Page 151).
> Sources: 46, 83, 151, 152, 153

**Q: Who is the CEO of Swiggy?**
> **Sriharsha Majety** is the Managing Director and Group CEO (Page 159).
> Sources: 1, 51, 145, 159, 170

**Q: What was Swiggy's total loss in FY24?**
> Loss for FY24 was **(₹23,502.43) Million** (Page 112, 154).
> Sources: 8, 86, 112, 154, 170

**Q: What are the key subsidiaries of Swiggy?**
> Scootsy Logistics, Supr Infotech Solutions, and Lynks Logistics (Page 116).
> Sources: 51, 92, 116, 145, 170

**Q: What was Swiggy's revenue in FY 2025?** *(hallucination test)*
> I don't know based on the document.
> *(Correct — FY 2025 data is not in the report)*

**Q: How many monthly transacting users does Swiggy have?**
> Approximately **14 million** average monthly transacting users in FY24 (Page 8).
> Sources: 8, 51, 83, 108, 170
