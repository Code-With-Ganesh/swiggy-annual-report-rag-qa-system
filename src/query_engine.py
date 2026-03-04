# query_engine.py - handles QA: embed question -> search FAISS -> prompt Gemini -> return answer

import os, sys, time
import numpy as np
from dotenv import load_dotenv

# load .env from project root
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from embeddings import load_model
from vector_store import load_from_disk, find_similar

from google import genai
from google.genai import errors as genai_errors

# read key from environment
API_KEY = os.getenv("GEMINI_API_KEY", "")

RETRY_DELAY = 3

# try these models in order, if one is rate limited move to next
MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

SYS_PROMPT = """You are a helpful assistant that answers questions about the Swiggy Annual Report (FY 2023-24).

STRICT RULES:
1. Answer ONLY using the context provided below.
2. If the answer is not found in the context, respond exactly: "I don't know based on the document."
3. Do NOT make up or infer information beyond what is in the context.
4. When possible, mention the page number(s) where the information was found.
5. Keep your answers clear and concise.
"""

def make_prompt(question, ctx_chunks):
    """puts together the prompt with context chunks and the question"""
    parts = []
    for c in ctx_chunks:
        parts.append(f"[Page {c['page_num']}]\n{c['text']}")
    ctx = "\n\n---\n\n".join(parts)
    return f"""{SYS_PROMPT}

CONTEXT FROM DOCUMENT:
{ctx}

QUESTION: {question}

ANSWER:"""

def ask(question, client, index, chunks, embed_model, top_k=5):
    """main QA function - embeds question, searches index, calls gemini"""
    qvec = embed_model.encode([question])
    qvec = np.array(qvec, dtype="float32")

    results = find_similar(qvec, index, chunks, top_k=top_k)
    prompt = make_prompt(question, results)

    # try each model until one works
    answer = None
    for mname in MODELS:
        try:
            resp = client.models.generate_content(model=mname, contents=prompt)
            answer = resp.text.strip()
            break
        except genai_errors.ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"{mname} rate-limited, trying next...")
                time.sleep(RETRY_DELAY)
                continue
            raise

    if answer is None:
        answer = "All models are currently rate-limited. Please wait a minute and try again."

    pages = sorted(set(r["page_num"] for r in results))
    return {"answer": answer, "sources": pages, "context": results}

def init_gemini(api_key=None):
    """sets up the gemini client"""
    key = api_key or API_KEY
    client = genai.Client(api_key=key)
    print(f"Gemini ready (models: {' > '.join(MODELS)})")
    return client
