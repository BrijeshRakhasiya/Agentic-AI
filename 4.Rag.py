import os
import sys
import faiss
import numpy as np
import requests
from typing import List
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer

# ========== CONFIG ========== #
# from dotenv import load_dotenv
# load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY="PUT_API_KEY_HERE"
GROQ_MODEL = "llama3-8b-8192"
CHUNK_SIZE = 100  # words per chunk
TOP_K = 3         # top chunks to retrieve
# ============================ #

# Extract raw text from PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    return extract_text(pdf_path)

# Split text into overlapping chunks
def split_text(text: str, max_words: int = 100) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# Embed and index with FAISS
def build_faiss_index(chunks: List[str], embedder):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, chunks

# Retrieve top-k relevant chunks
def retrieve_relevant_chunks(query: str, embedder, index, chunks, k=3):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]

# Query Groq API
def query_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert assistant. Answer only using the provided context."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    try:
        res = requests.post(url, headers=headers, json=data)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        if res.status_code == 401:
            print(f"❌ Authentication failed. Please check your GROQ API key.")
            print(f"Current API key starts with: {GROQ_API_KEY[:10]}...")
            print("Get a new API key from: https://console.groq.com/")
        raise e

# RAG pipeline
def run_rag(pdf_path: str, query: str):
    print(f"📄 Reading PDF: {pdf_path}")
    raw_text = extract_text_from_pdf(pdf_path)

    print(f"📚 Splitting text into chunks...")
    chunks = split_text(raw_text, max_words=CHUNK_SIZE)

    print(f"🔗 Embedding and building FAISS index...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    index, chunk_list = build_faiss_index(chunks, embedder)

    print(f"🔍 Retrieving top {TOP_K} relevant chunks...")
    relevant_chunks = retrieve_relevant_chunks(query, embedder, index, chunk_list, k=TOP_K)

    context = "\n".join(relevant_chunks)
    prompt = f"""Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""

    print(f"💬 Querying Groq API...")
    answer = query_groq(prompt)
    return answer

# Main entry
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:\n  python 4.Rag.py <path_to_pdf> <query>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    question = " ".join(sys.argv[2:])

    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    result = run_rag(pdf_path, question)
    print("\n🧠 Answer:", result)
