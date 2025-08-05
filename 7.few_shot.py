import os
import sys
import faiss
import requests
import numpy as np
from typing import List
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer

# === CONFIG === #
# from dotenv import load_dotenv
# load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY="PUT_API_KEY_HERE"
GROQ_MODEL = "llama3-8b-8192"
CHUNK_SIZE = 100
TOP_K = 3
# =============== #

def extract_text_from_pdf(pdf_path: str) -> str:
    return extract_text(pdf_path)

def split_text(text: str, max_words: int = 100) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def build_faiss_index(chunks: List[str], embedder):
    valid_chunks = [c for c in chunks if c.strip()]
    if not valid_chunks:
        raise ValueError("No valid text chunks found.")
    embeddings = embedder.encode(valid_chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, valid_chunks

def retrieve_relevant_chunks(query: str, embedder, index, chunks, k=TOP_K):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

def query_groq(prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert assistant that analyzes resumes and answers questions based only on provided context."
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def run_resume_rag(pdf_path: str, query: str):
    print(f"📄 Reading resume: {pdf_path}")
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = split_text(raw_text, max_words=CHUNK_SIZE)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    index, chunk_list = build_faiss_index(chunks, embedder)
    top_chunks = retrieve_relevant_chunks(query, embedder, index, chunk_list, k=TOP_K)

    actual_context = "\n".join(top_chunks)

    # === Few-Shot Prompt ===
    prompt = f"""
You are analyzing a resume to extract specific information.

Example 1:
Context:
John has 5 years of experience as a backend developer at XYZ Corp. He worked with Node.js, PostgreSQL, and AWS.
Question:
What is the candidate's experience?
Answer:
The candidate has 5 years of backend development experience using Node.js, PostgreSQL, and AWS.

Example 2:
Context:
Jane completed her B.Tech in Computer Science from NIT Trichy in 2020.
Question:
What is the candidate's educational qualification?
Answer:
The candidate holds a B.Tech in Computer Science from NIT Trichy, completed in 2020.

Example 3:
Context:
Ajay is skilled in Python, SQL, and React.js. He also has hands-on experience with Docker and Git.
Question:
List the candidate's technical skills.
Answer:
Python, SQL, React.js, Docker, Git

---

Context:
{actual_context}

Question:
{query}

Answer:
"""

    print("💬 Querying Groq with few-shot prompt...")
    return query_groq(prompt)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:\n  python rag_resume_groq_fewshot.py <resume.pdf> <question>")
        sys.exit(1)

    resume_pdf = sys.argv[1]
    query = " ".join(sys.argv[2:])

    if not os.path.exists(resume_pdf):
        print("❌ File not found.")
        sys.exit(1)

    answer = run_resume_rag(resume_pdf, query)
    print("\n🧠 Answer:", answer)
