import os
import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Get API key securely
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please set it in your .env file.")

GROQ_MODEL = "llama3-8b-8192"

# Sample documents
documents = [
    "The Eiffel Tower is located in Paris, France.",
    "Python is a popular programming language for AI.",
    "Groq is a platform that runs LLMs at blazing speed.",
    "The capital of Japan is Tokyo."
]

# Initialize embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")
document_embeddings = embedder.encode(documents, convert_to_numpy=True)

# FAISS indexing
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

# Query
query = "Where is the Eiffel Tower?"
query_embedding = embedder.encode([query], convert_to_numpy=True)

# Search for relevant chunks
k = 2
distances, indices = index.search(query_embedding, k)
retrieved_chunks = [documents[i] for i in indices[0]]

# Create prompt with retrieved context
context = "\n".join(retrieved_chunks)
prompt = f"""Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""

# Function to query Groq API
def query_groq(prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    response = requests.post(url, headers=headers, json=payload)
    
    # Improved error handling
    if response.status_code != 200:
        raise Exception(f"Groq API Error {response.status_code}: {response.text}")
    
    return response.json()["choices"][0]["message"]["content"]

# Execute and print result
try:
    answer = query_groq(prompt)
    print("Answer:", answer)
except Exception as e:
    print("❌ Failed to get a response:", str(e))