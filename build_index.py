import os
import fitz 
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle


#load and chunk pdfs
def load_and_chunk_pdfs(pdf_dir, chunk_size=500, overlap=100):
    all_chunks = []
    metadata = []
    for filename in sorted(os.listdir(pdf_dir)):
        if filename.endswith(".pdf"):
            path = os.path.join(pdf_dir, filename)
            doc = fitz.open(path)
            for page_num, page in enumerate(doc):
                text = page.get_text()
                # Clean and split into chunks
                words = text.split()
                for i in range(0, len(words), chunk_size - overlap):
                    chunk = " ".join(words[i:i + chunk_size])
                    all_chunks.append(chunk)
                    metadata.append({
                        "file": filename,
                        "page": page_num + 1,
                        "chunk_text": chunk
                    })
            doc.close()
    return all_chunks, metadata

pdf_dir = "./Q2"  #directory
chunks, metadata = load_and_chunk_pdfs(pdf_dir)


#word embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")  #lightweight
embeddings = model.encode(chunks, show_progress_bar=True)

#faiss index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

#saving metadata for use
faiss.write_index(index, "faiss.index")
with open("metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("✅ Index and metadata saved.")
