import faiss
import pickle
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

#loading faiss and metadata from buildindex
index = faiss.read_index("faiss.index")
with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

#embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

#llm
llm = Llama(
    model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",  # Change path if needed
    n_ctx=4096,
    n_gpu_layers=20,
    verbose=False
)

#chat function
def ask_question(query, top_k=5):
    query_vec = embed_model.encode([query])
    scores, indices = index.search(query_vec, top_k)

    #context for api
    context = ""
    sources = []
    for idx in indices[0]:
        meta = metadata[idx]
        context += f"\n---\nFrom {meta['file']} (Page {meta['page']}):\n{meta['chunk_text']}"
        sources.append(f"{meta['file']} (Page {meta['page']})")

    #user prompt
    prompt = f"""You are a helpful assistant. Answer the question using only the provided expert call transcripts.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {query}
Answer:"""

    response = llm(prompt=prompt, stop=["\n\n", "User:"], max_tokens=512)
    answer = response['choices'][0]['text'].strip()

    return answer, sources


#asking
if __name__ == "__main__":
    print("🔍 Chat Here!")
    while True:
        q = input("\nAsk a question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        answer, source_refs = ask_question(q)
        print("\nAnswer:\n", answer)
        print("\nSources:\n", "\n".join(source_refs))
