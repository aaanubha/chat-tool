import streamlit as st
#set streamlit call
st.set_page_config(page_title="Expert Call Chat", layout="wide") 

import faiss
import pickle
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

#loading metadata
@st.cache_resource
def load_models():
    index = faiss.read_index("faiss.index")
    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    llm = Llama(
        model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf", #lightweight
        n_ctx=4096,
        n_gpu_layers=20,
        verbose=False
    )
    return index, metadata, embed_model, llm

index, metadata, embed_model, llm = load_models()

#answering
def ask_question_streamlit(query, top_k=5):
    query_vec = embed_model.encode([query])
    scores, indices = index.search(query_vec, top_k)

    context = ""
    sources = []
    for idx in indices[0]:
        meta = metadata[idx]
        context += f"\n---\nFrom {meta['file']} (Page {meta['page']}):\n{meta['chunk_text']}"
        sources.append(f"{meta['file']} (Page {meta['page']})")

    prompt = f"""You are a helpful assistant. Answer the question using only the provided expert call transcripts.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {query}
Answer:"""

    response = llm(prompt=prompt, stop=["\n\n", "User:"], max_tokens=512)
    answer = response['choices'][0]['text'].strip()

    return answer, sources

#streamlit
st.title("🧠 Chat Tool!")

query = st.text_input("Ask a question based on expert call transcripts:", placeholder="e.g. What are the AV plans?")

if query:
    with st.spinner("Generating answer..."):
        answer, sources = ask_question_streamlit(query)
        st.markdown("### ✅ Answer")
        st.write(answer)

        st.markdown("### 📚 Sources")
        for s in sources:
            st.write(f"- {s}")
