from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import streamlit as st
import asyncio
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from operator import itemgetter

# --- GLOBAL INITIALIZATION (Load models once) ---

# Load environment variables from .env file for local development
# On Streamlit Cloud, these will be set as Secrets
load_dotenv()
st.set_page_config(
    page_title="Anwesha-A Bengali PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use Streamlit's secrets for the API key when deployed
groq_api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY is not set. Please add it to your environment variables or Streamlit secrets.")
    st.stop()

# --- Cached functions to load models and data once ---


@st.cache_resource
def load_embeddings():
    """Load the embedding model."""
    return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")


@st.cache_resource
def load_retriever(_embeddings):
    """Load the vector store and retriever."""
    persist_directory = "database/anwesha_faiss_db"
    if not os.path.exists(persist_directory):
        st.error(
            f"FAISS directory not found at '{persist_directory}'. Please ensure the database directory is in your repository and the path is correct."
        )
        st.stop()

    vectorstore = FAISS.load_local(
        folder_path=persist_directory,
        embeddings=_embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever()


@st.cache_resource
def load_llm(_groq_api_key):
    """Load the Language Model."""
    return ChatGroq(model="moonshotai/kimi-k2-instruct", api_key=_groq_api_key)


# --- Load all components ---
with st.spinner("Loading models and vector store..."):
    embeddings = load_embeddings()
    retriever = load_retriever(embeddings)
    llm = load_llm(groq_api_key)
    prompt_template = hub.pull("rlm/rag-prompt")


# --- RAG CHAIN DEFINITION ---

def format_docs(docs):
    """Formats retrieved documents into a single string."""
    if isinstance(docs, list) and docs and isinstance(docs[0], tuple):
        return "\n\n".join(doc.page_content for doc, score in docs)
    return "\n\n".join(doc.page_content for doc in docs)


multi_query_template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(multi_query_template)

generate_queries = (
    prompt_perspectives
    | llm
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)


def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


# Define only the final part of the chain that combines context and question
final_rag_chain = prompt_template | llm | StrOutputParser()


# --- STREAMLIT UI DEFINITION ---

with st.sidebar:
    st.title("📚 Anwesha RAG System")
    st.markdown(
        "A chatbot to answer questions about Rabindranath Tagore's 'Aparichita'.")
    st.markdown("---")
    st.info("This is a production version of the Anwesha RAG chatbot.")

st.title("Anwesha Chatbot")
st.markdown("Ask me anything about 'Aparichita' in Bangla or English.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("আপনার প্রশ্ন লিখুন..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                # 1. Generate multiple queries
                queries = generate_queries.invoke({"question": prompt})

                # 2. Retrieve documents for each query synchronously
                retrieved_docs = []
                for q in queries:
                    # Use the standard .invoke() which is synchronous
                    retrieved_docs.append(retriever.invoke(q))

                # 3. Rerank the results using Reciprocal Rank Fusion
                reranked_results = reciprocal_rank_fusion(retrieved_docs)

                # 4. Format the final context
                formatted_context = format_docs(reranked_results)

                # 5. Invoke the final chain with the prepared context
                assistant_response = final_rag_chain.invoke(
                    {"context": formatted_context, "question": prompt}
                )
                message_placeholder.markdown(assistant_response)

            except Exception as e:
                assistant_response = f"An error occurred: {e}"
                message_placeholder.error(assistant_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_response})
