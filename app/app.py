# --- IMPORTS AND ASYNCIO PATCH ---
# The nest_asyncio patch must be applied before any other imports, especially Streamlit and asyncio.
from langchain.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain import hub
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
from dotenv import load_dotenv
import os
import asyncio
import nest_asyncio
nest_asyncio.apply()


# --- GLOBAL INITIALIZATION ---
# Load environment variables from .env file for local development
load_dotenv()

st.set_page_config(
    page_title="Anwesha - A Bengali PDF RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use Streamlit's secrets for the API key when deployed
groq_api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY is not set. Please add it to your environment variables or Streamlit secrets.")
    st.stop()

# --- CACHED FUNCTIONS TO LOAD MODELS ONCE ---


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
    # Use a valid Groq model. Options: "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"
    return ChatGroq(model="llama3-70b-8192", api_key=_groq_api_key)


# --- LOAD ALL COMPONENTS ---
with st.spinner("Loading models and vector store... This may take a moment."):
    embeddings = load_embeddings()
    retriever = load_retriever(embeddings)
    llm = load_llm(groq_api_key)
    prompt_template = hub.pull("rlm/rag-prompt")

# --- RAG CHAIN DEFINITION ---


def format_docs(docs):
    """Formats retrieved documents into a single string."""
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
    """Fuses retrieved documents using Reciprocal Rank Fusion."""
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        loads(doc)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


final_rag_chain = prompt_template | llm | StrOutputParser()

# --- STREAMLIT UI DEFINITION ---

with st.sidebar:
    st.title("📚 Anwesha RAG System")
    st.markdown(
        "A chatbot to answer questions about Rabindranath Tagore's 'Aparichita'.")
    st.markdown("---")
    st.info("This is a production version of the Anwesha RAG chatbot, using multi-query retrieval and RRF.")

st.title("Anwesha Chatbot")
st.markdown("Ask me anything about 'Aparichita' in Bangla or English.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("আপনার প্রশ্ন লিখুন... (Write your question...)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                # 1. Generate multiple queries from the user's prompt
                queries = generate_queries.invoke({"question": prompt})

                # 2. Asynchronously retrieve documents for all queries in parallel
                retrieval_tasks = [retriever.ainvoke(q) for q in queries]
                retrieved_docs_lists = asyncio.run(
                    asyncio.gather(*retrieval_tasks))

                # 3. Rerank the collected documents using RRF
                reranked_docs = reciprocal_rank_fusion(retrieved_docs_lists)

                # 4. Format the context from the top reranked documents
                formatted_context = format_docs(reranked_docs)

                # 5. Invoke the final chain to generate a response
                assistant_response = final_rag_chain.invoke(
                    {"context": formatted_context, "question": prompt}
                )
                message_placeholder.markdown(assistant_response)

            except Exception as e:
                assistant_response = f"An error occurred: {e}"
                message_placeholder.error(assistant_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_response})
