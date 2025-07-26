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
# 'operator.itemgetter' was imported but not used.

# --- GLOBAL INITIALIZATION (Load models once) ---

# Load environment variables from .env file for local development
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
    # CORRECTED: Use a valid Groq model name.
    # Other options: "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"
    return ChatGroq(model="llama3-70b-8192", api_key=_groq_api_key)


# --- Load all components ---
with st.spinner("Loading models and vector store..."):
    embeddings = load_embeddings()
    retriever = load_retriever(embeddings)
    llm = load_llm(groq_api_key)
    prompt_template = hub.pull("rlm/rag-prompt")

# --- RAG CHAIN DEFINITION ---


def format_docs(docs):
    """Formats retrieved documents into a single string."""
    # This function is fine, but handling the RRF output directly is cleaner.
    # The reranked_results now directly contains Document objects.
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
    # Iterate through each list of retrieved documents
    for docs in results:
        # Iterate through each document in the list with its rank
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Calculate RRF score
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order
    reranked_results = [
        loads(doc)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


# Define the final part of the chain that combines context and question
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
                # Generate multiple queries
                queries = generate_queries.invoke({"question": prompt})

                # --- IMPROVEMENT: Run retrievals in parallel ---
                # Use the asynchronous 'ainvoke' method for concurrent execution
                retrieval_tasks = [retriever.ainvoke(q) for q in queries]
                # Gather results
                retrieved_docs_lists = asyncio.run(
                    asyncio.gather(*retrieval_tasks))

                # Rerank the results using RRF
                reranked_docs = reciprocal_rank_fusion(retrieved_docs_lists)

                # Format the final context
                formatted_context = format_docs(reranked_docs)

                # Generate the final response
                assistant_response = final_rag_chain.invoke(
                    {"context": formatted_context, "question": prompt}
                )
                message_placeholder.markdown(assistant_response)

            except Exception as e:
                assistant_response = f"An error occurred: {e}"
                message_placeholder.error(assistant_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_response})
