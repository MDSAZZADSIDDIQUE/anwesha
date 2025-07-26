# app.py
import streamlit as st
import requests
import json
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
import threading
import os

# --- 1. FLASK BACKEND SETUP ---

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Load RAG components ---
# NOTE: This part can be slow and memory-intensive. It runs once when the app starts.


@st.cache_resource
def load_rag_pipeline():
    """Loads all the components needed for the RAG pipeline and caches them."""
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large-instruct")

    # Check if the database directory exists, if not, this will fail.
    # You need to run the data ingestion part of your notebook first to create this.
    persist_directory = "database/anwesha_chroma_)db"
    if not os.path.exists(persist_directory):
        # A simple way to handle missing DB is to stop the app and inform the user.
        st.error(
            "ChromaDB database not found. Please run the data ingestion script from the notebook first to create it.")
        st.stop()

    vectorstore = Chroma(persist_directory=persist_directory,
                         embedding_function=embeddings)
    retriever = vectorstore.as_retriever()

    # Load prompt from hub
    prompt = hub.pull("rlm/rag-prompt")

    # Initialize LLM
    llm = ChatGroq(model="moonshotai/kimi-k2-instruct")

    # --- Helper functions ---
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

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
        return [doc for doc, _ in reranked_results]

    # --- RAG Chain Definition ---
    template = """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {question}"""
    prompt_perspectives = ChatPromptTemplate.from_template(template)

    generate_queries = (
        prompt_perspectives
        | ChatGroq(model="llama3-8b-8192")
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    retrieval_chain_with_reranking = generate_queries | retriever.map() | reciprocal_rank_fusion

    final_rag_chain = (
        {"context": retrieval_chain_with_reranking,
            "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    )
    return final_rag_chain


# Load the pipeline
final_rag_chain = load_rag_pipeline()

# --- API Endpoint ---


@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat requests."""
    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        response = final_rag_chain.invoke({"question": question})
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Function to run Flask app in a thread ---


def run_flask_app():
    # Note: Setting debug=False is important for production environments
    app.run(host='0.0.0.0', port=5000, debug=False)


# --- Start Flask app in a background thread ---
# This ensures the Flask server is running and ready to accept requests from Streamlit
flask_thread = threading.Thread(target=run_flask_app, daemon=True)
flask_thread.start()


# --- 2. STREAMLIT FRONTEND ---

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="RAG Chat App", page_icon="🤖")
st.title("HSC Bangla 1st Paper - RAG Chatbot")
st.write("Ask me anything about 'Aparichita' from the HSC Bangla 1st Paper!")

# --- Backend API URL ---
FLASK_BACKEND_URL = "http://127.0.0.1:5000/chat"

# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input and Backend Interaction ---
if prompt := st.chat_input("What is your question?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Prepare data for the POST request
            payload = {"question": prompt}

            # Send request to Flask backend
            response = requests.post(
                FLASK_BACKEND_URL, json=payload, timeout=120)  # Added timeout
            response.raise_for_status()  # Raise an exception for bad status codes

            assistant_response = response.json().get(
                "response", "Sorry, I couldn't get a response.")

            # Simulate stream of response with milliseconds delay
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        except requests.exceptions.RequestException as e:
            full_response = f"Error connecting to the backend: {e}"
            message_placeholder.error(full_response)
        except Exception as e:
            full_response = f"An error occurred: {e}"
            message_placeholder.error(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response})

# --- Clear History Button ---
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
