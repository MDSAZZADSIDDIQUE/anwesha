import os
import sys
import asyncio
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# LangChain and RAG components
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from operator import itemgetter

# RAGAs for evaluation
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

# --- GLOBAL INITIALIZATION (Load models once) ---

# Load environment variables from .env file for local development
# On Streamlit Cloud, these will be set as Secrets
load_dotenv()

st.title("📚 Anwesha RAG Chatbot")

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
def load_retriever():
    """Load the vector store and retriever."""
    persist_directory = "../database/anwesha_chroma_)db"
    if not os.path.exists(persist_directory):
        st.error(
            f"ChromaDB directory not found at {persist_directory}. Please ensure the database is created and accessible in your deployment environment.")
        st.stop()
    vectorstore = Chroma(persist_directory=persist_directory,
                         embedding_function=embeddings)
    return vectorstore.as_retriever()


@st.cache_resource
def load_llm(_groq_api_key):
    """Load the Language Model."""
    return ChatGroq(model="moonshotai/kimi-k2-instruct", api_key=_groq_api_key)


# --- Load all components ---
with st.spinner("Loading models and vector store..."):
    embeddings = load_embeddings()
    retriever = load_retriever()
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


retrieval_chain_with_reranking = generate_queries | retriever.map() | reciprocal_rank_fusion

final_rag_chain = (
    {"context": retrieval_chain_with_reranking |
        format_docs, "question": itemgetter("question")}
    | prompt_template
    | llm
    | StrOutputParser()
)

# --- STREAMLIT UI DEFINITION ---


with st.sidebar:
    st.title("Anwesha RAG System")
    st.markdown("A chatbot for Rabindranath Tagore's 'Aparichita'.")
    st.markdown("---")
    st.subheader("Evaluation")
    st.markdown("Click to run an evaluation of the RAG system's performance.")

    if st.button("📊 Run Evaluation"):
        with st.spinner("Running evaluation... This may take a moment."):
            try:
                sample_queries = [
                    "অপরিচিতা' গল্পে, অনুপমের মতে কে আসর জমাতে অদ্বিতীয়?", "অনুপম তার মামার চেয়ে কত বছরের ছোট ছিল?",
                    "মন্দ নয় হে! খাঁটি সোনা বটে!' - এই উক্তিটি কার?", "কল্যাণীর বাবার নাম কী?", "বিবাহ-উপলক্ষ্যে কন্যাপক্ষকে কোথায় আসতে হয়েছিল?",
                    "শম্ভুনাথ সেন পেশায় কী ছিলেন?", "অনুপম এবং তার মা কোন বাহনে তীর্থে যাচ্ছিলেন?", "রেলগাড়িতে কল্যাণীর সাথে কয়টি ছোট ছোট মেয়ে ছিল?",
                    "বিবাহ ভাঙার পর কল্যাণী কী ব্রত গ্রহণ করে?", "গল্পের শেষে অনুপমের বয়স কত?"
                ]
                expected_responses = [
                    "হরিশ", "বছর ছয়েক", "বিনুদা", "শম্ভুনাথ সেন", "কলিকাতা", "ডাক্তার", "রেলগাড়ি", "দুটি-তিনটি", "মেয়েদের শিক্ষার ব্রত", "সাতাশ"
                ]

                dataset_list = []
                for query, reference in zip(sample_queries, expected_responses):
                    retrieved_docs = retriever.invoke(query)
                    response_text = final_rag_chain.invoke({"question": query})
                    dataset_list.append({
                        "question": query, "contexts": [doc.page_content for doc in retrieved_docs],
                        "answer": response_text, "ground_truth": reference,
                    })

                df = pd.DataFrame(dataset_list)
                evaluator_llm = LangchainLLMWrapper(llm)
                metrics = [Faithfulness(), LLMContextRecall(),
                           FactualCorrectness()]

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(evaluate(
                    dataset=df, metrics=metrics, llm=evaluator_llm, raise_exceptions=False))
                loop.close()

                st.session_state.evaluation_metrics = result.scores.to_dict()
                st.success("Evaluation complete!")
            except Exception as e:
                st.error(f"An error occurred during evaluation: {e}")

    if 'evaluation_metrics' in st.session_state:
        st.markdown("---")
        st.subheader("Evaluation Results")
        metrics_data = st.session_state.evaluation_metrics
        faithfulness = metrics_data.get('faithfulness', 0)
        context_recall = metrics_data.get('context_recall', 0)
        factual_correctness = metrics_data.get('factual_correctness', 0)

        st.metric(label="Faithfulness", value=f"{faithfulness:.2f}")
        st.progress(float(faithfulness))
        st.metric(label="Context Recall", value=f"{context_recall:.2f}")
        st.progress(float(context_recall))
        st.metric(label="Factual Correctness",
                  value=f"{factual_correctness:.2f}")
        st.progress(float(factual_correctness))

        with st.expander("See Raw Metrics Data"):
            st.json(metrics_data)

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
                assistant_response = final_rag_chain.invoke(
                    {"question": prompt})
                message_placeholder.markdown(assistant_response)
            except Exception as e:
                assistant_response = f"An error occurred: {e}"
                message_placeholder.error(assistant_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_response})
