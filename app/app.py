# backend.py
from flask import Flask, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Load RAG components ---
# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large-instruct")

# Load vector store
persist_directory = "database/anwesha_chroma_)db"
vectorstore = Chroma(persist_directory=persist_directory,
                     embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# Load prompt from hub
prompt = hub.pull("rlm/rag-prompt")

# Initialize LLM
llm = ChatGroq(model="moonshotai/kimi-k2-instruct")

# --- Helper functions ---


def format_docs(docs):
    """Formats the retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def reciprocal_rank_fusion(results: list[list], k=60):
    """Applies Reciprocal Rank Fusion to rerank documents."""
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

# --- API Endpoint ---


@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat requests from the Streamlit frontend."""
    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        response = final_rag_chain.invoke({"question": question})
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
