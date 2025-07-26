import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain.load import dumps, loads
from operator import itemgetter

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas.evaluation import EvaluationResult
import pandas as pd
import asyncio

load_dotenv()

app = Flask(__name__)
CORS(app)

print("Initializing RAG components...")

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large-instruct")

persist_directory = "../database/anwesha_chroma_)db"
if not os.path.exists(persist_directory):
    raise FileNotFoundError(
        f"ChromaDB directory not found at {persist_directory}. Please ensure the database is created and accessible.")

vectorstore = Chroma(persist_directory=persist_directory,
                     embedding_function=embeddings)
retriever = vectorstore.as_retriever()

llm = ChatGroq(model="moonshotai/kimi-k2-instruct")

prompt = hub.pull("rlm/rag-prompt")

print("Initialization complete.")


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
    """Reranks documents using Reciprocal Rank Fusion."""
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
    | prompt
    | llm
    | StrOutputParser()
)


@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles chat requests by invoking the RAG chain.
    """
    try:
        data = request.get_json()
        question = data.get('question')

        if not question:
            return jsonify({'error': 'Question is required'}), 400

        ai_response = final_rag_chain.invoke({"question": question})

        return jsonify({'response': ai_response})
    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        return jsonify({'error': 'An internal error occurred.'}), 500


@app.route('/evaluate', methods=['GET'])
def evaluate_rag():
    """
    Runs the RAG evaluation using a predefined set of questions and answers.
    """
    try:
        print("Starting RAG evaluation...")
        sample_queries = [
            "অপরিচিতা' গল্পে, অনুপমের মতে কে আসর জমাতে অদ্বিতীয়?",
            "অনুপম তার মামার চেয়ে কত বছরের ছোট ছিল?",
            "মন্দ নয় হে! খাঁটি সোনা বটে!' - এই উক্তিটি কার?",
            # "কল্যাণীর বাবার নাম কী?",
            # "বিবাহ-উপলক্ষ্যে কন্যাপক্ষকে কোথায় আসতে হয়েছিল?",
            # "শম্ভুনাথ সেন পেশায় কী ছিলেন?",
            # "অনুপম এবং তার মা কোন বাহনে তীর্থে যাচ্ছিলেন?",
            # "রেলগাড়িতে কল্যাণীর সাথে কয়টি ছোট ছোট মেয়ে ছিল?",
            # "বিবাহ ভাঙার পর কল্যাণী কী ব্রত গ্রহণ করে?",
            # "গল্পের শেষে অনুপমের বয়স কত?"
        ]

        expected_responses = [
            "হরিশ", "বছর ছয়েক", "বিনুদা",
            # "শম্ভুনাথ সেন", "কলিকাতা",
            # "ডাক্তার", "রেলগাড়ি", "দুটি-তিনটি", "মেয়েদের শিক্ষার ব্রত", "সাতাশ"
        ]

        dataset_list = []
        for query, reference in zip(sample_queries, expected_responses):
            retrieved_docs = retriever.invoke(query)
            response = final_rag_chain.invoke({"question": query})
            dataset_list.append({
                "question": query,
                "contexts": [doc.page_content for doc in retrieved_docs],
                "answer": response,
                "ground_truth": reference,
            })

        df = pd.DataFrame(dataset_list)

        evaluator_llm = LangchainLLMWrapper(llm)
        metrics = [
            Faithfulness(),
            LLMContextRecall(),
            FactualCorrectness(),
        ]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            evaluate(
                dataset=df,
                metrics=metrics,
                llm=evaluator_llm,
                raise_exceptions=False
            )
        )
        loop.close()

        print("Evaluation complete.")
        return jsonify(result.scores.to_dict())

    except Exception as e:
        print(f"Error in /evaluate endpoint: {e}")
        return jsonify({'error': f'An evaluation error occurred: {e}'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)
