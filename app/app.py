from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large-instruct")
persist_directory = "../database/anwesha_chroma"
vectorstore = Chroma(persist_directory=persist_directory,
                     embedding_function=embeddings)
retriever = vectorstore.as_retriever()

llm = ChatGroq(model="moonshotai/kimi-k2-instruct")

prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    """Formats the retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]


rag_chain = (
    RunnablePassthrough.assign(
        context=contextualized_question | retriever | format_docs
    )
    | qa_prompt
    | llm
)


@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles the chat requests.
    """
    data = request.get_json()
    question = data.get('question')
    chat_history_json = data.get('chat_history', [])

    chat_history = ChatMessageHistory()
    for msg in chat_history_json:
        if msg.get('type') == 'human':
            chat_history.add_user_message(msg.get('content'))
        elif msg.get('type') == 'ai':
            chat_history.add_ai_message(msg.get('content'))

    ai_msg = rag_chain.invoke(
        {"question": question, "chat_history": chat_history.messages})

    return jsonify({'response': ai_msg.content})


if __name__ == '__main__':
    app.run(debug=True, port=5001)
