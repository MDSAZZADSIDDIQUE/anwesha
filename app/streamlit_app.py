import streamlit as st
import requests
import json

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("Anwesha - A Bengali PDF Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    chat_history_for_api = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_history_for_api.append(
                {"type": "human", "content": msg["content"]})
        elif msg["role"] == "assistant":
            chat_history_for_api.append(
                {"type": "ai", "content": msg["content"]})

    try:
        response = requests.post(
            "http://127.0.0.1:5001/chat",
            json={"question": prompt, "chat_history": chat_history_for_api}
        )
        response.raise_for_status()
        assistant_response = response.json()["response"]

        with st.chat_message("assistant"):
            st.markdown(assistant_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response})

    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the backend: {e}")
