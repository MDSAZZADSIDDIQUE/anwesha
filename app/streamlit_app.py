import streamlit as st
import requests
import json
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Anwesha RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- API URL ---
FLASK_API_URL = "http://127.0.0.1:5001"

# --- SIDEBAR ---
with st.sidebar:
    st.title("📚 Anwesha-A Bengali pdf chatbot.")
    st.markdown(
        "A chatbot to answer questions about Rabindranath Tagore's 'Aparichita'.")

    st.markdown("---")
    st.subheader("Evaluation")
    st.markdown(
        "Click the button below to run an evaluation of the RAG system's performance using a predefined set of questions.")

    if st.button("📊 Run Evaluation"):
        with st.spinner("Running evaluation... This may take a few minutes."):
            try:
                response = requests.get(f"{FLASK_API_URL}/evaluate")
                response.raise_for_status()
                metrics = response.json()

                st.session_state.evaluation_metrics = metrics
                st.success("Evaluation complete!")

            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the backend: {e}")
            except Exception as e:
                st.error(f"An error occurred during evaluation: {e}")

    if 'evaluation_metrics' in st.session_state:
        st.markdown("---")
        st.subheader("Evaluation Results")

        metrics_data = st.session_state.evaluation_metrics

        # Display key metrics with st.metric
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

# --- MAIN CHAT INTERFACE ---
st.title("Anwesha Chatbot")
st.markdown("Ask me anything about 'Aparichita' in Bangla or English.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("আপনার প্রশ্ন লিখুন..."):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from the backend
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{FLASK_API_URL}/chat",
                    json={"question": prompt}
                )
                response.raise_for_status()
                assistant_response = response.json().get(
                    "response", "Sorry, I encountered an error.")
                message_placeholder.markdown(assistant_response)

            except requests.exceptions.RequestException as e:
                assistant_response = f"Error: Could not connect to the backend. Please ensure it's running. Details: {e}"
                message_placeholder.error(assistant_response)

    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_response})
