# app.py
import streamlit as st
import requests
import json

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
            response = requests.post(FLASK_BACKEND_URL, json=payload)
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
