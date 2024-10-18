import streamlit as st
from transformers import pipeline

# Load the chatbot model
@st.cache_resource
def load_model():
    chatbot = pipeline("text-generation", model="gpt2")
    return chatbot

# Main app interface
st.title("English Learning Chatbot")
st.write("Ask me anything, and I will respond in English!")

# Get user input
user_input = st.text_input("You: ", "")

# Generate a response if input is provided
if user_input:
    chatbot = load_model()
    response = chatbot(user_input, max_length=100, num_return_sequences=1)
    st.write(f"Bot: {response[0]['generated_text']}")
