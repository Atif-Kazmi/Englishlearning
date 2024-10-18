import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the GPT-Neo or GPT-2 model and tokenizer from Hugging Face
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    model = GPT2LMHeadModel.from_pretrained("EleutherAI/gpt-neo-2.7B")
    return tokenizer, model

tokenizer, model = load_model()

def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit app interface
st.title("GPT Chatbot")

st.write(
    """
    This is a simple chatbot using GPT-Neo model from Hugging Face Transformers.
    Enter a message below and the chatbot will respond.
    """
)

# Input field for user query
user_input = st.text_input("You: ")

if user_input:
    st.write("Bot: ", generate_response(user_input))
