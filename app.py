import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

# Main Streamlit app code
st.title("English Learning Chatbot")
st.write("Ask me anything in English!")

# Get user input
user_input = st.text_input("You: ", "")

# If user input is provided, generate a response
if user_input:
    tokenizer, model = load_model()
    inputs = tokenizer.encode(user_input, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, do_sample=True)
    bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write(f"Bot: {bot_response}")
