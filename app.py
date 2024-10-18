import streamlit as st
import sqlite3
import openai

# Set OpenAI API key
openai.api_key = "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
"

# Create SQLite Database and Table
def create_db():
    """Creates a SQLite database and table for storing user inputs and bot responses."""
    conn = sqlite3.connect("chatbot_memory.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chatbot_memory
                 (id INTEGER PRIMARY KEY, user_input TEXT, bot_response TEXT)''')
    conn.commit()
    conn.close()

def store_in_memory(user_input, bot_response):
    """Stores user input and bot response in the SQLite database."""
    conn = sqlite3.connect("chatbot_memory.db")
    c = conn.cursor()
    c.execute("INSERT INTO chatbot_memory (user_input, bot_response) VALUES (?, ?)",
              (user_input, bot_response))
    conn.commit()
    conn.close()

def retrieve_memory(user_input):
    """Retrieves a stored response from the SQLite database based on user input."""
    conn = sqlite3.connect("chatbot_memory.db")
    c = conn.cursor()
    c.execute("SELECT bot_response FROM chatbot_memory WHERE user_input=?", (user_input,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def get_bot_response(user_input):
    """Fetches the bot's response, either from memory or generated by the AI model."""
    # Check if a memory exists for the user's query
    memory = retrieve_memory(user_input)
    if memory:
        return memory  # Return stored response if it exists
    else:
        # Generate a new response using OpenAI's Chat API (using chat-based method)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Or use "gpt-4" if available
            messages=[
                {"role": "system", "content": "You are a helpful chatbot."},
                {"role": "user", "content": user_input},
            ],
            max_tokens=150
        )
        bot_response = response['choices'][0]['message']['content'].strip()
        store_in_memory(user_input, bot_response)  # Store the response for future use
        return bot_response

# Create the database when the script runs
create_db()

# Streamlit UI for the chatbot
st.title("Chatbot with Memory (SQLite)")
st.write("Ask me anything, and I will remember it!")

# User input
user_input = st.text_input("You can ask a question:")

if user_input:
    bot_response = get_bot_response(user_input)
    st.write("Bot:", bot_response)
