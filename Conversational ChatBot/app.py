from dotenv import load_dotenv
import os

load_dotenv()

import streamlit as st
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

##function to load gemini-pro model
model = genai.GenerativeModel("gemini-2.0-flash")
chat = model.start_chat(history=[])

def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

## Streamlit app configuration
st.set_page_config(page_title="Conversational ChatBot", page_icon=":robot_face:", layout="wide")
st.header("Conversational ChatBot")

## Intitalize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []
    
input = st.text_input("Ask a question:", key="input")
submit = st.button("Submit")

if submit and input:
    response = get_gemini_response(input)
    st.session_state['chat_history'].append(("You", input))
    st.subheader("The response is")
    for chunk in response:
        st.write(chunk.text)
    st.session_state['chat_history'].append(("Bot", response.text))

st.subheader("Chat History")
for user, message in st.session_state['chat_history']:
    st.write(f"**{user}:** {message}")
