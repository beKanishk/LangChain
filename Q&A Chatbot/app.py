# # Q&A Chatbot
# from langchain.llms import OpenAI

# from dotenv import load_dotenv

# load_dotenv()  # take environment variables from .env.

# import streamlit as st
# import os


# ## Function to load OpenAI model and get respones

# def get_openai_response(question):
#     llm=OpenAI(model_name="text-davinci-003",temperature=0.5)
#     response=llm(question)
#     return response

# ##initialize our streamlit app

# st.set_page_config(page_title="Q&A Demo")

# st.header("Langchain Application")

# input=st.text_input("Input: ",key="input")
# response=get_openai_response(input)

# submit=st.button("Ask the question")

# ## If ask button is clicked

# if submit:
#     st.subheader("The Response is")
#     st.write(response)



# Q&A Chatbot with Gemini (LangChain + Streamlit)

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
import os

# Load environment variables from .env (containing your Google API key)
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")  # Optional if already in .env

# Function to get response from Gemini
def get_gemini_response(question):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
    response = llm.invoke(question)
    return response

# Initialize Streamlit app
st.set_page_config(page_title="Q&A Demo")
st.header("LangChain Gemini Q&A Application")

# Input from user
input_text = st.text_input("Input: ", key="input")

# Button to trigger the model
submit = st.button("Ask the question")

# If ask button is clicked
if submit:
    st.subheader("The Response is:")
    output = get_gemini_response(input_text)
    st.write(output)
