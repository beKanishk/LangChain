import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_core.documents import Document
from dotenv import load_dotenv
from astrapy import DataAPIClient
import google.generativeai as genai
import os

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Astra DB setup
client = DataAPIClient()
db = client.get_database(
    os.getenv("ASTRA_DB_API_ENDPOINT"),
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN")
)
collection = db.get_collection("pdf_chunks")

# Extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# Split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    return splitter.split_text(text)

# Store chunks into Astra DB with automatic vector embedding
def store_chunks(chunks):
    for chunk in chunks:
        collection.insert_one({"text": chunk, "$vectorize": chunk})

# Search for similar chunks using AstraDB's $vectorize
def search_chunks(query, k=3):
    cursor = collection.find({}, sort={"$vectorize": query}, include_similarity=True)
    results = list(cursor)[:k]
    return [Document(page_content=doc["text"]) for doc in results]

# LangChain QA chain using Gemini
def get_qa_chain():
    template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, say "answer is not available in the context".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Handle user question
def handle_question(question):
    docs = search_chunks(question)
    chain = get_qa_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])

# Clear all chunks
def delete_chunks():
    collection.delete_many({})
    st.success("All chunks deleted from database.")

# Main Streamlit UI
def main():
    st.set_page_config("Chat with PDF")
    st.header("📄 Chat with PDF using Gemini")

    question = st.text_input("Ask a question from your uploaded PDF")
    if question:
        handle_question(question)

    with st.sidebar:
        st.title("Options:")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                store_chunks(chunks)
                st.success("PDF processed and chunks stored.")

        if st.button("Delete All Chunks"):
            delete_chunks()

if __name__ == "__main__":
    main()
