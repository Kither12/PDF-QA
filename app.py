import os
from src.llm import LLM
from src.pdf_loader import pdf_loader
from src.vector_database import VectorDatabase
from dotenv import load_dotenv
import streamlit as st
import shutil

load_dotenv()
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY") or ""

vector_database = VectorDatabase()
retriever = LLM(vector_database)


def main():
    st.title("PDF-QA Chat Demo")

    uploaded_files = st.file_uploader(
        "Choose PDF files", type="pdf", accept_multiple_files=True
    )

    if uploaded_files is not None:
        document_folder = "document"
        if os.path.exists(document_folder):
            shutil.rmtree(document_folder)
        os.makedirs(document_folder)

        for uploaded_file in uploaded_files:
            with open(f"document/uploaded_document_{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())
            documents = pdf_loader(f"document/uploaded_document_{uploaded_file.name}")
            vector_database.add_documents(documents)

        query = st.text_input("Enter your query:")

        if st.button("Submit"):
            if query:
                response = retriever.query(query)
                st.write(response)
            else:
                st.write("Please enter a query.")


if __name__ == "__main__":
    main()
