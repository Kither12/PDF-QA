from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf4llm


def load_pdf_with_pymupdf(filepath: str) -> list[Document]:
    doc = pymupdf4llm.to_markdown(filepath)
    doc = Document(page_content=doc)
    return [doc]


def pdf_loader(filepath: str) -> list[Document]:
    docs = load_pdf_with_pymupdf(filepath)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits
