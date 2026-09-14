from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Re-export load_json_documents for backward compatibility
from rag_components.json_doc_parser import load_json_documents


def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splitted_docs = text_splitter.split_documents(documents)
    return splitted_docs