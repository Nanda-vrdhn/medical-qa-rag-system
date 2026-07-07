import os, chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorStoreManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        if not os.path.exists("./chroma_db"): os.makedirs("./chroma_db")
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
    def add_documents(self, documents):
        return Chroma.from_documents(documents=documents, embedding=self.embeddings, persist_directory="./chroma_db", client=self.client)
        
    def get_retriever(self, k=5):
        return Chroma(client=self.client, embedding_function=self.embeddings).as_retriever(search_kwargs={"k": k})
