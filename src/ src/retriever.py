from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

class RerankingRetriever:
    def __init__(self, base_retriever):
        self.base_retriever = base_retriever
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
    def get_relevant_documents(self, query, top_n=3):
        docs = self.base_retriever.invoke(query)
        if not docs: return []
        scores = self.cross_encoder.predict([[query, doc.page_content] for doc in docs])
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [Document(page_content=d.page_content, metadata={**d.metadata, "relevance_score": float(s)}) for d, s in scored_docs[:top_n]]
