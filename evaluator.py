import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

def run_evaluation(question, answer, contexts, ground_truth, provider="Gemini", api_key=None):
    dataset = Dataset.from_dict({"question": [question], "answer": [answer], "contexts": [contexts], "ground_truth": [ground_truth]})
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key) if provider == "Gemini" else ChatOpenAI(model="gpt-4o", openai_api_key=api_key)
    try: return evaluate(dataset, metrics=[faithfulness, answer_relevance, context_precision], llm=llm, embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    except Exception as e: return {"error": str(e)}
