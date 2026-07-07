import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class QAResponse(BaseModel):
    answer: str = Field(...)
    citations: list[str] = Field(...)
    confidence_score: float = Field(...)
    is_safe: bool = Field(...)

SYSTEM_PROMPT = """
You are a highly capable and strictly factual medical AI assistant. Answer using ONLY the provided retrieved context.
Rules: 1. No hallucinations. 2. Safety checking. 3. Output Citations. 4. Confidence score (0.0 to 1.0).
Context: {context}
"""

def answer_question(query, docs, provider="Gemini", api_key=None):
    if not docs: return {"answer": "No context found.", "confidence_score": 0.0, "is_safe": True}
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key or os.getenv("GOOGLE_API_KEY")) if provider == "Gemini" else ChatOpenAI(model="gpt-4o", openai_api_key=api_key or os.getenv("OPENAI_API_KEY"))
    parser = JsonOutputParser(pydantic_object=QAResponse)
    chain = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}\\n{format_instructions}")]) | llm | parser
    try: return chain.invoke({"context": "\\n".join([d.page_content for d in docs]), "question": query, "format_instructions": parser.get_format_instructions()})
    except Exception as e: return {"answer": str(e), "confidence_score": 0.0, "is_safe": False}
