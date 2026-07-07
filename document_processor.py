import os, tempfile, pdfplumber, pytesseract
from pdf2image import convert_from_path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def extract_text_from_pdf(pdf_path, use_ocr=False):
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text: documents.append(Document(page_content=text, metadata={"page": i + 1, "source": pdf_path}))
            elif use_ocr:
                images = convert_from_path(pdf_path, first_page=i+1, last_page=i+1)
                if images: documents.append(Document(page_content=pytesseract.image_to_string(images[0]), metadata={"page": i + 1, "source": pdf_path, "ocr": True}))
    return documents

def process_uploaded_file(uploaded_file, use_ocr=False):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp: tmp.write(uploaded_file.read())
    docs = extract_text_from_pdf(tmp.name, use_ocr=use_ocr)
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    os.remove(tmp.name)
    return chunks
