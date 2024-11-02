import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.schema.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from pdf2image import convert_from_bytes
import pytesseract
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
import io
import time

load_dotenv()

# Load environment variables for API keys
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize embeddings and language model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.2-90b-text-preview")
llm.temperature = 0.2

# Prompt setup for generating question papers
prompt = ChatPromptTemplate.from_template(
    """
    Context: {context}
    Your job is to predict next question paper by analyzing patterns in previous year test papers. 
    Specifically, identify questions or topics that frequently repeat and have a high probability of reappearing.

    - Analyze each section independently and maintain unique trends for each section.
    - Only produce a structured question paper.
    - Section A will have 10 small questions worth two marks each. Section B will have 5 questions with no subsections.
    - Section C will have 10 questions divided into 5 parts; question 1 will have two options (A and B) where one is to be attempted.
    - Keep all questions within A4 size. Extend to the next line if needed.
    - Only include in-context questions.
    
    Question: {input}
    """
)

def convert_pdf_to_text(pdf_bytes):
    """Converts each page of a PDF to text using OCR on images."""
    text = ""
    images = convert_from_bytes(pdf_bytes, dpi=300)
    for page_num, image in enumerate(images):
        page_text = pytesseract.image_to_string(image)
        text += f"\n\nPage {page_num + 1}:\n" + page_text
    return text

def create_vector_embedding(uploaded_files):
    """Process uploaded PDFs and create vector embeddings."""
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        docs = []
        for uploaded_file in uploaded_files:
            pdf_text = convert_pdf_to_text(uploaded_file.read())
            doc = Document(page_content=pdf_text, metadata={"source": uploaded_file.name})
            docs.append(doc)
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

def create_pdf(answer_content, question_text):
    """Generate a styled PDF document using reportlab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Title'], fontSize=22, spaceAfter=12, textColor=colors.HexColor("#003366"), alignment=1)
    question_style = ParagraphStyle('QuestionStyle', parent=styles['Heading2'], fontSize=16, spaceAfter=10, textColor=colors.green)
    answer_style = ParagraphStyle('AnswerStyle', parent=styles['BodyText'], fontSize=14, spaceAfter=8, leftIndent=10, rightIndent=10)
    
    elements = [
        Paragraph("📄 Generated Answer", title_style),
        Spacer(1, 0.2 * inch),
        Paragraph(f"<b>Question:</b> {question_text}", question_style),
        Spacer(1, 0.1 * inch)
    ]

    answer_lines = answer_content.split("\n\n")
    for line in answer_lines:
        elements.append(Paragraph(line, answer_style))
        elements.append(Spacer(1, 0.1 * inch))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

st.title("Question Paper Predictor")

# Upload multiple PDF files
uploaded_files = st.file_uploader("Upload multiple PDFs", type="pdf", accept_multiple_files=True)

user_prompt = st.text_input("Enter your query from the research paper")

if st.button("Document Embedding") and uploaded_files:
    create_vector_embedding(uploaded_files)
    st.write("Vector Database is ready")

if user_prompt and "vectors" in st.session_state:
    # Chain setup for question-answering with retrieved context
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt})
    print(f"Response time: {time.process_time() - start}")

    st.write(response['answer'])

    # PDF generation and download button
    pdf_buffer = create_pdf(response['answer'], user_prompt)
    st.download_button(
        label="Download Question Paper as PDF",
        data=pdf_buffer,
        file_name="Predicted_Question_Paper.pdf",
        mime="application/pdf"
    )
