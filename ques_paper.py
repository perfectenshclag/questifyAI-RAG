
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.schema.document import Document  # Import the Document schema
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from pdf2image import convert_from_path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from dotenv import load_dotenv
from pdf2image import convert_from_bytes
import pytesseract
import io
import time


load_dotenv()

# Load environment variables for API keys
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize embeddings and language model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.2-90b-text-preview")
llm.temperature = 0.45

# Prompt setup for generating question papers
prompt = ChatPromptTemplate.from_template(
    """
    Context: {context}
    Your job is to predict next question paper by analyzing patterns in previous year test papers. 
    Specifically, identify questions or topics that frequently repeat and have a high probability of reappearing.

    - Analyze each section like a,b,c etc independently, and ensure questions in each section reflect trends unique to that section.
    - Only produce a question paper, preserving the structure and order of the analyzed sections.
    - Section A will have 10 questions, Section B will have 5 questions only and will not have a subsection. Section C will have 10 questions in total divided into 5 sections like question -1 will have two options question A and question B from which one is to be attempted.
    - Keep in mind that section A is of two marks so it would be small.
    - Each question will be separate in line as derived from context and should fit in A4 page.
    - If any question extends size of A4 put the remanining part of question in next line.
    - The chances of any predection of question depends mostly on the number of times it has been asked before.
    - Don't add out of context questions just refer the context.
    
    
    Question: {input}
    """
)

def convert_pdf_to_text(pdf_bytes):
    """Converts each page of a PDF to text using OCR on images."""
    text = ""
    
    # Convert PDF bytes directly to images
    images = convert_from_bytes(pdf_bytes, dpi=300)  # Use convert_from_bytes instead of convert_from_path
    
    for page_num, image in enumerate(images):
        page_text = pytesseract.image_to_string(image)
        text += f"\n\nPage {page_num + 1}:\n" + page_text
        
    return text


def create_vector_embedding(uploaded_files):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        
        # Process each uploaded file
        docs = []
        for uploaded_file in uploaded_files:
            pdf_text = convert_pdf_to_text(uploaded_file.read())
            doc = Document(page_content=pdf_text, metadata={"source": uploaded_file.name})
            docs.append(doc)
        
        # Split and embed documents
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(docs)
        
        # Create vector store
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("Question Paper Predictor")

# Upload multiple PDF files
uploaded_files = st.file_uploader("Upload multiple PDFs", type="pdf", accept_multiple_files=True)

user_prompt = st.text_input("Enter your query from the research paper")

# Initialize vector database if user clicks the button
if st.button("Document Embedding") and uploaded_files:
    create_vector_embedding(uploaded_files)
    st.write("Vector Database is ready")

if user_prompt and "vectors" in st.session_state:
    # Chain setup for question-answering with retrieved context
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Retrieve answer
    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt})
    print(f"Response time: {time.process_time() - start}")

    st.write(response['answer'])

    # PDF generation and download button
    pdf_buffer = create_pdf(response['answer'])
    st.download_button(
        label="Download Question Paper as PDF",
        data=pdf_buffer,
        file_name="Question_Paper.pdf",
        mime="application/pdf"
    )
