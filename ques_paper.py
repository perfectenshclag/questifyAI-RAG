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

st.set_page_config(page_title="QuestifyAI - Predict Next Question Paper", page_icon="🎓", layout="centered")

# Load environment variables
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize embeddings and model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.2-90b-text-preview")
llm.temperature = 0.2

# Prompt setup
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

# Function definitions remain the same

st.title("🎓 QuestifyAI - Question Paper Predictor")
st.write("Upload past question papers and enter a query to predict probable questions for upcoming exams!")

# Upload section with instructions
uploaded_files = st.file_uploader("📂 Upload past question papers (PDF format)", type="pdf", accept_multiple_files=True)
st.info("Tip: You can upload multiple files to improve the prediction accuracy based on a larger context.")

# Text input for user's query
user_prompt = st.text_input("🔍 Enter your question or topic")
def create_vector_embedding(uploaded_files):
    """Process uploaded PDFs and create vector embeddings."""
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
# Embed documents button with progress indicator
if st.button("Generate Document Embedding") and uploaded_files:
    with st.spinner("Processing documents... Please wait"):
        create_vector_embedding(uploaded_files)
    st.success("Vector Database created successfully! Ready for question generation.")

# If embeddings are created and user enters a query
if user_prompt and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt})
    response_time = time.process_time() - start
    st.write(response['answer'])
    st.write(f"⏱️ Response generated in {response_time:.2f} seconds")

    # PDF generation and download button
    pdf_buffer = create_pdf(response['answer'], user_prompt)
    st.download_button(
        label="📥 Download Predicted Question Paper as PDF",
        data=pdf_buffer,
        file_name="Predicted_Question_Paper.pdf",
        mime="application/pdf"
    )

# Footer with branding
st.markdown("---")
st.markdown("### 🔖 Powered by QuestifyAI")
st.markdown(
    "QuestifyAI uses advanced NLP techniques to analyze previous papers and predict future questions. "
    "Good luck with your studies!"
)
