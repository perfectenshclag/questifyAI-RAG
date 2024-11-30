import streamlit as st
import os
import numpy as np
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
import easyocr
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
import time

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Streamlit page setup
st.set_page_config(page_title="QuestifyAI - Predict Next Question Paper", page_icon="🎓", layout="centered")
st.title("🎓 QuestifyAI - Question Paper Predictor")
st.write("Upload past question papers and enter a query to predict probable questions for upcoming exams!")

# Initialize embeddings and LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-70b-8192")
llm.temperature = 0.0000000001

# Prompt setup
prompt = ChatPromptTemplate.from_template(
    """
    Context: {context}
    Your job is to predict next question paper by analyzing patterns in previous year test papers. 
    Specifically, identify questions or topics that frequently repeat and have a high probability of reappearing.

    - Analyze each section independently and maintain unique trends for each section.
    - Only produce a structured question paper.
    - Section A will have 10 small questions worth two marks each. Section B will have 5 questions with no subsections.
    - Section C will have 5 questions divided into 2 subparts a and b.Keep in mind that section C is of 10 marks so it would be bigger.
    - The pattern would be same as the papers in the context 
    - Keep all questions within A4 size. Extend to the next line if needed.
    - Only include in-context questions.
    
    Question: {input}
    """
)

# File uploader
uploaded_files = st.file_uploader("📂 Upload past question papers (PDF format)", type="pdf", accept_multiple_files=True)
st.info("Tip: You can upload multiple files to improve the prediction accuracy based on a larger context.")

def convert_pdf_to_text(pdf_bytes):
    """Converts PDF pages to text using EasyOCR."""
    # Initialize the EasyOCR reader (specify language(s) if needed, e.g., ["en"])
    reader = easyocr.Reader(["en"], gpu=False)  # Set `gpu=True` if you want GPU acceleration

    # Convert PDF to images
    images = convert_from_bytes(pdf_bytes, dpi=300)

    # Perform OCR on each page and collect the text
    text = ""
    for page_num, image in enumerate(images):
        result = reader.readtext(np.array(image), detail=0)  # Set `detail=0` for plain text
        page_text = "\n".join(result)
        text += f"\n\nPage {page_num + 1}:\n" + page_text
    
    return text

def create_vector_embedding(uploaded_files):
    """Process uploaded PDFs and create vector embeddings."""
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Process each uploaded file
        docs = []
        for uploaded_file in uploaded_files:
            pdf_text = convert_pdf_to_text(uploaded_file.read())
            doc = Document(page_content=pdf_text, metadata={"source": uploaded_file.name})
            docs.append(doc)
        
        # Split and embed documents
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(docs)
        
        # # Debug: Check processed documents
        # st.write("Processed Documents:")
        # for doc in st.session_state.final_documents[:4]:  # Show first 4 documents
        #     st.write(doc.page_content[:500])  # Show first 500 characters of content

        # Create vector store
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Embed documents button with progress indicator
if st.button("Generate Document Embedding") and uploaded_files:
    with st.spinner("Processing documents... Please wait"):
        create_vector_embedding(uploaded_files)
    st.success("Vector Database created successfully! Ready for question generation.")

# User query
user_prompt = st.text_input("🔍 Enter your question or topic")
if user_prompt and "vectors" in st.session_state:
    # Debug: Check retriever
    retriever = st.session_state.vectors.as_retriever()
    retrieved_docs = retriever.get_relevant_documents(user_prompt)
    st.write(f"Retrieved {len(retrieved_docs)} documents.")
    # for doc in retrieved_docs[:2]:  # Display first 2 retrieved documents
    #     st.write(doc.page_content[:500])  # Show first 500 characters

    # Debug: Test LLM response
    try:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        response_time = time.process_time() - start

        st.write("Generated Response:")
        st.write(response.get('answer', 'No answer generated'))
        st.write(f"⏱️ Response generated in {response_time:.2f} seconds")

        # Generate PDF
        def create_pdf(answer_content, question_text):
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
        
        pdf_buffer = create_pdf(response['answer'], user_prompt)
        st.download_button(
            label="📥 Download Predicted Question Paper as PDF",
            data=pdf_buffer,
            file_name="Predicted_Question_Paper.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")

# Footer
st.markdown("---")
st.markdown("### 🔖 Powered by QuestifyAI")
st.markdown(
    "QuestifyAI uses advanced NLP techniques to analyze previous papers and predict future questions. "
    "Good luck with your studies!"
)
