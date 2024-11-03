Question Paper Predictor

A Streamlit-based application to predict potential question papers by analyzing patterns in previous year papers. This app uses advanced natural language processing (NLP) techniques, such as LangChain with FAISS embeddings and a language model, to generate possible future questions.
Features

PDF Upload and OCR: Supports multiple PDF uploads with OCR to extract text from image-based PDFs.
Vector Embedding and Retrieval: Creates embeddings for uploaded documents and enables retrieval based on user prompts.
Question Paper Prediction: Generates question papers based on recurring themes and question formats.
PDF Export: Allows users to download the generated question paper as a PDF document.
Tech Stack

Python Libraries: Streamlit, LangChain, FAISS, HuggingFace Embeddings, ReportLab, pdf2image, pytesseract.
OCR: pdf2image and pytesseract for PDF-to-text conversion.
Vector Store: FAISS for document embedding and retrieval.
NLP Model: ChatGroq with embeddings from HuggingFace.
PDF Generation: ReportLab for generating downloadable PDFs.
Installation

Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/Question-Paper-Predictor.git
cd Question-Paper-Predictor
Install required dependencies:
bash
Copy code
pip install -r requirements.txt
Environment Variables: Add your environment variables in a .env file:
plaintext
Copy code
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
Install Tesseract (if not installed):
Ubuntu:
bash
Copy code
sudo apt-get install tesseract-ocr
macOS:
bash
Copy code
brew install tesseract
Usage

Run the App:
bash
Copy code
streamlit run app.py
Upload PDFs: Drag and drop multiple PDFs to extract content.
Generate Vector Embeddings: Click "Document Embedding" after uploading files to initialize the vector database.
Enter Query: Input a prompt related to the question pattern you want to analyze.
Download PDF: After the app generates the response, download the question paper as a PDF.
Code Structure

app.py: Main application file with Streamlit components, PDF processing, embedding creation, retrieval chain setup, and PDF export.
requirements.txt: List of dependencies for the project.
.env: Contains API keys and tokens for secure access.
Dependencies

plaintext
Copy code
streamlit
langchain
langchain_community
langchain_core
dotenv
pdf2image
pytesseract
reportlab
Example Workflow

Upload PDFs containing past question papers.
Generate Embeddings and build the vector store for quick retrieval.
Predict Questions by providing a query related to frequent questions or topics.
Download the Generated PDF containing the predicted question paper.
Future Enhancements

Add support for non-PDF document formats.
Implement more sophisticated question pattern analysis using additional language models.
Expand to cover various question paper structures beyond a fixed format.
License

This project is licensed under the MIT License.