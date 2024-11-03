# Question Paper Predictor

A Streamlit-based application that predicts potential question papers by analyzing patterns in previous papers. This app utilizes advanced NLP techniques with LangChain, FAISS embeddings, and a language model to generate likely future questions based on historical data.

---

## Features

1. **PDF Upload and OCR**: Supports uploading multiple PDFs with OCR to extract text from image-based PDFs.
2. **Vector Embedding and Retrieval**: Creates embeddings for uploaded documents and enables retrieval based on user prompts.
3. **Question Paper Prediction**: Generates question papers based on recurring themes and question formats.
4. **PDF Export**: Allows users to download the generated question paper as a styled PDF document.

---

## Hosted Link

Try out the app live here: [Question Paper Predictor on Streamlit](https://question-paper.streamlit.app)

---

## Tech Stack

- **Python Libraries**: Streamlit, LangChain, FAISS, HuggingFace Embeddings, ReportLab, pdf2image, pytesseract.
- **OCR**: `pdf2image` and `pytesseract` for PDF-to-text conversion.
- **Vector Store**: FAISS for document embedding and retrieval.
- **NLP Model**: ChatGroq with HuggingFace embeddings.
- **PDF Generation**: ReportLab for generating downloadable PDFs.

---

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/Question-Paper-Predictor.git
    cd Question-Paper-Predictor
    ```

2. **Install required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables**: Add your environment variables in a `.env` file:

    ```plaintext
    GROQ_API_KEY=your_groq_api_key
    HF_TOKEN=your_huggingface_token
    ```

4. **Install Tesseract** (if not installed):

    - **Ubuntu**:
      ```bash
      sudo apt-get install tesseract-ocr
      ```
    - **macOS**:
      ```bash
      brew install tesseract
      ```

---

## Usage

1. **Run the App**:
    ```bash
    streamlit run app.py
    ```

2. **Upload PDFs**: Drag and drop multiple PDFs to extract content.

3. **Generate Vector Embeddings**: Click "Document Embedding" after uploading files to initialize the vector database.

4. **Enter Query**: Input a prompt related to the question pattern you want to analyze.

5. **Download PDF**: After the app generates the response, download the question paper as a PDF.

---

## Code Structure

- **app.py**: Main application file with Streamlit components, PDF processing, embedding creation, retrieval chain setup, and PDF export.
- **requirements.txt**: List of dependencies for the project.
- **.env**: Contains API keys and tokens for secure access.

---

## Dependencies

```plaintext
streamlit
langchain
langchain_community
langchain_core
dotenv
pdf2image
pytesseract
reportlab
