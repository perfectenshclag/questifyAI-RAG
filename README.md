
# 🎓 Question Paper Predictor

**A Streamlit-based application that predicts potential question papers by analyzing patterns in previous papers.** This app leverages advanced NLP techniques with **LangChain, FAISS embeddings, and a powerful language model** to forecast likely future questions based on historical data—ideal for educators and students preparing for exams! 🚀

---

## ✨ Features

- 📄 **PDF Upload and OCR**: Upload multiple PDFs for analysis. Our app extracts text even from image-based PDFs using OCR, making sure no data is left behind.
- 🔍 **Vector Embedding and Retrieval**: Generate embeddings for uploaded documents to retrieve information based on user queries.
- 📜 **Question Paper Prediction**: Anticipates questions by analyzing patterns in question formats and themes from past papers.
- 📥 **PDF Export**: Download the predicted question paper as a beautifully styled PDF for easy sharing and review.

Try it live here! 👉 [**Questifyai on Streamlit**](https://questifyai.streamlit.app)

---

## 🛠 Tech Stack

- **Python Libraries**: `Streamlit`, `LangChain`, `FAISS`, `HuggingFace Embeddings`, `ReportLab`, `pdf2image`, `pytesseract`
- **OCR**: `pdf2image` and `pytesseract` for PDF-to-text conversion
- **Vector Store**: `FAISS` for document embedding and retrieval
- **NLP Model**: `ChatGroq` with `HuggingFace embeddings`
- **PDF Generation**: `ReportLab` to create downloadable PDFs with style

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Question-Paper-Predictor.git
cd Question-Paper-Predictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set up environment variables with your credentials by creating a `.env` file:

```plaintext
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

Install Tesseract for OCR capabilities:

- **Ubuntu**:
  ```bash
  sudo apt-get install tesseract-ocr
  ```
- **macOS**:
  ```bash
  brew install tesseract
  ```

---

## ⚙️ Usage

1. **Run the App**:
   ```bash
   streamlit run app.py
   ```

2. **Upload PDFs**: Drag and drop PDFs to extract the content for analysis.

3. **Generate Vector Embeddings**: Initialize the vector database by selecting "Document Embedding" after uploading.

4. **Enter Query**: Input a question pattern prompt to analyze past questions and predict future trends.

5. **Download PDF**: Get a downloadable PDF with the generated question paper, ready to share or study from!

---

## 📁 Code Structure

- **`app.py`**: Main application file. It handles Streamlit components, PDF processing, embedding creation, retrieval chain setup, and PDF export.
- **`requirements.txt`**: Contains the list of dependencies.
- **`.env`**: Stores API keys and tokens securely.

---

## 🧰 Dependencies

```plaintext
streamlit
langchain
langchain_community
langchain_core
dotenv
pdf2image
pytesseract
reportlab
```

--- 

Uncover patterns, anticipate questions, and empower the education process with **Question Paper Predictor**! This project fits perfectly into the hackathon's **AI Innovation** theme, pushing boundaries in NLP and real-time analytics for smarter study resources. 📚✨
