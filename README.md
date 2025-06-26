# 🧠 Intelligent Document Processing with AI-Powered Evaluation Agent

## 🔍 Overview

This project extracts and processes text from PDF documents stored in **Amazon S3** using **AWS Textract**, applies **NLP preprocessing**, stores semantic vector embeddings in **Pinecone**, and enables intelligent **RAG-based multi-turn question-answering** using Amazon **Titan Express**, **Premier**, and **Lite** models via **Amazon Bedrock**.

It performs **automated evaluation and model selection** using similarity, fluency, and generation time metrics, and maintains chat history for natural multi-turn interaction.

---

## 🚀 Key Features

* 🔓 Extracts text from PDFs stored in **S3** using **AWS Textract** (asynchronous detection).
* 🧹 Preprocesses text using **NLTK**: tokenization, stopword removal, lemmatization.
* 🔗 Converts text to embeddings with **HuggingFace’s `multi-qa-mpnet-base-cos-v1`** model.
* 📦 Stores and retrieves embeddings with **Pinecone** for semantic similarity search.
* 💬 Answers questions using **retrieval-augmented generation (RAG)** with **Amazon Titan** LLMs:

  * `Titan Text Express v1`
  * `Titan Text Premier v1`
  * `Titan Text Lite v1`
* 📊 Evaluates responses automatically using:

  * **Query Similarity**
  * **Context Similarity**
  * **Fluency (token-based)**
  * **Generation Time**
* 🏆 Dynamically selects the **best model response** per query based on composite scoring.
* 🧠 Maintains **multi-turn conversation memory** to answer follow-up questions intelligently.
* 📈 Provides per-model **performance analytics**: latency, throughput, and score trends.

---

## 🛠️ Installation

```bash
pip install -U boto3 langchain langchain-pinecone nltk sentence-transformers langchain-huggingface
```

Then download NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## 🔧 Configuration

Set your credentials:

```python
# AWS
AWS_ACCESS_KEY_ID = 'your-access-key'
AWS_SECRET_ACCESS_KEY = 'your-secret-key'
AWS_REGION = 'us-east-1'

# Pinecone
PINECONE_API_KEY = 'your-pinecone-key'
PINECONE_INDEX = 'your-index-name'

# S3
S3_BUCKET_NAME = 'your-s3-bucket'
PDF_FILE_NAME = 'your-document.pdf'
```

---

## 📄 Usage Instructions

### 1. 📤 Upload PDF to S3

Ensure your PDF is uploaded to the specified S3 bucket.

### 2. 📝 Extract Text Using AWS Textract

Text is extracted asynchronously using `start_document_text_detection`.

### 3. 🧽 NLP Preprocessing

The extracted text is cleaned using:

* Tokenization
* Stopword removal
* Lemmatization

### 4. 📐 Embedding & Storage

The preprocessed text is split into chunks and embedded using HuggingFace, and stored in Pinecone for retrieval.

### 5. 💬 Ask Questions via Interactive Loop

The system retrieves relevant chunks, generates answers from all 3 Titan models, evaluates their quality, and shows the best one.

---

## 🧠 AI Agent Behavior

* Maintains conversation history to handle **follow-up questions.**
* Prompts are enriched with both **retrieved context** and **dialogue history.**
* Uses a custom **prompt template** optimized for Titan models.

---

## 📊 Model Evaluation Metrics

| Metric             | Description                                                |
| ------------------ | ---------------------------------------------------------- |
| Query Similarity   | Semantic similarity between user question and model output |
| Context Similarity | Similarity between document context and model output       |
| Fluency Score      | Word count-based fluency score (normalized)                |
| Generation Time    | Time taken by model to generate output                     |
| Final Score        | Weighted average of all metrics (with time penalty)        |

---

## 🧪 Example Interaction

```
Ask a question (or type 'exit' to quit): If a company’s employee leaks customer data intentionally, which section(s) apply?

Answer from Express:
Section 72 of the IT Act applies to the company’s employee in case of intentional leaking of customer data.
```

### 🧮 Evaluation Summary

```
Model: Titan Express
Query Similarity:     0.714
Context Similarity:   0.295
Fluency:              High (0.95)
Generation Time:      3.14 sec
Total Time:           3.56 sec
Average Throughput:   0.065 responses/sec
```

---

## 📈 Performance Dashboard (Example)

| Metric                | Titan Express | Titan Premier | Titan Lite |
| --------------------- | ------------- | ------------- | ---------- |
| Query Similarity      | 0.734         | 0.729         | 0.704      |
| Context Similarity    | 0.510         | 0.498         | 0.490      |
| Fluency Score         | 0.980         | 0.940         | 0.910      |
| Final Score           | 0.698         | 0.690         | 0.675      |
| Embedding Time (s)    | 0.12          | 0.13          | 0.11       |
| Retrieval Time (s)    | 0.38          | 0.42          | 0.33       |
| Generation Time (s)   | 3.22          | 4.85          | 2.17       |
| Total Time (s)        | 3.72          | 5.40          | 2.61       |
| Avg. Throughput (r/s) | 0.28          | 0.18          | 0.38       |

---

## 📁 Output Files

* `extracted_text.txt`: raw text extracted from PDF
* `chat_history[]`: list of all user-assistant interactions
* `DataFrame Summary`: Printed table with all average metrics

---

## 📌 Requirements

* AWS Textract & Bedrock permissions
* Pinecone index with `cosine` similarity
* Python 3.8+
* NLTK, Sentence Transformers, LangChain, Pandas, NumPy

---

## 📚 Future Enhancements

* Add **UI/Streamlit frontend**
* Integrate **OpenSearch or FAISS** as vector DB alternatives
* Extend to **multiple PDFs** or **structured data (tables/forms)**
* Add **chat summarization** and **knowledge graph generation**
