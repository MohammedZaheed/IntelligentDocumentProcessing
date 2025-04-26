# Intelligent Document Processing with AI-Powered Agent

## Overview
This project extracts text from PDF documents stored in Amazon S3 using AWS Textract, preprocesses it with NLP techniques, stores vector embeddings in Pinecone, and enables intelligent document-based question-answering using Amazon Titan LLMs.  
It automatically evaluates multiple model outputs (Titan Express and Titan Premier) and selects the best answer based on similarity and fluency scoring.

It supports multi-turn conversation with memory, functioning as an intelligent AI agent.

## Features
- Extracts text from PDF documents stored in Amazon S3 using **AWS Textract**.
- Preprocesses text using **NLTK** (tokenization, stopword removal, lemmatization).
- Converts text into embeddings with **Hugging Face's multi-qa-mpnet-base-cos-v1** model.
- Stores and retrieves document embeddings via **Pinecone**.
- Answers questions using a **retrieval-augmented generation (RAG)** approach with **Amazon Titan Express** and **Amazon Titan Premier** models.
- **Evaluates and selects** the best model response based on query similarity, context similarity, and fluency.
- Maintains conversation memory to enable follow-up questions and natural dialogue flow.

## Installation

### Install Required Packages
```bash
pip install -U boto3 langchain langchain-pinecone langchain-community nltk sentence-transformers langchain-huggingface
```

### Download NLTK Resources
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Set Up Required Credentials
Before running, configure your credentials and settings:

```python
AWS_ACCESS_KEY_ID = 'your-aws-access-key-id'
AWS_SECRET_ACCESS_KEY = 'your-aws-secret-access-key'
AWS_REGION = 'us-east-1'

PINECONE_API_KEY = 'your-pinecone-api-key'
PINECONE_INDEX = 'your-pinecone-index-name'

S3_BUCKET_NAME = 'your-s3-bucket-name'
PDF_FILE_NAME = 'your-document.pdf'
```

## Usage

1. **Upload your PDF document** to your S3 bucket.
2. **Run the script** to extract text using AWS Textract.
3. **Preprocess the extracted text** using NLP techniques.
4. **Generate vector embeddings** and **store them** in Pinecone.
5. **Ask questions** in a loop.  
   The system:
   - Retrieves relevant document context,
   - Queries **Titan Express** and **Titan Premier** models,
   - **Evaluates** their answers automatically,
   - **Selects and displays the best** response to you,
   - **Maintains chat history** to improve follow-up answers.

## AI Agent Behavior
This AI system remembers your previous questions and responses to maintain context throughout the conversation.  
It uses both document knowledge and prior dialogue to answer your follow-up questions intelligently.

**Example Conversation:**
```
Ask a question (or type 'exit' to quit): what is section 66A

Answer from Premier:
Section 66A of the Information Technology Act, 2000 of India was a provision that defined the punishment for sending offensive messages through communication services.

[Evaluation Summary]
Query Similarity: 0.450
Context Similarity: 0.515
Fluency: High (0.220)
Final Score: 0.426
```

The system automatically evaluates:
- **Query Similarity** (How close the answer is to the question),
- **Context Similarity** (How well the answer matches the document context),
- **Fluency** (Answer quality in English),
- and selects the highest scoring answer.

## Example Use Case
Assume your document is an annual company report. You might ask:
- "What was the revenue in 2022?"
- "What led to the increase?"
- "Were there any divisions that underperformed?"

The agent will pull context from the document, maintain memory of previous queries, and continuously validate the best answer for you.

## Future Enhancements
- Build a Graphical User Interface (GUI) or REST API for easier interaction.
- Support querying across **multiple documents**.
- Improve conversation evaluation with more sophisticated scoring strategies.
- Integrate into enterprise document pipelines.
