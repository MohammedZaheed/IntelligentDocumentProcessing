# Intelligent Document Processing with AI-Powered Agent

## Overview
This project extracts text from PDF documents stored in Amazon S3 using AWS Textract, preprocesses the text with NLP techniques, stores vector embeddings in Pinecone, and enables context-based question-answering using Amazon Titan LLM. It supports multi-turn conversation with memory, functioning as an intelligent AI agent.

## Features
- Extracts text from PDF documents stored in Amazon S3 using **AWS Textract**.
- Preprocesses text using **NLTK** (tokenization, stopword removal, lemmatization).
- Converts text into embeddings with Hugging Face's `multi-qa-mpnet-base-cos-v1` model.
- Stores and retrieves embeddings via **Pinecone**.
- Answers questions using **Amazon Titan LLM** via a retrieval-augmented generation (RAG) approach.
- Maintains conversation context with memory to enable follow-up questions.

## Installation

### Install Required Packages
```bash
pip install -U boto3 langchain langchain-pinecone langchain-community nltk sentence-transformers langchain-huggingface
```

### Download NLTK Resources
Before running the script, download the required NLTK components:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Set Up Required Credentials
Before running, update the following values in your script:
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
2. **Run the script** to extract and preprocess text using AWS Textract and NLTK.
3. **Generate embeddings** using the Hugging Face sentence-transformer and store them in Pinecone.
4. **Start the conversation loop** and interact with the AI agent by asking questions. The agent will provide answers based on the document context and previous conversation history.

## AI Agent Behavior
This system uses memory to maintain context over multiple interactions. For example:

```
User: What was the net revenue of Acme Corp in 2023?
AI: Acme Corp reported a net revenue of $5.2 million in 2023.

User: What contributed to that growth?
AI: The main contributors were increased product sales in the North American market and the successful launch of their AI-driven analytics platform.

User: Was there any decline in other segments?
AI: Yes, there was a slight decline in the hardware division due to supply chain issues.
```

The AI remembers your previous questions to enhance context and answer follow-ups accordingly.

## Example Use Case
Assume your document is an annual report for a company. You might ask questions like:
- "What was the profit in 2022?"
- "Which sectors performed best this year?"
- "How does this year’s performance compare to last year?"
  
The agent uses the document context along with conversation memory to provide accurate, context-aware answers.

## Future Enhancements
- Develop a Graphical User Interface (GUI) or REST API for easier interaction.
- Support multiple documents simultaneously.
- Integrate with enterprise document processing pipelines.
