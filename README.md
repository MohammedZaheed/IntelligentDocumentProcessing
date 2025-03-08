# Intelligent Document Processing with AI-Powered Query System

## Overview
This project extracts text from documents stored in Amazon S3 using AWS Textract, preprocesses the text with NLP techniques, stores embeddings in Pinecone, and enables retrieval-augmented generation (RAG) using Amazon Titan LLM for querying.

## Features
- Extracts text from PDF documents stored in Amazon S3 using AWS Textract.
- Processes text using NLTK (tokenization, stopword removal, lemmatization).
- Embeds text using a Hugging Face sentence-transformer model.
- Stores and retrieves embeddings using Pinecone.
- Uses Amazon Titan LLM to generate answers to user queries based on retrieved context.
- Future Integration: AI agent for enhanced query processing.

## Requirements
### Install Dependencies
```bash
pip install boto3 langchain langchain-pinecone langchain-community nltk sentence-transformers
```

### Set Up Required Credentials
Before running the script, update the following credentials in the script:
```python
AWS_ACCESS_KEY_ID = ''  # Replace with your AWS Access Key ID
AWS_SECRET_ACCESS_KEY = ''  # Replace with your AWS Secret Access Key
AWS_REGION = 'us-east-1'  # AWS region (default: us-east-1)
PINECONE_API_KEY = ''  # Replace with your Pinecone API Key
PINECONE_INDEX = ''  # Replace with your Pinecone Index Name
S3_BUCKET_NAME = ''  # S3 bucket name where the document is stored
PDF_FILE_NAME = ''  # Filename of the document to process
```

## Usage
1. Upload your document (PDF) to an S3 bucket.
2. Run the script to extract and preprocess the text.
3. The text is embedded and stored in Pinecone for retrieval.
4. Enter your query when prompted, and the system will generate an answer using Amazon Titan LLM.

## Future Enhancements
- **AI Agent Integration:** Plan to integrate an AI agent for more advanced query processing and automation.
- **Multi-document Support:** Extend capabilities to handle multiple documents.
- **GUI or API Development:** Create a user-friendly interface or API for easier access.

## License
This project is licensed under [MIT License](LICENSE).
