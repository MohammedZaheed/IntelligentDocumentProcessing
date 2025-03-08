import os
import json
import time
import boto3
import pinecone
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Load environment variables
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# AWS Textract client
client = boto3.client(
    "textract",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# Start document text detection
response = client.start_document_text_detection(
    DocumentLocation={"S3Object": {"Bucket": "mybucketforprojects", "Name": "CyberCrime.pdf"}}
)
job_id = response["JobId"]
print(f"Job started with Job ID: {job_id}")

# Polling for job completion
while True:
    result = client.get_document_text_detection(JobId=job_id)
    status = result["JobStatus"]
    
    if status in ["SUCCEEDED", "FAILED"]:
        break

    print("Processing...")
    time.sleep(5)

if status == "FAILED":
    raise Exception("Textract job failed!")

print("Processing completed!")

# Extract text from response
extracted_text = []
while True:
    if "Blocks" in result:
        for block in result["Blocks"]:
            if block["BlockType"] == "LINE" and "Text" in block:
                extracted_text.append(block["Text"])

    if "NextToken" in result:
        result = client.get_document_text_detection(JobId=job_id, NextToken=result["NextToken"])
    else:
        break

# Save extracted text to a file
full_text = "\n".join(extracted_text)
with open("demo_rag_on_image.txt", "w") as output_file_io:
    output_file_io.write(full_text)

# NLP Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

preprocessed_text = preprocess_text(full_text)

# Prepare document for embedding
docs = [Document(page_content=preprocessed_text)]

# Split document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=250, separator="\n")
split_docs = text_splitter.split_documents(docs)

# Use sentence-transformers embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1")

# Initialize Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment="us-east1-gcp")  # Change to your Pinecone environment

docsearch = PineconeVectorStore.from_documents(split_docs, embedding_model, index_name=PINECONE_INDEX)

print("Processing complete!")

# Query processing
human_input = input("Enter your question: ")
query_embedding = embedding_model.embed_query(human_input)

existing_search = PineconeVectorStore.from_existing_index(index_name=PINECONE_INDEX, embedding=embedding_model)
search_results = existing_search.similarity_search(human_input, k=5)

# Retrieve relevant context
MAX_CONTEXT_LENGTH = 6000
context_string = "\n\n".join([f"Document {ind+1}: " + i.page_content[:MAX_CONTEXT_LENGTH] for ind, i in enumerate(search_results)])

# Define RAG prompt template
RAG_PROMPT_TEMPLATE = """You are a helpful and knowledgeable AI assistant. Use the provided context to answer the question.

If the context is insufficient, rely on your own knowledge to provide the best possible response.

Context:
{context}

Question: {human_input}

Answer:
"""

PROMPT = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
prompt_data = PROMPT.format(human_input=human_input, context=context_string)

# AWS Bedrock client for Amazon Titan
boto3_bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

body_part = json.dumps({
    "inputText": prompt_data,
    "textGenerationConfig": {"maxTokenCount": 8192, "stopSequences": [], "temperature": 0.7, "topP": 1}
})

response = boto3_bedrock.invoke_model(
    body=body_part,
    contentType="application/json",
    accept="application/json",
    modelId="amazon.titan-text-express-v1"
)

output_text = json.loads(response["body"].read())["results"][0]["outputText"]
output_text = output_text.replace(". ", ".\n")
print(f"Answer:\n{output_text}")
