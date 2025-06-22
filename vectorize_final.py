from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
import params
import json

# MongoDB connection
client = MongoClient(params.mongodb_conn_string)
db = client[params.db_name]
collection = db[params.collection_name]

# First load and process the JSON file
with open('final.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=params.openai_api_key)

# Create vector store
vector_store = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name=params.index_name
)

# Process each document
print(f"Processing {len(data)} documents...")
for i, doc in enumerate(data):
    # Create embedding for the document
    try:
        # Extract content and metadata
        content = doc.get('answer', '')
        metadata = {
            'question': doc.get('question', ''),
            'category': doc.get('category', ''),
            'user_type': doc.get('user_type', ''),
            'priority': doc.get('priority', 1)
        }
        
        # Skip if no content
        if not content:
            continue
            
        # Add document to vector store
        vector_store.add_texts(
            texts=[content],
            metadatas=[metadata]
        )
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} documents...")
            
    except Exception as e:
        print(f"Error processing document {i}: {str(e)}")

print("Vectorization completed successfully!")