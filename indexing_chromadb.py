import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
import chromadb

# Load the text document
txt_file = 'data/domod.txt'

def load_txt(txt_file):
    with open(txt_file, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

document = load_txt(txt_file)

# Split the document into chunks
def split_docs(document, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_text(document)
    return docs

docs = split_docs(document)

# Create embeddings for the document chunks
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
doc_embeddings = [embeddings.embed_query(doc) for doc in docs]

# Initialize ChromaDB client
client = chromadb.Client()

# Create a collection to store embeddings
collection = client.create_collection(
    name="langchain-chatbot2",
    embedding_dimension=384  # Ensure this matches your model's output dimension
)

# Store the embeddings in ChromaDB
for i, (doc, embedding) in enumerate(zip(docs, doc_embeddings)):
    collection.insert_one(
        id=f'doc_{i}',
        embedding=embedding,
        metadata={'text': doc}
    )

print("Document embeddings have been stored in ChromaDB.")
