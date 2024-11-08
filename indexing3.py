import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
import pinecone

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

pc = pinecone.Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))

# Check if the index exists; if not, create it
if 'langchain-chatbot2' not in pc.list_indexes().names():
    pc.create_index(
        name='langchain-chatbot2', 
        dimension=384,  # Adjust dimension according to your model
        metric='cosine',
        spec=pinecone.ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Connect to the index
index = pc.Index('langchain-chatbot2')

# Store the embeddings in Pinecone
for i, embedding in enumerate(doc_embeddings):
    index.upsert([(f'doc_{i}', embedding, {'text': docs[i]})])

print("Document embeddings have been stored in Pinecone.")
