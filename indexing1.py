import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
import pinecone
from langchain_community.vectorstores import Pinecone

# Load the PDF document
pdf_file = 'data/AI_Trends_2023-AI_Time_Journal_Ebook.pdf'

def load_pdf(pdf_file):
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    return documents

documents = load_pdf(pdf_file)

# Split the document into chunks
def split_docs(documents, chunk_size=500, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

docs = split_docs(documents)

# Create embeddings for the document chunks
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
doc_embeddings = [embeddings.embed_query(doc.page_content) for doc in docs]

pc = pinecone.Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))

# Check if the index exists; if not, create it
if 'langchain-chatbot' not in pc.list_indexes().names():
    pc.create_index(
        name='langchain-chatbot', 
        dimension=384,  # Adjust dimension according to your model
        metric='cosine',
        spec=pinecone.ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Connect to the index
index = pc.Index('langchain-chatbot')

# Store the embeddings in Pinecone
for i, embedding in enumerate(doc_embeddings):
    index.upsert([(f'doc_{i}', embedding, {'text': docs[i].page_content})])

print("Document embeddings have been stored in Pinecone.")
