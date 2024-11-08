from sentence_transformers import SentenceTransformer
import pinecone
import streamlit as st
import os
from langchain import PromptTemplate, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Path to the local model directory
local_model_path = './all-MiniLM-L6-v2'  # Adjust this path if needed

# Load the model from the local directory
model = SentenceTransformer(local_model_path)

# Initialize the local Hugging Face model for LangChain
# Initialize the local Hugging Face model for LangChain, adjusting max_length and max_new_tokens
llm_pipeline = pipeline(
    "text-generation",
    model="facebook/opt-125m",  # Adjust this to your specific model
    device=-1,  # Use CPU; change to 0 if using GPU
    max_length=1000,  # Increase this value to accommodate longer inputs
    max_new_tokens=100  # Limit the number of new tokens to generate
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

template = """Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.
CONVERSATION LOG:
{conversation}
Query: {query}
Refined Query:"""

prompt = PromptTemplate(input_variables=["conversation", "query"], template=template)
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))

# Check if the index exists; if not, create it
if 'langchain-chatbot1' not in pc.list_indexes().names():
    pc.create_index(
        name='langchain-chatbot1', 
        dimension=384,  # Adjust dimension according to your model
        metric='cosine',
        spec=pinecone.ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Connect to the index
index = pc.Index('langchain-chatbot1')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(
        vector=input_em,
        top_k=2,
        include_metadata=True
    )
    
    matches = result.get('matches', [])
    
    if len(matches) == 0:
        return "No matches found."
    elif len(matches) == 1:
        return matches[0]['metadata']['text']
    else:
        return matches[0]['metadata']['text'] + "\n" + matches[1]['metadata']['text']

def query_refiner(conversation, query):
    refined_query = llm_chain.run(conversation=conversation, query=query)
    print(refined_query)
    return refined_query

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string
