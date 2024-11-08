from sentence_transformers import SentenceTransformer
import pinecone
#import streamlit as st
import os
from langchain import LLMChain, PromptTemplate
#from langchain.chains.LLMChain import LLMChain
#from langchain_core.prompts.PromptTemplate import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Path to the local SentenceTransformer model directory
local_st_model_path = './all-MiniLM-L6-v2'  # Adjust this path if needed

# Load the SentenceTransformer model from the local directory
model = SentenceTransformer(local_st_model_path)

# Path to the local Hugging Face model directory for OPT-125M
local_opt_model_path = './opt-125m'  # Adjust this path to point to your local OPT-125M model directory

# Initialize the local Hugging Face model for LangChain using the local OPT-125M model
llm_pipeline = pipeline(
    "text-generation",
    model=local_opt_model_path,  # Pointing to the local model directory
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

conversation_history = {
    #'responses': ["How can I assist you?"],
    'responses': [],
    'requests': []
}

def get_conversation_string():
    # Use the global conversation_history instead of st.session_state
    global conversation_history
    conversation_str = ""
    for i in range(len(conversation_history['responses']) - 1):
        conversation_str += f"Human: {conversation_history['requests'][i]}\n"
        conversation_str += f"AI: {conversation_history['responses'][i + 1]}\n"
    return conversation_str
