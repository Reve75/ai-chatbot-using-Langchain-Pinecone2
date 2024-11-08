from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st
import os
from zhipuai import ZhipuAI

#openai.api_key = "" ## find at platform.openai.com
#model = SentenceTransformer('all-MiniLM-L6-v2')

# Path to the local model directory
local_model_path = './all-MiniLM-L6-v2'  # Adjust this path if needed

# Load the model from the local directory
model = SentenceTransformer(local_model_path)

client = ZhipuAI(
    api_key=os.environ['ZHIPUAI_API_KEY'],  # Set your API key
)

#pinecone.init(api_key=os.environ['PINECONE_API_KEY'])
#index = pinecone.Index('langchain-chatbot')

# Initialize Pinecone
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


#def find_match(input):
#    input_em = model.encode(input).tolist()
#    result = index.query(input_em, top_k=2, includeMetadata=True)
#    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

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



#def query_refiner(conversation, query):

#    response = openai.Completion.create(
#    model="text-davinci-003",
#    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
#    temperature=0.7,
#    max_tokens=256,
#    top_p=1,
#    frequency_penalty=0,
#    presence_penalty=0
#    )
#    return response['choices'][0]['text']

def query_refiner(conversation, query):
    response = client.chat.completions.create(
        model="glm-4",  # Adjust the model name if needed
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"}
        ],
        extra_body={"temperature": 0.7, "max_tokens": 256}
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string
