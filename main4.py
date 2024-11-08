from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils2 import *
import os
import torch

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
st.title("Personal AI Bot")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Path to the local model directory
local_model_path = './chinese-bert-wwm'  # Adjust this to your local folder path

# Load the tokenizer and model with trust_remote_code=True
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
tokenizer.model_max_length = 512
model = AutoModelForCausalLM.from_pretrained(local_model_path, trust_remote_code=True)

# Define a preprocessing function to handle input lengths
def preprocess_input(input_text, max_length=512):
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )
    return inputs

# Initialize the pipeline with proper max_length and max_new_tokens settings
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,  # Use CPU (MPS or GPU if available)
    max_length=512,  # Adjust this to a reasonable value based on your needs
    max_new_tokens=150  # Adjust this as needed to control the length of the output
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = find_match(refined_query)
            
            # Preprocess the input to ensure it does not exceed the max length
            context_inputs = preprocess_input(context)
            query_inputs = preprocess_input(query)
            
            # Run the prediction using the preprocessed inputs
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
