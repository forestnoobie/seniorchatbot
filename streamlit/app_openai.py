import streamlit as st
from langchain import PromptTemplate
from langchain.llms import OpenAI
import numpy as np
from PIL import Image

import random
import time

template = """
    You are banker specialized for assiting old seniors who are older than 60.
    Please answer the customers questions.

    Below is the query:
    query: {query}
    
    YOUR RESPONSE:
"""

prompt = PromptTemplate(
    input_variables=["query"],
    template=template,
)

def load_LLM(openai_api_key):
    """Logic for loading the chain you want to use should go here."""
    # Make sure your openai_api_key is set as an environment variable
    ######## TODO 1. OpenAI -> Gemini
    llm = OpenAI(temperature=.7, openai_api_key=openai_api_key)
    return llm

st.set_page_config(page_title="(working title) Idea's Bank Senior supporter", page_icon=":robot:")
st.header("(working title) Idea's Bank Senior supporter")



openai_api_key = "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"

col1, col2 = st.columns(2)


# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Hello there! I am elderly-friendly chatbot how can I assist you today?",
            "Hi, human! Do you have trouble with any banking service? I can lend a hand for you",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

def llm_response_generator(query, llm):

    #### TODO 2. ChatLLM / LLMQA
    prompt_with_query = prompt.format(query=query)
    formatted_query = llm(prompt_with_query)

    for word in formatted_query.split():
        yield word + " "
        time.sleep(0.05)



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

## load llm
llm = load_LLM(openai_api_key=openai_api_key)

## Load image
img_file_buffer = st.file_uploader('Upload a PNG image', type='png')
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image)


# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    #### TODO 3. STT

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        if   len(st.session_state.messages) == 0 : # Initial message
            response = st.write_stream(response_generator())
        else :
            #### TODO 4. Fraud detection + MultiModal
            response = st.write_stream(llm_response_generator(prompt, llm))

    #### TODO 5. TTS

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

