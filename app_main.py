import random
import time

import streamlit as st
from langchain import PromptTemplate
from langchain.llms import OpenAI
import numpy as np
from PIL import Image

from modules import add_speech as sp
from modules import add_embedding as emb
from modules import add_multimodal as mm


### Initial setting
_PROJECT="primeval-argon-420311"
_INDEX_ENDPOINT="projects/854115243710/locations/us-central1/indexEndpoints/7122253694686461952"
_DEPLOYED_INDEX_ID="multimodal_embedding_endpoint"
llm_type = "gemini" # openai




######## Gemini
# def load_LLM(openai_api_key):
#     """Logic for loading the chain you want to use should go here."""
#     # Make sure your openai_api_key is set as an environment variable]
#     ######## TODO 1. OpenAI -> Gemini
#     llm = OpenAI(temperature=.7, openai_api_key=openai_api_key)
#     return llm

def load_gemini():
    model = get_gemini()
    return model


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


##### Streamlit

st.set_page_config(page_title="(working title) Idea's Bank Senior supporter", page_icon=":robot:")
st.header("(working title) Idea's Bank Senior supporter")



openai_api_key = "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"

col1, col2 = st.columns(2)


## load llm
#llm = load_LLM(openai_api_key=openai_api_key)
llm = mm.get_gemini()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mic" not in st.session_state:
    st.session_state.mic = False
if "play_stt" not in st.session_state:
    st.session_state.play_stt = False
if "uploader_visible" not in st.session_state:
    st.session_state["uploader_visible"] = False
if "chat_session" not in st.session_state:
    st.session_state["chat_session"] = llm.start_chat(history=[])


# Display image upload on app
with st.chat_message("system"):
    cols= st.columns((3,1,1))
    cols[0].write("Do you want to upload a photo that you want to know is smishing?")
    cols[1].button("yes", use_container_width=True, on_click=emb.show_upload, args=[True])
    cols[2].button("no", use_container_width=True, on_click=emb.show_upload, args=[False]) 
    
## upload image
file = None
if st.session_state["uploader_visible"]:
    file = st.file_uploader("Upload your data", type=['png', 'jpg', 'jpeg'])    
    if file:
        with st.spinner("Processing your file"):
            
             # To read file as bytes:
            bytes_data = file.getvalue()          
            embedding_client = emb.EmbeddingPredictionClient.getInstance(project = _PROJECT)
            vectorsearch_client = emb.VectorSearchClient.getInstance(index_endpoint = _INDEX_ENDPOINT, deployed_index_id = _DEPLOYED_INDEX_ID)
            image_embedding = embedding_client.get_embedding(image_bytes = bytes_data).image_embedding

            # run query   
            
            # response = vectorsearch_client.find_neighbors(
            #         query = image_embedding,
            #         num_neighbors = 3
            # )
            # #client to access GCS bucket
            # storage_client = StorageClient.getInstance(project = _PROJECT)
            # bucket = storage_client.get_bucket(bucket = "smishing-image")
    
            result = [bytes_data]
            
            
        # if response :
        #     for r in response[0]:
        #         name = r.id
        #         name = name.split("/")[-1]
        #         # r.distance
        #         result.append(bucket.blob(name).download_as_bytes())
        #         #st.write(result)
    
        if len(result)>=1:
            with st.chat_message("assistant"):
                st.write("Smishing is suspected. Related photos are as follows:")
                st.image(result, width=300)                    
        else:
            with st.chat_message("assistant"):
                st.write("Smishing is not suspected.")

        

# Display chat messages from history on app rerun
for content in st.session_state.chat_session.history:
    with st.chat_message("assistant" if content.role == "model" else "user"):
        st.markdown(content.parts[0].text)




# Accept user input

##### Speech
with col1 :
    sp.speech_button()
with col2:
    # Voice input
    if st.session_state.mic:
        sp.voice_input_button()
        prompt = None
        if st.session_state.play_stt:
            prompt = get_stt_input()
            if prompt is not None:
                st.code(prompt, language="markdown")


file_gemini = None
file_gemini = st.file_uploader("Upload your data", type=['png', 'jpg', 'jpeg'])    


#Text  input
if prompt := st.chat_input("What's up?") :
    parts = [prompt]


    
    if file_gemini :
        parts.append(mm.transform_file(file_gemini))

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
            response = st.session_state.chat_session.send_message(parts) 
            st.markdown(response.text)
    #### TODO 5. TTS
        #sp.get_tts_output(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

