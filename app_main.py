import random
import time
import datetime

import streamlit as st
import numpy as np
from PIL import Image

from modules import add_speech as sp
from modules import add_embedding as emb
from modules import add_multimodal as mm
from modules import add_google_calendar as gc


### Initial setting
_PROJECT="primeval-argon-420311"
_INDEX_ENDPOINT="projects/854115243710/locations/us-central1/indexEndpoints/7122253694686461952"
_DEPLOYED_INDEX_ID="multimodal_embedding_endpoint"
llm_type = "gemini" # openai


######## Gemini
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
if "direct_llm" not in st.session_state:
    st.session_state["direct_llm"] = False
if "chat_session" not in st.session_state:
    st.session_state["chat_session"] = llm.start_chat(history=[])


# Accept user input
col1, col2 = st.columns([1, 12])


# Display image upload on app
with st.chat_message("system"):
    cols= st.columns((3,1,1))
    cols[0].write("Do you want to upload a photo that you want to know is smishing?")
    cols[1].button("yes", use_container_width=True, on_click=emb.show_upload, args=[True])
    cols[2].button("no", use_container_width=True, on_click=emb.show_upload, args=[False]) 
    
## upload image
file = None
if st.session_state["uploader_visible"]:
    fl_cols =  st.columns((1,1))
    fl_cols[0].button("Smishing test", on_click=emb.direct_llm, use_container_width=True, args=[True])
    fl_cols[1].button("Financial counseling", on_click=emb.direct_llm, use_container_width=True, args=[False])
    file = st.file_uploader("Upload your data", type=['png', 'jpg', 'jpeg'])    
    if not st.session_state["direct_llm"] and file:
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
        #st.markdown(content.parts[0].text)
        st.markdown(content.parts[0].text)


##### Speech
voice_prompt = None
with col1 :
    sp.speech_button()
with col2:
    # Voice input
    if st.session_state.mic:
        sp.voice_input_button()
        if st.session_state.play_stt:
            voice_prompt = sp.get_stt_input()
    
if voice_prompt :
    st.session_state.messages.append({"role": "user", "content": voice_prompt})
    with st.chat_message("user"):
        st.markdown(voice_prompt)

## image input
parts = []
img_prompt = None

if file : 
    img_prompt = mm.transform_file(file)
    parts.append(img_prompt)

#Text  input
chat_prompt = None
if chat_prompt := st.chat_input("What's up?") :
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": chat_prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(chat_prompt)

#### TODO 3. STT

# Display assistant response in chat message container
if chat_prompt :
    parts.insert(0, chat_prompt)
    if "take out a loan" in chat_prompt:
        today_date = datetime.datetime.now().strftime("%Y-%m-%d")
        gc.add_calendar(today_date)
        with st.chat_message("assistant"):
            st.markdown("I would add important dates in your google calander.:)")

if parts : 
        with st.chat_message("assistant"):
            if   len(st.session_state.messages) == 0 : # Initial message
                response = st.write_stream(response_generator())
                # TTS 
                sp.get_tts_output(response)
            else : # Multimodal
                response = st.session_state.chat_session.send_message(parts) 
                # TTS 
                sp.get_tts_output(response.text)
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response})
                

    

    
