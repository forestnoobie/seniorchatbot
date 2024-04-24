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
from modules.scenario import image2txt, image2rag


### Initial setting
config = {
    "_PROJECT" : "primeval-argon-420311",
    "_INDEX_ENDPOINT" : "projects/854115243710/locations/us-central1/indexEndpoints/7122253694686461952",
    "_DEPLOYED_INDEX_ID" : "multimodal_embedding_endpoint"    
}

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
    
        
# Display chat messages from history on app rerun
for content in st.session_state.chat_session.history:
    with st.chat_message("assistant" if content.role == "model" else "user"):
        #st.markdown(content.parts[0].text)
        st.markdown(content.parts[0].text)

#bot = image2txt(config)
bot = image2rag(config)
bot.run()




    
