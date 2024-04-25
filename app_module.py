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
from modules.scenario import image2txt, image2rag, image2geminirag, image2txtsmishing


### Initial setting
config = {
    "_PROJECT" : "primeval-argon-420311",
    #"_INDEX_ENDPOINT" : "projects/854115243710/locations/us-central1/indexEndpoints/7122253694686461952",
    "_INDEX_ENDPOINT" : "projects/854115243710/locations/us-central1/indexEndpoints/1709912105005613056",
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

st.set_page_config(page_title="Ideas Bank Senior supperter (IDS)", 
                   page_icon=":robot_face:",
                   layout="wide",
                   #initial_sidebar_state="expanded"
                   )

st.header("Ideas Bank Senior supperter (IDS) :robot_face:")
# page_bg_img = f"""
# <style>
# [data-testid="stAppViewContainer"] > .main {{
# background-image: url("https://i.postimg.cc/4xgNnkfX/Untitled-design.png");
# background-size: cover;
# background-position: center center;
# background-repeat: no-repeat;
# background-attachment: local;
# }}
# [data-testid="stHeader"] {{
# background: rgba(0,0,0,0);
# }}
# </style>
# """

#st.markdown(page_bg_img, unsafe_allow_html=True)
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
# Display image upload on app
expander = st.expander(label = "Advnaced tools")
with expander :
    cols= st.columns((1,1))
    cols[0].button("Voice assistant", use_container_width=True, on_click=sp.click_microphone)
    cols[1].button("Upload image", use_container_width=True, on_click=emb.show_upload, args=[True])


# with st.form('my_form'):
#     text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
#     submitted = st.form_submit_button('Submit')
# if submitted : 
#     st.warning('Smishing Warning!!',  icon='⚠')
#     with st.chat_message("assistant"):

#         st.write("Write test")
#         st.info("hello")
        

        
# Display chat messages from history on app rerun
# for content in st.session_state.chat_session.history:
#     with st.chat_message("assistant" if content.role == "model" else "user"):
#         #st.markdown(content.parts[0].text)
#         st.markdown(content.parts[0].text)

#bot = image2txt(config)
bot = image2rag(config)
#bot = image2geminirag(config)
#bot = image2txtsmishing(config)
bot.run()




    
