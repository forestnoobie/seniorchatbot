import time
import random
import argparse
import datetime

from streamlit_option_menu import option_menu
import streamlit as st
import numpy as np
from PIL import Image


from modules import add_speech as sp
from modules import add_embedding as emb
from modules import add_multimodal as mm
from modules import add_google_calendar as gc
from modules.scenario import image2txt, image2rag, image2geminirag, image2txtsmishing


if __name__  == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='primeval-argon-420311')
    parser.add_argument('--index_endpoint', 
                        default=
                        'projects/854115243710/locations/us-central1/indexEndpoints/1709912105005613056')
    parser.add_argument('--deployed_index_id', default='multimodal_embedding_endpoint')

    args = parser.parse_args()
    
    # Initial setting
    config = {
        "_PROJECT" : args.project,
        "_INDEX_ENDPOINT" : args.index_endpoint,
        "_DEPLOYED_INDEX_ID" : args.deployed_index_id    
    }
    
    
    # Streamlit
    st.set_page_config(page_title="Silver Aid", 
                       page_icon=":robot_face:",
                       layout="wide",
                       )
    
    
    st.image("./images/sa_logo.png", width=200)
    st.sidebar.image("./images/logo_long.png")
    with st.sidebar:
        choice = option_menu("Features", ["Default", "Smishing test", "Ragtext", "Ragimage"],
                             icons=['house', 'kanban','bi bi-card-text', 'bi bi-image', "bi bi-zoom-in"],
                             menu_icon="app-indicator", default_index=0, 
                             styles={
            "container": {"padding": "4!important", "background-color": "#08619b"},
            "icon": {"color": "black", "font-size": "25px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#fafafa"},
            "nav-link-selected": {"background-color": "#08c7b4"},
        }
        )
    
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
        st.session_state["chat_session"] = llm.start_chat(history=[], response_validation=False)
    
    # Accept user input
    # Display image upload on app
    expander = st.expander(label = "Advanced tools 🛠️")
    with expander :
        cols= st.columns((1,1))
        cols[0].button("Voice assistant", use_container_width=True, on_click=sp.click_microphone)
        cols[1].button("Upload image", use_container_width=True, on_click=emb.show_upload)
    
    if choice == "Default" :
        bot = image2txt(config)
    elif choice == "Smishing test":
        bot = image2txtsmishing(config)
    elif choice == "Ragtext":
        bot = image2geminirag(config)
    elif choice == "Ragimage":
        bot = image2rag(config)
    
    
    bot.run()
    
    


    
