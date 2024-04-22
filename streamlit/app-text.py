# st_chatbot.py
import os

import streamlit as st

import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)

PROJECT_ID = os.environ.get("GCP_PROJECT")  
LOCATION = os.environ.get("GCP_REGION")  
vertexai.init(project=PROJECT_ID, location=LOCATION)


st.title("Gemini-Bot")

@st.cache_resource
def load_model():
    model = GenerativeModel("gemini-1.0-pro")
    print("model loaded...")
    return model

model = load_model()

if "chat_session" not in st.session_state:    
    st.session_state["chat_session"] = model.start_chat(history=[]) 

for content in st.session_state.chat_session.history:
    with st.chat_message("ai" if content.role == "model" else "user"):
        st.markdown(content.parts[0].text)

if prompt := st.chat_input("메시지를 입력하세요."):    
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("ai"):
        response = st.session_state.chat_session.send_message(prompt)        
        st.markdown(response.text)

