import random
import time

import streamlit as st
import numpy as np
from PIL import Image

from . import add_speech as sp
from . import add_embedding as emb
from . import add_multimodal as mm
from . import add_google_calendar as gc

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



class image2txt():
    def __init__(self, config ):
        self._PROJECT = config['_PROJECT']
        self._INDEX_ENDPOINT = config['_INDEX_ENDPOINT']
        self._DEPLOYED_INDEX_ID = config['_DEPLOYED_INDEX_ID']

    def run(self) :
        ## Upload image / Smishing test
        file = None
        if st.session_state["uploader_visible"]:
            # fl_cols =  st.columns((1,1))
            # fl_cols[0].button("Smishing test", on_click=emb.direct_llm, use_container_width=True, args=[True])
            # fl_cols[1].button("Financial counseling", on_click=emb.direct_llm, use_container_width=True, args=[False])
            file = st.file_uploader("Upload your data", type=['png', 'jpg', 'jpeg'])    

        ## Speech
        # Accept user input
        col1, col2 = st.columns([12, 1])

        voice_prompt = None
        with col1 :
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
        chat_prompt = voice_prompt
        if chat_prompt == None :
            chat_prompt =  st.chat_input("What's up?")
        
        if chat_prompt :
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": chat_prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(chat_prompt)

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
                   # sp.get_tts_output(response)
                else : # Multimodal
                    response = st.session_state.chat_session.send_message(parts) 
                    # TTS 
                    #sp.get_tts_output(response.text)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response})
              

class image2txtsmishing():
    def __init__(self, config ):
        self._PROJECT = config['_PROJECT']
        self._INDEX_ENDPOINT = config['_INDEX_ENDPOINT']
        self._DEPLOYED_INDEX_ID = config['_DEPLOYED_INDEX_ID']

    def run(self) :
        ## Upload image / Smishing test
        file = None
        if st.session_state["uploader_visible"]:
            # fl_cols =  st.columns((1,1))
            # fl_cols[0].button("Smishing test", on_click=emb.direct_llm, use_container_width=True, args=[True])
            # fl_cols[1].button("Financial counseling", on_click=emb.direct_llm, use_container_width=True, args=[False])
            file = st.file_uploader("Upload your data", type=['png', 'jpg', 'jpeg'])    

        ## Speech
        # Accept user input
        col1, col2 = st.columns([12, 1])

        voice_prompt = None
        with col1 :
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
        chat_prompt = voice_prompt
        if chat_prompt == None :
            chat_prompt =  st.chat_input("What's up?")
        
        # if chat_prompt :
        #     # Add user message to chat history
        #     st.session_state.messages.append({"role": "user", "content": chat_prompt})
        #     # Display user message in chat message container
        #     with st.chat_message("user"):
        #         st.markdown(chat_prompt)

        #     parts.insert(0, chat_prompt)
        #     if "take out a loan" in chat_prompt:
        #         today_date = datetime.datetime.now().strftime("%Y-%m-%d")
        #         gc.add_calendar(today_date)
        #         with st.chat_message("assistant"):
        #             st.markdown("I would add important dates in your google calander.:)")

        ###### Smishing Test / 
        # 1. smism_prompt and return / Don't get user input
        smishing_prompt = """
            You are a professional banker. Image of a captured mobile screen will be 
            given as an input. Find out if the message is a smishing text or not. Alert the
            user it the image is a smishsing image.
        """
        #import pdb; pdb.set_trace()
        time.sleep(3)
        parts.insert(0, smishing_prompt)
        if parts :
            with st.chat_message("assistant"):
                if   len(st.session_state.messages) == 0 : # Initial message
                    response = st.write_stream(response_generator())
                    # TTS 
                   # sp.get_tts_output(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                else : # Multimodal
                    response = st.session_state.chat_session.send_message(parts) 
                    # TTS 
                    #sp.get_tts_output(response.text)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response})
 


class image2geminirag():
    def __init__(self, config ):
        self._PROJECT = config['_PROJECT']
        self._INDEX_ENDPOINT = config['_INDEX_ENDPOINT']
        self._DEPLOYED_INDEX_ID = config['_DEPLOYED_INDEX_ID']

    def run(self) :
        ## Upload image / Smishing test
        file = None
        if st.session_state["uploader_visible"]:
            fl_cols =  st.columns((1,1))
            fl_cols[0].button("Smishing test", on_click=emb.direct_llm, use_container_width=True, args=[True])
            fl_cols[1].button("Financial counseling", on_click=emb.direct_llm, use_container_width=True, args=[False])
            file = st.file_uploader("Upload your data", type=['png', 'jpg', 'jpeg'])    


            ## RAG Output
            result = []
            if file :
                img_prompt = mm.transform_file(file)
                ## Capture goods name with gemini
                parse_prompt = """
                you will be given a captured chatting image on a mobile screen. give me the name of the financial goods in the following chat image. answer in less than 10 words in short word format
                """
                parts = [parse_prompt, img_prompt]
                response = st.session_state.chat_session.send_message(parts)   
                goods_name = response.text
                embedding_prompt = """You are a banker assisting seniors. Return a short explanation of the following account named by searching in the vector database
                
                account_name : {}""".format(goods_name)
                result = emb.get_emb_result_text(embedding_prompt)
                if len(result)>=1:
                    with st.chat_message("assistant"):
                        st.write(result)
                        #st.image(result, width=300)                    
                else:
                    with st.chat_message("assistant"):
                        st.write("Smishing is not suspected.")

        
        ## Speech
        # Accept user input
        col1, col2 = st.columns([12, 1])

        voice_prompt = None
        with col1 :
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
        chat_prompt = voice_prompt
        if chat_prompt == None :
            chat_prompt =  st.chat_input("What's up?")
        
        if chat_prompt :
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": chat_prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(chat_prompt)

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
                   # sp.get_tts_output(response)
                else : # Multimodal
                    response = st.session_state.chat_session.send_message(parts) 
                    # TTS 
                    #sp.get_tts_output(response.text)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response})



class image2rag():
    def __init__(self, config ):
        self._PROJECT = config['_PROJECT']
        self._INDEX_ENDPOINT = config['_INDEX_ENDPOINT']
        self._DEPLOYED_INDEX_ID = config['_DEPLOYED_INDEX_ID']
        
    def run(self) :
        ## Upload image / Smishing test
        file = None
        if st.session_state["uploader_visible"]:
            fl_cols =  st.columns((1,1))
            fl_cols[0].button("Smishing test", on_click=emb.direct_llm, use_container_width=True, args=[True])
            fl_cols[1].button("Financial counseling", on_click=emb.direct_llm, use_container_width=True, args=[False])
            file = st.file_uploader("Upload your data", type=['png', 'jpg', 'jpeg'])    


            ## RAG Output
            result = []
            if file :
                img_prompt = mm.transform_file(file)
                # ## Capture goods name with gemini
                # parse_prompt = """
                # you will be given a captured chatting image on a mobile screen. 
                # give me the name of the financial goods in the following chat image. 
                # Answer in less than 10 words in short word format.
                # """
                # parts = [parse_prompt, img_prompt]
                # response = st.session_state.chat_session.send_message(parts)   
                # goods_name = response.text

                # embedding_prompt = """You are a banker assisting seniors. Return a short explanation of the following account named by searching in the vector database
                # account_name : {}""".format(goods_name)
                
                result = emb.get_emb_result_image(file.getvalue())
            if len(result)>=1:
                with st.chat_message("assistant"):
                    st.write("Your anwser :")
                    st.image(result, width=600)                    
            # else:
            #     with st.chat_message("assistant"):
            #         st.write("Smishing is not suspected.")

        
        ## Speech
        # Accept user input
        col1, col2 = st.columns([12, 1])

        voice_prompt = None
        with col1 :
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
        chat_prompt = voice_prompt
        if chat_prompt == None :
            chat_prompt =  st.chat_input("What's up?")
        
        if chat_prompt :
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": chat_prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(chat_prompt)

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



if __name__ == "__main__" :
    print("good")
    