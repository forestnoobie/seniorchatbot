import os
import streamlit as st
from PIL import Image
import base64
import io

import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)

from vertexai import generative_models

PROJECT_ID = os.environ.get("GCP_PROJECT")  
LOCATION = os.environ.get("GCP_REGION")  
vertexai.init(project=PROJECT_ID, location=LOCATION)

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

model = GenerativeModel("gemini-1.5-pro-preview-0409")

st.set_page_config(page_title="Gemini Image Demo")
st.header("Gemini Application")

## Display uploaded image ##
# st.image(image, caption="Uploaded Image.", use_column_width=True)

# submit = st.button("Tell me about the image")

# if submit:
#    response = get_gemini_response(input_text, image_base64)
#    st.subheader("The Response is")
#    st.write(response)

# chat_session
if "chat_session" not in st.session_state:    
    st.session_state["chat_session"] = model.start_chat(history=[]) 

# history
for content in st.session_state.chat_session.history:
    with st.chat_message("ai" if content.role == "model" else "user"):
        st.markdown(content.parts[0].text)

# upload_file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# chat
if prompt := st.chat_input("메시지를 입력하세요."):   
    parts = [prompt]

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Convert PIL Image to bytes
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_byte = buffered.getvalue()
        image_base64 = base64.b64encode(image_byte).decode('utf-8')
        image_content = Part.from_data(data=base64.b64decode(image_base64), mime_type="image/png")
        parts.append(image_content)
        
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("ai"):
        response = st.session_state.chat_session.send_message(parts)  
        st.markdown(response.text)
#        response = st.session_state.chat_session.send_message([prompt], 
#                                                              safety_settings=safety_settings, 
#                                                              generation_config=generation_config,
#                                                              stream = True)  
