import streamlit as st
from langchain import PromptTemplate
from langchain.llms import OpenAI
import streamlit as st
import time
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from embedding.predict_request import *



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

_PROJECT="primeval-argon-420311"
_INDEX_ENDPOINT="projects/854115243710/locations/us-central1/indexEndpoints/7122253694686461952"
_DEPLOYED_INDEX_ID="multimodal_embedding_endpoint"







st.set_page_config(page_title="(working title) Idea's Bank Senior supporter", page_icon=":robot:")
st.header("(working title) Idea's Bank Senior supporter")


if "uploader_visible" not in st.session_state:
    st.session_state["uploader_visible"] = False

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []



openai_api_key = "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"
col1, col2 = st.columns(2)

def load_LLM(openai_api_key):
    """Logic for loading the chain you want to use should go here."""
    # Make sure your openai_api_key is set as an environment variable
    ######## TODO 1. OpenAI -> Gemini
    llm = OpenAI(temperature=.7, openai_api_key=openai_api_key)
    return llm


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
        

def show_upload(state:bool):
    st.session_state["uploader_visible"] = state

# Display image upload on app
with st.chat_message("system"):
    cols= st.columns((3,1,1))
    cols[0].write("Do you want to upload a photo that you want to know is smishing?")
    cols[1].button("yes", use_container_width=True, on_click=show_upload, args=[True])
    cols[2].button("no", use_container_width=True, on_click=show_upload, args=[False]) 
    
## upload image

if st.session_state["uploader_visible"]:
    file = st.file_uploader("Upload your data", type=['png', 'jpg', 'jpeg'])    
    if file:
        with st.spinner("Processing your file"):
            
             # To read file as bytes:
            bytes_data = file.getvalue()          
            embedding_client = EmbeddingPredictionClient.getInstance(project = _PROJECT)
            vectorsearch_client = VectorSearchClient.getInstance(index_endpoint = _INDEX_ENDPOINT, deployed_index_id = _DEPLOYED_INDEX_ID)
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
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

## load llm
llm = load_LLM(openai_api_key=openai_api_key)




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
        if  not len(st.session_state.messages)  : # Initial message
            response = st.write_stream(response_generator())
        else :
            #### TODO 4. Fraud detection + MultiModal
            response = st.write_stream(llm_response_generator(prompt, llm))

    #### TODO 5. TTS

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})






