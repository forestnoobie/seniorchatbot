import base64
import time
import typing
import json
from subprocess import call

from dataclasses import dataclass
from google.cloud import aiplatform, storage
from google.protobuf import struct_pb2
import numpy as np

import vertexai
from vertexai.language_models import TextEmbeddingModel

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import (
    VectorSearchVectorStore,
    VectorSearchVectorStoreDatastore,
)
from langchain.chains import RetrievalQA
from langchain_google_vertexai import VertexAI
from langchain_google_vertexai import VertexAIEmbeddings

from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    Namespace,
    NumericNamespace,
)

import tempfile
from absl import app
from absl import flags
import streamlit as st



# _IMAGE_FILE = flags.DEFINE_string('image_file', None, 'Image filename')
# _TEXT = flags.DEFINE_string('text', None, 'Text to input')
# _PROJECT = flags.DEFINE_string('project', None, 'Project id')

### Initial setting
_PROJECT = "primeval-argon-420311"

_TEXT_LOCATION = "us-central1"
_IMAGE_LOCATION = "us-east1"

_IMAGE_DIMENSIONS = 1408
_TEXT_DIMENSIONS = 768

_TEXT_BUCKET = "embedding-text"
_IMAGE_BUCKET = "embedding-image"

_TEXT_BUCKET_URI = f"gs://{_TEXT_BUCKET}"
_IMAGE_BUCKET_URI = f"gs://{_IMAGE_BUCKET}"

# Set variables for the current deployed text index.
_TEXT_DISPLAY_NAME = "text_embedding"
_TEXT_DEPLOYED_INDEX_ID = "text_embedding_endpoint"
_TEXT_API_ENDPOINT="1754456625.us-central1-854115243710.vdb.vertexai.goog"
_TEXT_INDEX_ENDPOINT="projects/854115243710/locations/us-central1/indexEndpoints/1709912105005613056"
_TEXT_INDEX_ID = "5709952999040745472"
_TEXT_ENDPOINT_ID = "1011572687786475520"


# Set variables for the current deployed text index.
_IMAGE_DISPLAY_NAME = "image_embedding"
_IMAGE_DEPLOYED_INDEX_ID = "image_embedding_endpoint"
_IMAGE_API_ENDPOINT = "1253489155.us-east1-854115243710.vdb.vertexai.goog"
_IMAGE_INDEX_ENDPOINT = "projects/854115243710/locations/us-east1/indexEndpoints/3300514004257996800"
_IMAGE_INDEX_ID = "6930485672662794240"
_IMAGE_ENDPOINT_ID = "3300514004257996800"

vertexai.init(project=_PROJECT)
aiplatform.init(project=_PROJECT)


class SingletonInstance:
    __instance = None
    
    @classmethod
    def getInstance(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = cls(*args, **kwargs)
        return cls.__instance


class ImageEmbeddingResponse(typing.NamedTuple):
    text_embedding: typing.Sequence[float]
    image_embedding: typing.Sequence[float]


class StorageClient():
    def __init__(self, project: str= _PROJECT):
        self.client = storage.Client(project = project)

    def get_bucket(self, bucket:str):
        bucket = self.client.bucket(bucket = bucket)
        return bucket
        

class VectorSearchClient():
    def __init__(self, index_endpoint: str , deployed_index_id: str, bucket_uri: str, index_id: str, endpoint_id: str, location:str):
        self.index_endpoint_name = index_endpoint
        self.deployed_index_id = deployed_index_id
        self.location = location
        self.bucket_uri = bucket_uri
        self.index_id = index_id
        self.endpoint_id = endpoint_id
        
        self.index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name = self.index_endpoint_name,
            location = self.location)
        
        self.index = aiplatform.MatchingEngineIndex(
            index_name = self.index_id,
            location = self.location
        )
        
        

    # similarity search by vector 
    def find_neighbors(self, query: typing.List[float], num_neighbors: int = 3):
        response = self.index_endpoint.find_neighbors(
                        deployed_index_id = self.deployed_index_id,
                        queries = [query],
                        num_neighbors = num_neighbors
        )
        return response

    # update index from bukcet
    def update_index(self):
        self.index.update_embeddings(contents_delta_uri=self.bucket_uri, is_complete_overwrite=True)

    # upsert datapoint
    def upsert_datapoint(self):
        pass


# Using Google API
class TextEmbeddingPredictionClient(SingletonInstance):
    def __init__(self, project = _PROJECT, location: str = _TEXT_LOCATION, model: str = "textembedding-gecko@001"):    
        self.location = location
        self.project = project
        self.model = model
        self.client = TextEmbeddingModel.from_pretrained(self.model)
        
    def generate_text_embeddings(self, texts: list) -> list:
        response = self.client.get_embeddings(texts)
        text_embeddings = [embedding.values for embedding in response]
        return text_embeddings
        

# Using Langchain API
class TextEmbeddingLCHClient(SingletonInstance):
    def __init__(self,  index_id: str, endpoint_id:str, project = _PROJECT, location: str = _TEXT_LOCATION, model: str = "textembedding-gecko@001", bucket: str = _TEXT_BUCKET):
        self.location = location
        self.project = project
        self.index_id = index_id
        self.endpoint_id = endpoint_id
        self.bucket = bucket
        

        
        self.embedding_model = VertexAIEmbeddings(model_name=model)
        self.vectorstore = VectorSearchVectorStore.from_components(
                            project_id=self.project,
                            region=self.location,
                            gcs_bucket_name=self.bucket,
                            index_id=self.index_id,
                            endpoint_id=self.endpoint_id,
                            embedding=self.embedding_model,
                            stream_update=True,
                        )
        
    # upload text embedding to vectorstore
    def upload_text(self, record_data:typing.List[typing.Dict]):
        texts = []
        metadatas = []
        for record in record_data:
            record = record.copy()
            page_content = record.pop("description")
            texts.append(page_content)
            if isinstance(page_content, str):
                metadata = {**record}
                metadatas.append(metadata)
                
        return self.vectorstore.add_texts(texts=texts, metadatas=metadatas, is_complete_overwrite=True)

    # search by text
    def similarity_search(self, query: str, k: int = 3, filters: dict=None):
        # Try running a similarity search with text filter
        # filters = [Namespace(name="season", allow_tokens=["spring"])]
        return self.vectorstore.similarity_search(query, k=k, filter = filters)

    def as_retriever(self):
        # Initialize the vectore_store as retriever
        return self.vectorstore.as_retriever()
    

    
            
class ImageEmbeddingPredictionClient(SingletonInstance):
    def __init__(self, project: str= _PROJECT,
                 location: str = _IMAGE_LOCATION,
                 api_regional_endpoint: str = "us-central1-aiplatform.googleapis.com"):
        
        client_options = {"api_endpoint": api_regional_endpoint}
        self.client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
        self.location = location
        self.project = project
        
    # get image embedding
    def generate_image_embedding(self, text: str = None, image_bytes: bytes = None):
        if not text and not image_bytes:
            raise ValueError('At least one of text or image_bytes must be specified.')

        instance = struct_pb2.Struct()
        
        if text:
            instance.fields['text'].string_value = text

        if image_bytes:
            encoded_content = base64.b64encode(image_bytes).decode("utf-8")
            image_struct = instance.fields['image'].struct_value
            image_struct.fields['bytesBase64Encoded'].string_value = encoded_content
            
        instances = [instance]
        endpoint = (f"projects/{self.project}/locations/{self.location}"
                    "/publishers/google/models/multimodalembedding@001")
        
        response = self.client.predict(endpoint=endpoint, instances=instances)

        text_embedding = None
        if text:
            text_emb_value = response.predictions[0]['textEmbedding']
            text_embedding = [v for v in text_emb_value]

        image_embedding = None
        if image_bytes:
            image_emb_value = response.predictions[0]['imageEmbedding']
            image_embedding = [v for v in image_emb_value]
         
        return ImageEmbeddingResponse(
            text_embedding=text_embedding,
            image_embedding=image_embedding)


class ImageEmbeddingClient(SingletonInstance):
    def __init__(self, project: str = _PROJECT, location: str = _IMAGE_LOCATION):
        self.location = location
        self.project = project
        self.storage_client = storage.Client(project = self.project)
        self.client = ImageEmbeddingPredictionClient(project = self.project)
        

    def list_blobs(self, bucket_name):
        """Lists all the blobs in the bucket."""
        blobs = self.storage_client.list_blobs(bucket_name)
        return [blob.name for blob in blobs]

    # download images from bucket
    def download_images(self, bucket_name):
        bucket = self.storage_client.bucket(bucket_name)
        image_paths = [ obj_nm for obj_nm in self.list_blobs(bucket_name)]

        return [bucket.blob(image).download_as_bytes() for image in image_paths]

    # make embeddings from bucket
    def make_image_embeddings(self, bucket_name):
        images_bytes = self.download_images(bucket_name)       
        #client to access multimodal-embeddings model to convert image to embeddings
        images_embeddings = [self.client.generate_image_embedding(image_bytes=image) for image in images_bytes]

        return images_embeddings
        
    # upload embeddings to bucket
    def upload_image_embeddings(self, images_path: str, images_embedding:str, embedding_url: str = "gs://embedding-image/", 
                        ):
        # Create temporary file to write embeddings to
        embeddings_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)

        with open(embeddings_file.name, "a") as f:
            # Append to file
            embeddings_formatted = [
                json.dumps(
                    {
                        "id": str(path),
                        "embedding": [str(value) for value in embedding.image_embedding],
                    }
                )
                + "\n"
                for path, embedding in zip(images_path, images_embedding)
            ]
            f.writelines(embeddings_formatted)
            
        call(["/home/nakyil/Desktop/llm/jg/google-cloud-sdk/bin/gsutil", "cp", f"{embeddings_file.name}", f"{embedding_url}"])

     



def show_upload(state:bool):
    st.session_state["uploader_visible"] = state

def direct_llm(state:bool):
    st.session_state["direct_llm"] = state

def get_emb_result(text):
    
    embedding_model = VertexAIEmbeddings(model_name="multimodalembedding")
    vector = embedding_model.embed_query(text)
    
    vector_search_client = VectorSearchClient.getInstance(index_endpoint= _IMAGE_INDEX_ENDPOINT, deployed_index_id=_IMAGE_DEPLOYED_INDEX_ID, bucket_uri=_IMAGE_BUCKET_URI, index_id=_IMAGE_INDEX_ID, endpoint_id=_IMAGE_ENDPOINT_ID, location = _IMAGE_LOCATION)

    # run query 
    response = vector_search_client.find_neighbors(
            query = vector,
            num_neighbors = 1
    )
    if response:
        path = response[0][0].id
    
        #client to access GCS bucket
        storage_client = StorageClient.getInstance(project = _PROJECT)
        bucket = storage_client.get_bucket(bucket = "embedding-image")
        bytes_data = bucket.blob(image).download_as_bytes()

    result = [bytes_data]
    return result


def get_emb_result_text(text):
    
    text_embedding_client = TextEmbeddingLCHClient(index_id=_INDEX_ID, endpoint_id = _ENDPOINT_ID)
    
    llm = VertexAI(model_name="gemini-pro")
    retriever = text_embedding_client.as_retriever()
    
    retrieval_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,)
    
    response = retrieval_qa({"query": text})
    #print(f"{response['result']}")
    #print("REFERENCES")
    #print(f"{response['source_documents']}")

    
    return response['result']
    
def main(argv):
    pass

if __name__ == "__main__":
    app.run(main)