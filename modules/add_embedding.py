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
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import (
    VectorSearchVectorStore,
    VectorSearchVectorStoreDatastore,
)

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

def show_upload(state:bool):
    st.session_state["uploader_visible"] = state

def direct_llm(state:bool):
    st.session_state["direct_llm"] = state

def get_emb_result(file):
    
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
    return result


def get_emb_result_text(file, text):
    
    bytes_data = file.getvalue()          
    embedding_client = EmbeddingPredictionClient.getInstance(project = _PROJECT)
    vectorsearch_client = VectorSearchClient.getInstance(index_endpoint = _INDEX_ENDPOINT, deployed_index_id = _DEPLOYED_INDEX_ID)
    emb = embedding_client.get_embedding(image_bytes = bytes_data, text=text).image_embedding
    image_embedding = emb.image_embedding
    text_embedding = emb.text_embedding


    ## Image_text search
    mix_embedding = (image_embedding + text_embedding) / 2
    mix_embedding = np.linalg.norm(mix_embedding)
    
    # run query   
    
    # response = vectorsearch_client.find_neighbors(
    #         query = image_embedding,
    #         num_neighbors = 3
    # )
    # #client to access GCS bucket
    # storage_client = StorageClient.getInstance(project = _PROJECT)
    # bucket = storage_client.get_bucket(bucket = "smishing-image")

    result = [bytes_data]
    return result

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
    def __init__(self, project: str, bucket: str = "smishing-image"):
        self.client = storage.Client(project = project)

    def get_bucket(self, bucket:str):
        bucket = self.client.bucket(bucket = bucket)
        return bucket
        

class VectorSearchClient():
    def __init__(self, index_endpoint: str, deployed_index_id: str, bucket_uri: str, index_id: str, endpoint_id: str):
        self.index_endpoint = index_endpoint
        self.deployed_index_id = deployed_index_id
        self.index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint)
        self.bucket_uri = bucket_uri
        self.index_id = index_id
        self.endpoint_id = endpoint_id
        

    # similarity search by vector 
    def find_neighbors(self, query: typing.List[float], num_neighbors: int = 3):
        response = self.index_endpoint.find_neighbors(
                        deployed_index_id = self.deployed_index_id,
                        queries = [query],
                        num_neighbors = num_neighbors
        )
        return response
    
    def update_index(self):
        index = aiplatform.MatchingEngineIndex(self.index_id)
        index.update_embeddings(contents_delta_uri=self.bucket_uri, is_complete_overwrite=True)

# Using Google API
class TextEmbeddingPredictionClient(SingletonInstance):
    def __init__(self, project = "primeval-argon-420311", location: str = "us-central1", model: str = "textembedding-gecko@001"):
        self.location = location
        self.project = project
        self.model = model
        
        vertexai.init(
		project=self.project,
		location=self.location
		)
        
        self.client = TextEmbeddingModel.from_pretrained(self.model)
        
    def generate_text_embeddings(self, texts: list) -> list:
        response = self.client.get_embeddings(texts)
        text_embeddings = [embedding.values for embedding in response]
        return text_embeddings
        

# Using Langchain
class TextEmbeddingClient(SingletonInstance):
    def __init__(self,  index_id: str, endpoint_id:str, project = "primeval-argon-420311", location: str = "us-central1", model: str = "textembedding-gecko@001", bucket: str = "embedding-text"):
        self.location = location
        self.project = project
        self.embedding_model = VertexAIEmbeddings(model_name=model)
        self.vectorstore = VectorSearchVectorStore.from_components(
                            project_id=self.project,
                            region=self.location,
                            gcs_bucket_name=bucket,
                            index_id=index_id,
                            endpoint_id=endpoint_id,
                            embedding=self.embedding_model,
                            stream_update=True,
                        )
    # upload text embedding to vectorstore
    def upload_embedding(self, record_data:typing.List[typing.Dict]):
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
    

    
            
class ImageEmbeddingPredictionClient(SingletonInstance):
    def __init__(self, project: str= "primeval-argon-420311",
                 location: str = "us-central1",
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
    def __init__(self, project: str = "primeval-argon-420311", location: str = "us-central1"):
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
    def make_embeddings(self, bucket_name):
        images_bytes = self.download_images(bucket_name)       
        #client to access multimodal-embeddings model to convert image to embeddings
        images_embeddings = [self.client.generate_image_embedding(image_bytes=image) for image in images_bytes

        return images_embeddings
        
    # upload embeddings to bucket
    def upload_embedding(self, images_path, images_embedding, embedding_url: str = "gs://embedding-image/", 
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
            
        call(["gsutil", "cp", f"{embeddings_file.name}", f"{embedding_url}"])

     



def main(argv):
    pass

if __name__ == "__main__":
    app.run(main)