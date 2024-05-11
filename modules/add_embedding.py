import base64
import time
import typing
import json
from subprocess import call
from google.cloud import aiplatform, storage
import numpy as np
import tempfile

import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel
from vertexai.language_models import TextEmbeddingModel
from vertexai.language_models import TextGenerationModel
import streamlit as st

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
_TEXT_INDEX_ID = "5709952999040745472"
_TEXT_ENDPOINT_ID = "1011572687786475520"

_TEXT_DEPLOYED_INDEX_ID="text_embedding_endpoint_1715386956665"
_TEXT_INDEX_ENDPOINT="projects/854115243710/locations/us-central1/indexEndpoints/1011572687786475520"


# Set variables for the current deployed image index.
_IMAGE_DEPLOYED_INDEX_ID = "image_embedding_endpoint_1714090648818"
_IMAGE_INDEX_ENDPOINT = "projects/854115243710/locations/us-east1/indexEndpoints/3300514004257996800"

_IMAGE_INDEX_ID = "6930485672662794240"
_IMAGE_ENDPOINT_ID = "3300514004257996800"

vertexai.init(project=_PROJECT)
aiplatform.init(project=_PROJECT)


class SingletonInstance:
    '''Create only one client'''
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
    '''Connect to Google cloud storage'''
    def __init__(self, project: str= _PROJECT):
        self.client = storage.Client(project = project)

    def get_bucket(self, bucket:str):
        bucket = self.client.bucket(bucket)
        return bucket
    
    def list_blobs(self, bucket:str):
        """Lists all the blobs in the bucket."""
        blobs = self.client.list_blobs(bucket)
        return [blob.name for blob in blobs]
        

class VectorSearchClient(SingletonInstance):
    '''
    Connect to google vertex ai vector search 

    
    Below settings are for deployed index.

    - index_endpoint: when you create endpoint, you can get index endpoint. ex) projects/854115243710/locations/us-central1/indexEndpoints/1011572687786475520
    - deployed_index_id: when you deploy index to endpoint, you can set unique index id. ex) image_embedding_endpoint_1714090648818 
    - bucket_uri: where you store vectors. ex) gs://{BUCKET_NAME}

    - location: where you create vector search. ex) us-central1
    - index_id: when you create index, you can get index id.(it can be replaced by index name.) ex) 5709952999040745472
    - endpoint_id: when you create endpoint, you can get endpoint id. ex) 1011572687786475520
    
    
    '''
    def __init__(self, index_endpoint_name: str , deployed_index_id: str, bucket_uri: str, index_id: str, endpoint_id: str, location:str):
        
        self.index_endpoint_name = index_endpoint_name
        self.deployed_index_id = deployed_index_id

        self.location = location
        self.bucket_uri = bucket_uri

        self.index_id = index_id
        self.endpoint_id = endpoint_id
        
        # endpoint that you want to query
        self.index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name = self.index_endpoint_name,
            location = self.location)
        
        
        # index that you want to insert or update
        self.index = aiplatform.MatchingEngineIndex(
            index_name = self.index_id,
            location = self.location
        )
        
    
    def find_neighbors(self, query: typing.List[float], num_neighbors: int = 3):
        # you can query against the deployed index to find nearest neighbors.
        response = self.index_endpoint.find_neighbors(
                        deployed_index_id = self.deployed_index_id,
                        queries = [query],
                        num_neighbors = num_neighbors
        )
        return response

    def update_index(self):
        # update index from bukcet 
        self.index.update_embeddings(contents_delta_uri=self.bucket_uri, is_complete_overwrite=True)
        return
    
    def upsert_datapoint(self, datapoint: typing.Tuple[str, typing.Optional[typing.List[float]]]):
        ''' 
        upsert datapoint to index

        - id : it can be metadata to distinguish datapoint. ex) path
        - vectors : vector list 

        '''
        datapoints =[
            {
                "datapoint_id": id,
                "feature_vector": vector,
            }
               for id, vector in datapoint
          ]
        
        self.index.upsert_datapoints(datapoints=datapoints)
        return


class TextEmbeddingPredictionClient(SingletonInstance):
    '''

    Load the text embedding model from vertex ai and obtain the vector value.
    project and location must be preset by vertexai.init(project=_PROJECT)

    model : A embedding model provided by vertex AI.

    '''

    def __init__(self, model: str = "textembedding-gecko@001"):    
        self.model = model
        self.client = TextEmbeddingModel.from_pretrained(self.model)
        
    def generate_text_embedding(self, texts: typing.List[str]) -> typing.Optional[typing.List[float]]:
        embedding = self.client.get_embeddings(texts)
        text_embedding = [e.values for e in embedding]
        return text_embedding

class TextEmbeddingClient(SingletonInstance):
    def __init__(self, project: str = _PROJECT, location: str = _TEXT_LOCATION):
        self.location = location
        self.project = project
        self.storage_client = StorageClient(project = self.project)
        self.client = TextEmbeddingPredictionClient()
        
    def __download_texts(self, bucket_name):
        """download texts from bucket"""
        bucket = self.storage_client.get_bucket(bucket_name)
        text_paths = [ obj_nm for obj_nm in self.storage_client.list_blobs(bucket_name)]
        texts = [(text_path,bucket.blob(text_path).download_as_string().decode("utf-8")) for text_path in text_paths]

        return texts


    def __make_text_embeddings(self, bucket_name) -> typing.List[typing.Tuple[str, typing.Optional[typing.List[float]]]]:
        """
            make embeddings from text bucket
        """      
        # client to access text-embeddings model to convert image to embeddings
        texts = self.__download_texts(bucket_name)
        text_embeddings = self.client.generate_text_embedding([e[1] for e in texts])

        return [(i[0],v) for i,v in zip(texts,text_embeddings)]
    
    def upload_text_embeddings(self, bucket_name, embedding_url: str = "gs://embedding-text/"):
        """ 
            upload embeddings to embedding bucket
        """
        text_embeddings = self.__make_text_embeddings(bucket_name)
        
        # Create temporary file to write embeddings to
        embeddings_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)

        with open(embeddings_file.name, "a") as f:

            # Append to file
            embeddings_formatted = [
                    json.dumps(
                        {
                            "id": path,
                            "embedding": [str(value) for value in embedding],
                        }
                    )
                    + "\n"
                for path, embedding in text_embeddings
            ]
            print(embeddings_formatted)
            f.writelines(embeddings_formatted)
            
        call(["gsutil", "cp", f"{embeddings_file.name}", f"{embedding_url}"])

class ImageEmbeddingPredictionClient(SingletonInstance):
    def __init__(self, model:str = "multimodalembedding") :
        self.model = model
        self.client = MultiModalEmbeddingModel.from_pretrained(self.model)       

    # get image embeddings
    def generate_image_embedding(self, text: str = None, image_bytes: bytes = None) -> ImageEmbeddingResponse:
        if not text and not image_bytes:
            raise ValueError('At least one of text or image_bytes must be specified.')
        
        encoded_content =   None

        if image_bytes:
            encoded_content = base64.b64encode(image_bytes)
            
            embeddings = self.client.get_embeddings(
                image=encoded_content,
                contextual_text=text,
            )  
                
            return ImageEmbeddingResponse(
                text_embedding=embeddings.text_embedding,
                image_embedding=embeddings.image_embedding)
        
        return None


class ImageEmbeddingClient(SingletonInstance):
    def __init__(self, project: str = _PROJECT, location: str = _IMAGE_LOCATION):
        self.location = location
        self.project = project

        self.storage_client = StorageClient(project = self.project)
        self.client = ImageEmbeddingPredictionClient(project = self.project)
        
    def __download_images(self, bucket_name):
        """download images from bucket"""

        bucket = self.storage_client.get_bucket(bucket_name)
        image_paths = [obj_nm for obj_nm in self.storage_client.list_blobs(bucket_name)]
        return [(image_path, bucket.blob(image_path).download_as_bytes()) for image_path in image_paths]

    def __make_image_embeddings(self, bucket_name):
        """make embeddings from bucket"""

        images_bytes = self.__download_images(bucket_name)       
        #client to access multimodal-embeddings model to convert image to embeddings
        images_embeddings = self.client.generate_image_embedding([e[1] for e in images_bytes])

        return [(i[0],v) for i, v in zip(images_bytes, images_embeddings)]
        
    def upload_image_embeddings(self, bucket_name, embedding_url: str = "gs://embedding-image/"):
        """upload embeddings to bucket"""
        images_embeddings = self.__make_image_embeddings(bucket_name)
        
        # Create temporary file to write embeddings to
        embeddings_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)

        with open(embeddings_file.name, "a") as f:
            # Append to file
            embeddings_formatted = [
                json.dumps(
                    {
                        "id": path,
                        "embedding": [str(value) for value in embedding.image_embedding],
                    }
                )
                + "\n"
                for path, embedding in images_embeddings
            ]
            f.writelines(embeddings_formatted)
            
        call(["gsutil", "cp", f"{embeddings_file.name}", f"{embedding_url}"])


def show_upload():
    st.session_state["uploader_visible"] = ~st.session_state["uploader_visible"]

def direct_llm(state:bool):
    st.session_state["direct_llm"] = state

def get_emb_result_text(text):
    text_embdding_client = TextEmbeddingPredictionClient().getInstance()
    vector = text_embdding_client.generate_text_embedding([text])
    vector_search_client = VectorSearchClient.getInstance(index_endpoint_name= _TEXT_INDEX_ENDPOINT , deployed_index_id= _TEXT_DEPLOYED_INDEX_ID, bucket_uri= _TEXT_BUCKET_URI, index_id= _TEXT_INDEX_ID, endpoint_id= _TEXT_ENDPOINT_ID, location=_TEXT_LOCATION)
    
    # run query 
    response = vector_search_client.find_neighbors(
            query = vector[0],
            num_neighbors = 3
    )

    if response:
        path = response[0][0].id
        #client to access GCS bucket
        storage_client = StorageClient(project = _PROJECT)
        bucket = storage_client.get_bucket(bucket = "financial-product-text")
        result = bucket.blob(path).download_as_string().decode("utf-8")

                        
        embedding_prompt = f"""You are a banker assisting senario. Please answer by referring to the context. \n
            context: {result} \n account_name : {text}"""
        

        model = TextGenerationModel.from_pretrained("text-bison@002")
        response = model.predict(embedding_prompt)
        return response.text
        






def get_emb_result_image(file: bytes):
    image_embdding_client = ImageEmbeddingPredictionClient().getInstance()
    vector = image_embdding_client.generate_image_embedding(image_bytes = file)
    vector_search_client = VectorSearchClient.getInstance(index_endpoint= _IMAGE_INDEX_ENDPOINT, deployed_index_id=_IMAGE_DEPLOYED_INDEX_ID, bucket_uri=_IMAGE_BUCKET_URI, index_id=_IMAGE_INDEX_ID, endpoint_id=_IMAGE_ENDPOINT_ID, location = _IMAGE_LOCATION)

    # run query 
    response = vector_search_client.find_neighbors(
            query = vector.image_embedding,
            num_neighbors = 1
    )
    if response:
        path = response[0][0].id   
        #client to access GCS bucket
        storage_client = StorageClient(project = _PROJECT)
        bucket = storage_client.get_bucket(bucket = "financial-product-image")
        bytes_data = bucket.blob(path).download_as_bytes()
        result = [bytes_data]
        return result

    
def main():
    pass

if __name__ == "__main__":
    main()