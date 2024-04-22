import base64
import time
import typing
from dataclasses import dataclass
from google.cloud import aiplatform, storage
from google.protobuf import struct_pb2
import tempfile
import json
from subprocess import call
from absl import app
from absl import flags

# _IMAGE_FILE = flags.DEFINE_string('image_file', None, 'Image filename')
# _TEXT = flags.DEFINE_string('text', None, 'Text to input')
# _PROJECT = flags.DEFINE_string('project', None, 'Project id')



class SingletonInstance:
    __instance = None
    
    @classmethod
    def getInstance(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = cls(*args, **kwargs)
        return cls.__instance


class EmbeddingResponse(typing.NamedTuple):
    text_embedding: typing.Sequence[float]
    image_embedding: typing.Sequence[float]


class StorageClient(SingletonInstance):
    def __init__(self, project: str, bucket: str = "smishing-image"):
        self.client = storage.Client(project = project)

    def get_bucket(self, bucket:str):
        bucket = self.client.bucket(bucket = bucket)
        return bucket
        
    
    

class VectorSearchClient(SingletonInstance):
    def __init__(self, index_endpoint: str, deployed_index_id: str):
        self.index_endpoint = index_endpoint
        self.deployed_index_id = deployed_index_id
        self.index = aiplatform.MatchingEngineIndexEndpoint(index_endpoint)

    def find_neighbors(self, query, num_neighbors: int = 3):
        response = self.index.find_neighbors(
                        deployed_index_id = self.deployed_index_id,
                        queries = [query],
                        num_neighbors = num_neighbors
                )
        return response
    
            
class EmbeddingPredictionClient(SingletonInstance):
    def __init__(self, project: str,
                 location: str = "us-central1",
                 api_regional_endpoint: str = "us-central1-aiplatform.googleapis.com"):
        
        client_options = {"api_endpoint": api_regional_endpoint}
        self.client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
        self.location = location
        self.project = project

    def get_embedding(self, text: str = None, image_bytes: bytes = None, video_uri: str = None,
                      start_offset_sec: int = 0, end_offset_sec: int = 120, interval_sec: int = 16):
        if not text and not image_bytes and not video_uri:
            raise ValueError('At least one of text or image_bytes or video_uri must be specified.')

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
         
        return EmbeddingResponse(
            text_embedding=text_embedding,
            image_embedding=image_embedding)


class EmbeddingClient:
    def __init__(self, project: str = "primeval-argon-420311", location: str = "us-central1"):
        self.storage_client = storage.Client()
        self.location = location
        self.project = project
        self.client = EmbeddingPredictionClient(project = self.project)
        

    def list_blobs(self, bucket_name):
        """Lists all the blobs in the bucket."""
        blobs = storage_client.list_blobs(bucket_name)
        return [blob.name for blob in blobs]

    def download_images(self, bucket_name):
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        image_paths = [ obj_nm for obj_nm in list_blobs(bucket_name)]

        return [bucket.blob(image).download_as_bytes() for image in image_paths]
    
    def make_embedding(self, bucket_name):
        images_byte = self.download_images(bucket_name)       
        #client to access multimodal-embeddings model to convert text to embeddings
        images_embedding = [self.client.get_embedding(image_bytes=image) for image in images_byte]

        return images_embedding
    
    def upload_embedding(self, embedding_url: str = "gs://embedding-image/", 
                         images_path, images_embedding):
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