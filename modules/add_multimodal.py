import base64
import io
import os

from PIL import Image
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
from vertexai import generative_models


PROJECT_ID = "primeval-argon-420311"
LOCATION = "us-central1"
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


def get_gemini():

    model = GenerativeModel("gemini-1.5-pro-preview-0409")

    return model

def transform_file(image_file):
    image = Image.open(image_file)
    
    # Convert PIL Image to bytes
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_byte = buffered.getvalue()
    image_base64 = base64.b64encode(image_byte).decode('utf-8')
    image_content = Part.from_data(data=base64.b64decode(image_base64), mime_type="image/png")

    return image_content    