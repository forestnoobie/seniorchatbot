
from langchain_google_genai import ChatGoogleGenerativeAI
import getpass
import vertexai
import langchain

import os

## init
vertexai.init(
		project="primeval-argon-420311",
		location="us-central1",
		staging_bucket="gs://gemini_test_idb"
		
		)

## check api key
if "GOOGLE_API_KEY" not in os.environ:
	os.environ["GOOGLE_API_KEY"] = getpass.getpass("AIzaSyDEZqZ8IpHF9sV5LY2hdwQfZHyc0u3xirk")
	
## test run
llm = ChatGoogleGenerativeAI(model="gemini-pro")
result = llm.invoke("tell me you are alive")
print(result.content)
