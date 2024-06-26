from datasets import load_dataset
import pandas as pd

# print(process.env.API_KEY)

dataset = load_dataset("AIatMongoDB/embedded_movies")

dataset_df = pd.DataFrame(dataset['train'])

print(dataset_df.head(5))

# Remove data point where plot column is missing

dataset_df = dataset_df.dropna(subset=['plot'])

print("\nNumber of missing values in each column after removal:")
print(dataset_df.isnull().sum())

# Remove the plot_embedding from each data point in the dataset as we are going to create new embeddings with the new OpenAI embedding Model "text-embedding-3-small"

dataset_df = dataset_df.drop(columns=['plot_embedding'])

print(dataset_df.head(5))

from llama_index.core.settings import Settings
from llama_index.embeddings import openai
from llama_index.embeddings.openai import OpenAIEmbedding

embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=256)

llm = OpenAIEmbedding()

Settings.llm = llm

Settings.embed_model = embed_model

import json
from llama_index.core import Document
from llama_index.core.schema import MetadataMode

# Convert the DataFrame to a JSON string representation
documents_json = dataset_df.to_json(orient='records')

# Load the JSON string into a python list of dictionaries
documents_list = json.loads(documents_json)

llama_documents = []

# count = 0

for document in documents_list:
    # Value for metadata must be one of (str, int, float, None)
    document["writers"] = json.dumps(document["writers"])
    document["languages"] = json.dumps(document["languages"])
    document["genres"] = json.dumps(document["genres"])
    document["cast"] = json.dumps(document["cast"])
    document["directors"] = json.dumps(document["directors"])
    document["countries"] = json.dumps(document["countries"])
    document["imdb"] = json.dumps(document["imdb"])
    document["awards"] = json.dumps(document["awards"])

    # print(document)

    # if count == 96:
    #     print(document)

    if document["fullplot"] == None:
        document["fullplot"] = "Not Given"

    # Create a Document object with the text and excluded metadata for llm and embedding models
    llama_document = Document(
        text=document["fullplot"],
        metadata=document,
        excluded_llm_metadata_keys=["fullplot", "metacritic"],
        excluded_embed_metadata_keys=["fullplot", "metacritic", "poster", "num_mflix_comments", "runtime", "rated"],
        metadata_template="{key}=>{value}",
        text_template="Metadata: {metadata_str} \n-----\n Content: {content}"
        )
    
    print("this is the llama document ") 
    print(llama_document)
    print("\n" + "\n" + "\n" + "\n" + "\n" + "\n" + "\n")

    llama_documents.append(llama_document)

    print("document appended")

    # count += 1
    # print(count)

    # print(llama_documents)

# Observing an example of what the LLM and Embedding model receive as input
print(
    "\nThe LLM sees this: \n",
    llama_documents[0].get_content(metadata_mode=MetadataMode.LLM),
)
print(
    "\nThe Embedding model sees this: \n",
    llama_documents[0].get_content(metadata_mode=MetadataMode.EMBED),
)

from llama_index.core.node_parser import SentenceSplitter

parser = SentenceSplitter()
nodes = parser.get_nodes_from_documents(llama_documents)

for node in nodes:
    node_embedding = embed_model.get_text_embedding(node.get_content(metadata_mode = "all"))
    node.embedding = node_embedding

print("parser part done")

import pymongo
from google.colab import userdata

def get_mongo_client(mongo_uri):
    """Establish connection to MongoDB"""
    try:
        client = pymongo.MongoClient(mongo_uri)
        print("Connection to MongoDB successful")
        return client
    except pymongo.errors.ConnectionFailure as e:
        print(f"Connection failed: {e}")
        return None
    
mongo_uri = userdata.get('MONGO_URI_2')
if not mongo_uri:
    print("MONGO_URI not set in environment variables")

mongo_client = get_mongo_client(mongo_uri)

DB_NAME="movies"
COLLECTION_NAME="movies_records"

db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]

collection.delete_many({})