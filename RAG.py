from datasets import load_dataset
import pandas as pd

dataset = load_dataset("AIatMongoDB/embedded_movies")

dataset_df = pd.DataFrame(dataset['train'])

dataset_df.head(5)

# Remove data point where plot column is missing

dataset_df = dataset_df.dropna(subset=['plot'])

print("\nNumber of missing values in each column after removal:")
print(dataset_df.isnull().sum())

# Remove the plot_embedding from each data point in the dataset as we are going to create new embeddings with the new OpenAI embedding Model "text-embedding-3-small"

dataset_df = dataset_df.drop(columns=['plot_embedding'])

dataset_df.head(5)

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

    # Create a Document object with the text and excluded metadata for llm and embedding models
    llama_document = Document(
        text=document["fullplot"],
        metadata=document,
        excluded_llm_metadata_keys=["fullplot", "metacritic"],
        excluded_embed_metadata_keys=["fullplot", "metacritic", "poster", "num_mflix_comments", "runtime", "rated"],
        metadata_template="{key}=>{value}",
        text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
        )
    
    llama_documents.append(llama_document)

# Observing an example of what the LLM and Embedding model receive as input
print(
    "\nThe LLM sees this: \n",
    llama_documents[0].get_content(metadata_mode=MetadataMode.LLM),
)
print(
    "\nThe Embedding model sees this: \n",
    llama_documents[0].get_content(metadata_mode=MetadataMode.EMBED),
)