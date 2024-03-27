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