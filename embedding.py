from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import CohereEmbeddings
from langchain_chroma import Chroma # Or another vector store
import os
import dotenv

# Load API
dotenv.load_dotenv()
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")

# 1. Load the CSV, each row becomes a document
loader = CSVLoader(file_path="Book-1.csv")
documents = loader.load()

# 2. Split (chunking is handled by treating each row as a document) and Embed
embeddings = CohereEmbeddings(user_agent="hhh")
# This creates the embeddings and stores them in a local vector store
db = Chroma.from_documents(documents, embeddings, persist_directory="chroma_db" )
