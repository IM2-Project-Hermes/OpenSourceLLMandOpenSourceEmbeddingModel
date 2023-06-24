from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain.text_splitter import CharacterTextSplitter
import uuid


# Read the file
def process_file(file_path):
    """
    Read the file and return the text
    :param: file_path: path to the file
    :return: text of the file
    """
    with open(file_path) as f:
        text = f.read()
    return text


def find_name_from_source(file_path):
    """
    Extract the name of the file from the file path
    :param: file_path: path to the file
    :return: name of the file
    """
    filename = file_path.split("/")[-1]  # Extract the last part of the string after splitting by "/"
    name = filename.split(".")[0]  # Extract the part before the dot (file extension)

    return name


def create_collection_with_sample_data():
    """
    Create a new collection in the database
    :return:
    """
    # Initialize env
    load_dotenv()

    # Get the database
    chroma_client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="db"
    ))

    # all-MiniLM-L6-v2 embedding function
    embedding_function = embedding_functions.DefaultEmbeddingFunction()

    try:
        collection = chroma_client.create_collection(
            name="documents",
            embedding_function=embedding_function
        )
        print('Collection created')
    except:
        print('Collection already exists')
        return

    file_paths = [
        "data/ComplaintManagement.md",
        "data/CustomerSupport.md",
        "data/Onboarding.md",
        "data/OrderDeveloping.md",
        "data/OrderProcessing.md",
        "data/ProjectManagement.md",
    ]

    # Initialize the text splitter
    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=10)
    texts = []

    for file_path in file_paths:

        # Split the text
        text = process_file(file_path)
        split_text = text_splitter.split_text(text)

        sources = []
        ids = []

        # Add the metadata to each chunk
        for x in range(len(split_text)):
            sources.append({"name": find_name_from_source(file_path), "source": file_path})
            ids.append(str(uuid.uuid4()).replace("-", ""))

        collection.add(
            documents=split_text,
            metadatas=sources,
            ids=ids
        )

        print(f"Saved {file_path}")


if __name__ == '__main__':
    create_collection_with_sample_data()
