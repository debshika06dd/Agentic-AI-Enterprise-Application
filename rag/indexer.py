from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv


load_dotenv()


DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"


def create_vector_store():

    print("Loading documents...")

    documents = []

    for file_path in DATA_DIR.glob("*.txt"):

        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )

        documents.extend(loader.load())

    print("Documents loaded:", len(documents))


    print("Splitting documents...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    print("Chunks created:", len(chunks))


    print("Creating embeddings...")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )


    print("Creating vector database...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="enterprise_knowledge",
        persist_directory=CHROMA_DIR
    )


    print("Vector database created successfully.")

    return vector_store


if __name__ == "__main__":

    create_vector_store()