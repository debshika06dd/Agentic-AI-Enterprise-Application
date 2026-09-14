from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()


CHROMA_DIR = "chroma_db"


def get_retriever():

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    vector_store = Chroma(
        collection_name="enterprise_knowledge",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 3
        }
    )

    return retriever