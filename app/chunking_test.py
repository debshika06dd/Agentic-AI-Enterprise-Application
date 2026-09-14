from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Location of data folder
DATA_DIR = Path("data")


# Load documents
documents = []

for file_path in DATA_DIR.glob("*.txt"):

    loader = TextLoader(
        str(file_path),
        encoding="utf-8"
    )

    documents.extend(loader.load())


# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)


# Split documents into chunks
chunks = text_splitter.split_documents(documents)


# Display results
print("Original documents:", len(documents))
print("Total chunks:", len(chunks))


for i, chunk in enumerate(chunks):

    print(f"\n--- Chunk {i + 1} ---")

    print("Source:", chunk.metadata.get("source"))

    print(chunk.page_content)