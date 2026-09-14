from pathlib import Path

from langchain_community.document_loaders import TextLoader


# Location of the data folder
DATA_DIR = Path("data")


# Store all loaded documents
documents = []


# Find all text files
for file_path in DATA_DIR.glob("*.txt"):

    # Load the file
    loader = TextLoader(
        str(file_path),
        encoding="utf-8"
    )

    # Convert file into LangChain Documents
    loaded_documents = loader.load()

    # Add them to our document list
    documents.extend(loaded_documents)


# Display loaded documents
print("Number of documents:", len(documents))

for document in documents:
    print("\nSource:", document.metadata.get("source"))
    print(document.page_content[:200])