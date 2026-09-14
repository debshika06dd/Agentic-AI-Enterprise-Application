from pathlib import Path

from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_chroma import Chroma

from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate


# Load environment variables
load_dotenv()


# -----------------------------
# 1. Create embedding model
# -----------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# -----------------------------
# 2. Create LLM
# -----------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# -----------------------------
# 3. Load documents
# -----------------------------

data_dir = Path("data")

documents = []

for file_path in data_dir.glob("*.txt"):

    loader = TextLoader(
        str(file_path),
        encoding="utf-8"
    )

    documents.extend(loader.load())


print("Documents loaded:", len(documents))


# -----------------------------
# 4. Split documents
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Chunks created:", len(chunks))


# -----------------------------
# 5. Create vector database
# -----------------------------

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="enterprise_knowledge"
)


# -----------------------------
# 6. Create retriever
# -----------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# -----------------------------
# 7. Create RAG prompt
# -----------------------------

prompt = ChatPromptTemplate.from_template(
    """
    You are an enterprise knowledge assistant.

    Answer the user's question using only the provided context.

    If the answer cannot be found in the context,
    say that you do not have enough information
    in the knowledge base.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)


# -----------------------------
# 8. Ask question
# -----------------------------

question = "How many casual leaves do employees get?"


# -----------------------------
# 9. Retrieve relevant chunks
# -----------------------------

retrieved_documents = retriever.invoke(question)


# -----------------------------
# 10. Build context
# -----------------------------

context = "\n\n".join(
    document.page_content
    for document in retrieved_documents
)


# -----------------------------
# 11. Create prompt
# -----------------------------

messages = prompt.invoke({
    "context": context,
    "question": question
})


# -----------------------------
# 12. Generate answer
# -----------------------------

response = llm.invoke(messages)


# -----------------------------
# 13. Print answer
# -----------------------------

print("\nFinal Answer:\n")

print(response.content)


# -----------------------------
# 14. Show sources
# -----------------------------

print("\nSources:\n")

for document in retrieved_documents:

    print(document.metadata.get("source"))