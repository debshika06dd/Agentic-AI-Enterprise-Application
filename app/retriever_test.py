from rag.retriever import get_retriever


retriever = get_retriever()


question = "How many casual leaves do employees receive?"


results = retriever.invoke(question)


print("\nRetrieved Documents:\n")

for i, document in enumerate(results):

    print(f"--- Result {i + 1} ---")

    print("Source:", document.metadata.get("source"))

    print("Content:")
    print(document.page_content)

    print()