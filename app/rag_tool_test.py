from tools.knowledge_search import knowledge_search


result = knowledge_search.invoke({
    "query": "How many casual leaves do employees receive?"
})


print("\nRAG Tool Result:\n")
print(result)