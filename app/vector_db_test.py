from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

#Load environment variables
load_dotenv()

#create embedding model 
embeddings = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-001"
)

#Create documents 
documents = [
    "Employees receive 12 casual leaves every year.", 
    "Employees can work from home according to the company hybrid work policy.", 
    "The company provides health insurance to eligible employees.", 
    "Employees must complete mandatory security training",

]

#create vector database 
vector_store = Chroma.from_texts(
    texts = documents, 
    embedding = embeddings, 
    collection_name = "company_knowledge"
)

#search the vector database 
query = "How many casual leaves do employees get?"

results = vector_store.similarity_search(
    query, 
    k = 2
)

#Print results
for result in results:
    print(result.page_content)