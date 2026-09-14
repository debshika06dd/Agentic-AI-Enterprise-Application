from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

#Load environment variables 
load_dotenv()

#create embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-001"
)

#Text that we want to convert into a vector
text = "I love learning Artificial Intelligence."

#Generate embedding 
vector = embeddings.embed_query(text)

#Print the vector
print(vector)

#Print number of dimensions
print("Vector dimensions: ", len(vector))