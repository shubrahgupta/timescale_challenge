class OpenAIAPIKEY:
    OPENAI_API_KEY = "your-openai-key"
 
class VectorStoreConfig:
    DB_CONNECTION_STRING = "connection-string"

#Funstion to get VectorStore configuration
def get_vectorstore_config():
    return VectorStoreConfig()

def get_open_ai_api_key():
    return OpenAIAPIKEY()
