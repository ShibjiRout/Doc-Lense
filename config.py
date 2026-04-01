from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Security
    API_SECRET_KEY: str = "my-super-secret-key-123" # This matches the .env file we made
    
    # OpenAI Config
    OPENAI_API_KEY: str
    EMBED_MODEL: str = "text-embedding-3-small"
    CHAT_MODEL: str = "gpt-4o-mini"
    
    # RAG Config
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    TOP_K: int = 5
    CHROMA_PATH: str = "./chroma_data"

    class Config:
        env_file = ".env"

# Create a global settings object to use in other files
settings = Settings()