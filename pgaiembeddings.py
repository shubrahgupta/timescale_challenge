from typing import List
import psycopg2
from langchain_core.embeddings import Embeddings

class PgAIEmbeddings(Embeddings):
    """Custom embeddings class that uses pgAI's OpenAI embedding function"""
    
    def __init__(self, connection_string: str):
        """Initialize with PostgreSQL connection string"""
        self.connection_string = connection_string
        
    def _parse_embedding(self, embedding_str: str) -> List[float]:
        """Parse the embedding string returned by pgAI into a list of floats"""
        # Remove any surrounding whitespace and brackets
        cleaned = embedding_str.strip('[] ')
        # Split by comma and convert to floats
        return [float(x) for x in cleaned.split(',')]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using pgAI's OpenAI embedding function"""
        embeddings = []
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                for text in texts:
                    cur.execute(
                        "SELECT ai.openai_embed('text-embedding-ada-002', %s) as embedding",
                        (text,)
                    )
                    result = cur.fetchone()
                    # Parse the embedding string into a list of floats
                    embedding = self._parse_embedding(result[0])
                    embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using pgAI's OpenAI embedding function"""
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT ai.openai_embed('text-embedding-ada-002', %s) as embedding",
                    (text,)
                )
                result = cur.fetchone()
                # Parse the embedding string into a list of floats
                return self._parse_embedding(result[0])