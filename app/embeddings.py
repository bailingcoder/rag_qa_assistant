from app.config import load_config,get_api_key
from langchain_openai import OpenAIEmbeddings

_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        config=load_config()
        emb_config=config["embedding"]
        _embeddings=OpenAIEmbeddings(
            model=emb_config["model"],
            base_url=emb_config["base_url"],
            api_key=get_api_key(emb_config["api_key_env"]),
            check_embedding_ctx_length=False,
        )
    return _embeddings