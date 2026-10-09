from langchain_core.embeddings import Embeddings


def get_embeddings() -> Embeddings:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as huggingface_error:
        raise RuntimeError(
            "Failed to load local Hugging Face embeddings. Install the required "
            "packages with `uv pip install langchain-huggingface "
            "sentence-transformers`, then try again."
        ) from huggingface_error