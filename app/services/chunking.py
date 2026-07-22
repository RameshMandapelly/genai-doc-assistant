from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """
    Splits extracted document text into overlapping chunks suitable for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)