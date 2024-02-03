from langchain.text_splitter import RecursiveCharacterTextSplitter


class SplitterFactory:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_splitter(self) -> RecursiveCharacterTextSplitter:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        return splitter
