#  ____   ___  _   _    _
# / ___| / _ \| | | |  / \
# \___ \| | | | | | | / _ \
#  ___) | |_| |\ V / / ___ \
# |____/ \___/  \_/ /_/   \_\
#
#    developer alakei1

from langchain_text_splitters import RecursiveCharacterTextSplitter

from service.utils.config import settings


class TextChunker:
    """
    Рекурсивный сплиттер текста без использования AI.
    Разбивает текст на чанки фиксированного размера с перекрытием,
    стараясь сохранить границы абзацев/предложений.
    """

    def __init__(
        self,
        chunk_size: int | None= None,
        chunk_overlap: int | None= None,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size or getattr(settings, "CHUNK_SIZE", 500)
        self.chunk_overlap = chunk_overlap or getattr(settings, "CHUNK_OVERLAP", 50)
        self.separators = separators or getattr(settings, "CHUNK_SEPARATORS", ["\n\n", "\n", ". ", " ", ""])

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

    def split_text(self, text: str) -> list[str]:
        """Разбивает один текст на список чанков."""
        if not text:
            return []
        return self.splitter.split_text(text)

    def split_documents(self, documents: list[dict]) -> list[dict]:
        """
        Принимает список документов (каждый с полями 'text' и 'metadata').
        Возвращает список чанков с добавленными метаданными (индекс чанка).
        """
        chunks = []
        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {}).copy()
            parts = self.split_text(text)
            for idx, part in enumerate(parts):
                chunk_meta = metadata.copy()
                chunk_meta["chunk_index"] = idx
                chunk_meta["total_chunks"] = len(parts)
                chunks.append(
                    {
                        "text": part,
                        "metadata": chunk_meta,
                    }
                )
        return chunks
