import asyncio
import json

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from service.utils.config import settings
from service.utils.log import logger


class LocalChunker:
    """
    Семантический чанкер на основе локальной генеративной модели (HuggingFace).
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_chars_per_block: int = 2000,   # меньше, чем для API, т.к. локальная модель имеет меньший контекст
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        system_prompt: str | None = None,
        min_chunk_length: int = 20,
        device: int = -1,  # -1 для CPU, 0 для GPU
    ):
        """
        Args:
            model_name: Имя модели на HuggingFace (например, "mistralai/Mistral-7B-Instruct-v0.2").
                       Если не указано, берётся из settings.LOCAL_CHUNKER_MODEL.
            max_chars_per_block: Максимальная длина блока в символах.
            max_new_tokens: Максимальное число генерируемых токенов на блок.
            temperature: Температура (чем ниже, тем строже).
            system_prompt: Системный промпт (можно переопределить).
            min_chunk_length: Минимальная длина итогового чанка.
            device: -1 для CPU, 0 для GPU.
        """
        self.model_name = model_name or getattr(settings, "LOCAL_CHUNKER_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
        self.max_chars_per_block = max_chars_per_block
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.min_chunk_length = min_chunk_length

        self.system_prompt = system_prompt or (
            "Ты — лингвистический модуль RAG-системы. Разбей входной текст на самостоятельные смысловые чанки. "
            "Каждый чанк должен быть понятен без контекста. Если есть местоимения — замени их на явные сущности. "
            "Верни результат строго в формате JSON: {\"chunks\": [\"чанк1\", \"чанк2\", ...]}. "
            "Не придумывай информацию, которой нет в тексте."
        )

        # Загружаем модель и токенизатор
        logger.info(f"Загрузка локальной модели {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True) # type: ignore
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto" if device == 0 else None,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        # Создаём пайплайн для генерации
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            top_p=0.9,
            return_full_text=False,
        )
        logger.info(f"Модель {self.model_name} загружена, размерность: {self.model.config.hidden_size}")

    def _rough_split(self, text: str) -> list[str]:
        """Разбивает текст на блоки по абзацам, не превышая max_chars_per_block."""
        if len(text) <= self.max_chars_per_block:
            return [text]

        paragraphs = text.split("\n\n")
        blocks = []
        current_block = []
        current_length = 0

        for para in paragraphs:
            para_len = len(para)
            if current_length + para_len > self.max_chars_per_block and current_block:
                blocks.append("\n\n".join(current_block))
                current_block = []
                current_length = 0
            current_block.append(para)
            current_length += para_len

        if current_block:
            blocks.append("\n\n".join(current_block))

        return blocks

    def _filter_chunks(self, chunks: list[str]) -> list[str]:
        """Удаляет пустые, слишком короткие и дублирующиеся чанки."""
        seen = set()
        filtered = []
        for ch in chunks:
            ch = ch.strip()
            if not ch:
                continue
            if len(ch) < self.min_chunk_length:
                continue
            if ch in seen:
                continue
            seen.add(ch)
            filtered.append(ch)
        return filtered

    def _call_model_sync(self, prompt: str) -> str:
        """Синхронный вызов модели – будет обёрнут в asyncio.to_thread."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        # Формируем промпт в стиле чата (зависит от модели, для Mistral используется chat template)
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        result = self.generator(formatted_prompt, max_new_tokens=self.max_new_tokens, temperature=self.temperature)
        # result — список, берём первый элемент
        generated_text = result[0]["generated_text"]
        return generated_text

    async def split_text(self, text: str) -> list[str]:
        """
        Асинхронно разбивает текст на чанки с использованием локальной модели.
        """
        if not text or not text.strip():
            return []

        rough_blocks = self._rough_split(text)
        all_chunks = []

        logger.info(f"Локальный чанкинг: {len(rough_blocks)} блоков для обработки.")

        for idx, block in enumerate(rough_blocks):
            if not block.strip():
                continue

            prompt = f"Выдели смысловые чанки из следующего фрагмента текста:\n\n{block}"

            # Вызов модели в отдельном потоке (т.к. это синхронная операция)
            try:
                response = await asyncio.to_thread(self._call_model_sync, prompt)
            except Exception as e:
                logger.error(f"Ошибка при вызове локальной модели для блока {idx+1}: {e}")
                # Fallback: сохраняем блок целиком
                all_chunks.append(block.strip())
                continue

            # Парсим JSON из ответа
            try:
                # Ищем JSON-объект в строке (модель может добавить пояснения)
                import re
                json_match = re.search(r'\{.*}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    chunks = data.get("chunks", [])
                else:
                    # Если JSON не найден, пробуем разбить по строкам (примитивный fallback)
                    logger.warning(f"Блок {idx+1}: модель не вернула JSON, используем построчный сплит")
                    chunks = [line.strip() for line in block.split('\n') if line.strip()]
            except json.JSONDecodeError as e:
                logger.error(f"Ошибка парсинга JSON для блока {idx+1}: {e}")
                chunks = [block.strip()]

            # Фильтруем и добавляем
            filtered = self._filter_chunks(chunks)
            all_chunks.extend(filtered)
            logger.debug(f"Блок {idx+1}: получено {len(filtered)} чанков")

        # Финальная фильтрация (удаляем дубли между блоками)
        final_chunks = self._filter_chunks(all_chunks)
        logger.info(f"Локальный чанкинг завершён. Итоговое число чанков: {len(final_chunks)}")
        return final_chunks
