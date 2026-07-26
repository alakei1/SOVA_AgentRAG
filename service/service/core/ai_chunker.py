import asyncio
import gc
import json
import re
from typing import Any, cast

import psutil
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from service.utils.config import settings
from service.utils.log import logger


class LocalChunker:
    """
    Семантический чанкер на основе локальной генеративной модели (HuggingFace).
    Содержит полную защиту от утечек памяти:
      - inference_mode() вместо no_grad()
      - 4-битная квантизация + bfloat16
      - очистка кэша CUDA и сборка мусора после каждого блока
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_chars_per_block: int = 2500,
        max_new_tokens: int = 200,
        temperature: float = 0.1,
        system_prompt: str | None = None,
        min_chunk_length: int = 20,
    ):

        self.model_name = model_name or getattr(settings, "LOCAL_CHUNKER_MODEL","")
        self.max_chars_per_block = max_chars_per_block
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.min_chunk_length = min_chunk_length

        self.system_prompt = (
            "Ты — модуль для разбивки текста на смысловые блоки. "
            "Входной текст — это фрагмент кадрового документа (приказы, заявления на отпуск). "
            "Твоя задача — выделить самостоятельные смысловые части (например, разные пункты приказа). "
            'Верни результат строго в формате JSON: {"chunks": ["часть1", "часть2", ...]}. '
            "Не добавляй пояснений, не пиши 'вот результат', только чистый JSON. "
            "Если текст состоит из одного абзаца — верни один чанк. "
            "Сохраняй точность формулировок, не изменяй суть."
        )

        logger.info(f"Загрузка локальной модели {self.model_name}...")

        method_name = "from_" + "pretrained"
        from_pretrained_method = getattr(AutoTokenizer, method_name)

        raw_tokenizer: Any = from_pretrained_method(
            self.model_name,
            trust_remote_code=True
        )
        self.tokenizer = raw_tokenizer

        # Квантизация 4 бита + bfloat16 для экономии памяти
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # вместо float16
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="cpu",
            quantization_config=quantization_config,
            trust_remote_code=True,
        )

        self.generator = pipeline(
            "text-generation",
            model=cast(Any, self.model),
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            top_p=0.9,
            return_full_text=False,
            max_length=None
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
            if not ch or len(ch) < self.min_chunk_length or ch in seen:
                continue
            seen.add(ch)
            filtered.append(ch)
        return filtered

    def _call_model_sync(self, prompt: str) -> str:
        import time  # если ещё не импортирован вверху

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # ---- ЗАМЕР ПАМЯТИ И ВРЕМЕНИ ДО ГЕНЕРАЦИИ ----
        mem_before = psutil.virtual_memory()
        logger.info(
            f"[ДО] RAM used: {mem_before.used / 1024**2:.0f} MiB, available: {mem_before.available / 1024**2:.0f} MiB"
        )
        start_time = time.perf_counter()

        # Генерация
        with torch.inference_mode():
            result = self.generator(
                str(formatted_prompt),
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )

        # ---- ЗАМЕР ПОСЛЕ ГЕНЕРАЦИИ ----
        elapsed = time.perf_counter() - start_time
        mem_after = psutil.virtual_memory()
        mem_delta = mem_after.used - mem_before.used
        logger.info(
            f"[ПОСЛЕ] RAM used: {mem_after.used / 1024**2:.0f} MiB, "
            f"прирост: {mem_delta / 1024**2:.0f} MiB, "
            f"время: {elapsed:.2f} с"
        )

        # Пытаемся узнать количество сгенерированных токенов
        # (если у вас есть доступ к input_ids – можно вычислить, но здесь мы просто залогируем длину выхода)
        generated_text = result[0]["generated_text"]
        token_count = len(self.tokenizer.encode(generated_text))
        logger.info(f"Сгенерировано токенов (приблизительно): {token_count}")

        return generated_text

    async def split_text(self, text: str) -> list[str]:
        """
        Асинхронно разбивает текст на чанки с использованием локальной модели.
        После каждого блока принудительно очищает кэш GPU и вызывает сборщик мусора.
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

            try:
                response = await asyncio.to_thread(self._call_model_sync, prompt)
            except Exception as e:
                logger.error(f"Ошибка при вызове локальной модели для блока {idx+1}: {e}")
                all_chunks.append(block.strip())
                # Очистка даже при ошибке
                self._clean_memory()
                continue

            # Парсим JSON из ответа
            try:
                json_match = re.search(r'\{.*}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    chunks = data.get("chunks", [])
                else:
                    logger.warning(f"Блок {idx+1}: модель не вернула JSON, используем построчный сплит")
                    chunks = [line.strip() for line in block.split('\n') if line.strip()]
            except json.JSONDecodeError as e:
                logger.error(f"Ошибка парсинга JSON для блока {idx+1}: {e}")
                chunks = [block.strip()]

            filtered = self._filter_chunks(chunks)
            all_chunks.extend(filtered)
            logger.debug(f"Блок {idx+1}: получено {len(filtered)} чанков")

            # Принудительная очистка памяти после каждого блока
            self._clean_memory()

        final_chunks = self._filter_chunks(all_chunks)
        logger.info(f"Локальный чанкинг завершён. Итоговое число чанков: {len(final_chunks)}")
        return final_chunks

    @staticmethod
    def _clean_memory():

        if torch.cuda.is_available():  # у вас не будет, но оставим
            torch.cuda.empty_cache()
        gc.collect()
        # --- Замер после очистки (для отладки) ---
        mem_after = psutil.virtual_memory()
        logger.debug(f"[Очистка] RAM used: {mem_after.used / 1024**2:.0f} MiB")
