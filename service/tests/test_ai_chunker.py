import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from service.core.ai_chunker import LocalChunker


@pytest.fixture
def mock_transformers():
    with (
        patch("service.core.ai_chunker.AutoModelForCausalLM") as mock_model_cls,
        patch("service.core.ai_chunker.AutoTokenizer") as mock_tokenizer_cls,
        patch("service.core.ai_chunker.pipeline") as mock_pipeline,
        patch("service.core.ai_chunker.logger"),
    ):
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted_prompt"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.config.hidden_size = 4096
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_generator = MagicMock()
        mock_generator.return_value = [{"generated_text": '{"chunks": ["test"]}'}]
        mock_pipeline.return_value = mock_generator

        yield {
            "tokenizer": mock_tokenizer,
            "model": mock_model,
            "generator": mock_generator,
            "pipeline": mock_pipeline,
            "model_cls": mock_model_cls,
            "tokenizer_cls": mock_tokenizer_cls,
        }


@pytest.fixture
def chunker(mock_transformers):
    return LocalChunker(device=-1)


# ---------- Тесты инициализации ----------
def test_init_defaults(mock_transformers):
    chunker = LocalChunker()
    assert chunker.model_name == "mistralai/Mistral-7B-Instruct-v0.2"
    assert chunker.max_chars_per_block == 2000
    assert chunker.max_new_tokens == 512
    assert chunker.temperature == 0.1
    assert chunker.min_chunk_length == 20
    assert "лингвистический модуль RAG-системы" in chunker.system_prompt
    mock_transformers["model_cls"].from_pretrained.assert_called_once()
    mock_transformers["tokenizer_cls"].from_pretrained.assert_called_once()


def test_init_custom(mock_transformers):
    chunker = LocalChunker(
        model_name="custom/model",
        max_chars_per_block=3000,
        max_new_tokens=1024,
        temperature=0.7,
        min_chunk_length=10,
        system_prompt="Custom system prompt",
        device=0,
    )
    assert chunker.model_name == "custom/model"
    assert chunker.max_chars_per_block == 3000
    assert chunker.max_new_tokens == 1024
    assert chunker.temperature == 0.7
    assert chunker.min_chunk_length == 10
    assert chunker.system_prompt == "Custom system prompt"


# ---------- Тесты _rough_split ----------
def test_rough_split_small_text(chunker):
    text = "Короткий текст."
    assert chunker._rough_split(text) == [text]


def test_rough_split_by_paragraphs(chunker):
    chunker.max_chars_per_block = 15
    text = "Параграф 1.\n\nПараграф 2."
    blocks = chunker._rough_split(text)
    assert blocks == ["Параграф 1.", "Параграф 2."]


def test_rough_split_combine_paragraphs(chunker):
    chunker.max_chars_per_block = 50
    text = "A.\n\nB.\n\nC."
    blocks = chunker._rough_split(text)
    assert blocks == ["A.\n\nB.\n\nC."]


def test_rough_split_long_paragraph_no_split_inside(chunker):
    chunker.max_chars_per_block = 10
    text = "Оченьдлинныйабзацбезпробелов"
    blocks = chunker._rough_split(text)
    assert blocks == [text]


def test_rough_split_mixed(chunker):
    chunker.max_chars_per_block = 30
    text = "P1 long.\n\nP2 short.\n\nP3 short."
    blocks = chunker._rough_split(text)
    # Все три абзаца помещаются в один блок
    assert len(blocks) == 1
    assert "P1" in blocks[0] and "P2" in blocks[0] and "P3" in blocks[0]


# ---------- Тесты _filter_chunks ----------
def test_filter_chunks_removes_empty_and_short(chunker):
    chunker.min_chunk_length = 0
    chunks = ["", "   ", "ok", "a", "long enough", "dup", "dup"]
    result = chunker._filter_chunks(chunks)
    assert result == ["ok", "a", "long enough", "dup"]


def test_filter_chunks_strips_whitespace(chunker):
    chunker.min_chunk_length = 0
    chunks = ["  hello  ", "world  "]
    assert chunker._filter_chunks(chunks) == ["hello", "world"]


def test_filter_chunks_removes_duplicates_globally(chunker):
    chunker.min_chunk_length = 0
    chunks = ["first", "second", "first", "third"]
    assert chunker._filter_chunks(chunks) == ["first", "second", "third"]


# ---------- Тесты _call_model_sync ----------
def test_call_model_sync_uses_chat_template_and_generator(chunker, mock_transformers):
    chunker.generator = mock_transformers["generator"]
    chunker.tokenizer = mock_transformers["tokenizer"]

    result = chunker._call_model_sync("user prompt")
    assert result == '{"chunks": ["test"]}'
    mock_transformers["tokenizer"].apply_chat_template.assert_called_once_with(
        [
            {"role": "system", "content": chunker.system_prompt},
            {"role": "user", "content": "user prompt"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    mock_transformers["generator"].assert_called_once_with(
        "formatted_prompt",
        max_new_tokens=chunker.max_new_tokens,
        temperature=chunker.temperature,
    )


# ---------- Тесты split_text (асинхронные) ----------
@pytest.mark.asyncio
async def test_split_text_success(chunker, monkeypatch):
    async def mock_to_thread(func, *args, **kwargs):
        return '{"chunks": ["первый", "второй"]}'

    monkeypatch.setattr(asyncio, "to_thread", mock_to_thread)
    chunker.min_chunk_length = 0  # отключаем фильтр по длине
    result = await chunker.split_text("Тестовый текст.")
    assert result == ["первый", "второй"]


@pytest.mark.asyncio
async def test_split_text_multiple_blocks(chunker, monkeypatch):
    monkeypatch.setattr(chunker, "_rough_split", lambda self: ["block1", "block2"])
    call_count = 0

    async def mock_to_thread(func, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return '{"chunks": ["a", "b"]}'
        else:
            return '{"chunks": ["b", "c"]}'

    monkeypatch.setattr(asyncio, "to_thread", mock_to_thread)
    chunker.min_chunk_length = 0
    result = await chunker.split_text("some text")
    assert result == ["a", "b", "c"]
    assert call_count == 2


@pytest.mark.asyncio
async def test_split_text_model_error_fallback(chunker, monkeypatch):
    monkeypatch.setattr(chunker, "_rough_split", lambda self: ["block1", "block2"])

    async def mock_to_thread(func, *args, **kwargs):
        raise RuntimeError("Model crashed")

    monkeypatch.setattr(asyncio, "to_thread", mock_to_thread)
    chunker.min_chunk_length = 0
    result = await chunker.split_text("text")
    assert result == ["block1", "block2"]


@pytest.mark.asyncio
async def test_split_text_no_json_fallback(chunker, monkeypatch):
    monkeypatch.setattr(chunker, "_rough_split", lambda self: ["line1\nline2\nline3"])

    async def mock_to_thread(func, *args, **kwargs):
        return "Это просто текст, не JSON"

    monkeypatch.setattr(asyncio, "to_thread", mock_to_thread)
    chunker.min_chunk_length = 0
    result = await chunker.split_text("text")
    assert result == ["line1", "line2", "line3"]


@pytest.mark.asyncio
async def test_split_text_json_parse_error(chunker, monkeypatch):
    monkeypatch.setattr(chunker, "_rough_split", lambda self: ["block"])

    async def mock_to_thread(func, *args, **kwargs):
        return '{ "chunks": [ "x" '  # невалидный JSON

    monkeypatch.setattr(asyncio, "to_thread", mock_to_thread)
    chunker.min_chunk_length = 0
    result = await chunker.split_text("text")
    assert result == ["block"]


@pytest.mark.asyncio
async def test_split_text_empty_input(chunker):
    assert await chunker.split_text("") == []
    assert await chunker.split_text("   ") == []


@pytest.mark.asyncio
async def test_split_text_final_filtering_applied(chunker, monkeypatch):
    monkeypatch.setattr(chunker, "_rough_split", lambda self: ["block"])

    async def mock_to_thread(func, *args, **kwargs):
        return '{"chunks": ["a", "b", "a", "c"]}'

    monkeypatch.setattr(asyncio, "to_thread", mock_to_thread)
    chunker.min_chunk_length = 1  # оставляем фильтр по минимальной длине
    result = await chunker.split_text("text")
    assert result == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_split_text_uses_asyncio_to_thread(chunker, monkeypatch):
    mock_to_thread = AsyncMock(return_value='{"chunks": ["ok"]}')
    monkeypatch.setattr(asyncio, "to_thread", mock_to_thread)
    monkeypatch.setattr(chunker, "_rough_split", lambda self: ["block"])
    chunker.min_chunk_length = 0
    await chunker.split_text("text")
    mock_to_thread.assert_called_once()
    args, kwargs = mock_to_thread.call_args
    assert args[0] == chunker._call_model_sync
    assert args[1] == "Выдели смысловые чанки из следующего фрагмента текста:\n\nblock"
