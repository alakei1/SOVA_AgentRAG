import asyncio

from service.rag.llm_client import get_llm_client
from service.utils.log import logger


async def test_deepseek():
    client = get_llm_client()

    question = "Какие правила оформления документов?"
    context = """
    Правила оформления документов:
    1. Все документы должны быть подписаны руководителем
    2. Документы должны быть в формате PDF
    3. Срок хранения — 5 лет
    4. Утверждено приказом №45 от 15.01.2024
    """

    answer = await client.generate(question, context)

    logger.info(f"Вопрос: {question}")
    logger.info(f"Ответ: {answer}")


if __name__ == "__main__":
    asyncio.run(test_deepseek())
