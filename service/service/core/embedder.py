import redis

from service.utils.config import settings


class Embedder:
    def __init__(self):
        self.redis_client = redis.Redis.from_url(
            settings.REDIS_URL, decode_responses=True
        )

