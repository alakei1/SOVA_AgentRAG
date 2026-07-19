from minio import Minio

from service.utils.config import settings


class DocumentProcessor:
    def __init__(self):
        self.minio = Minio(settings.MINIO_URL,
                           access_key=settings.MINIO_ACCESS_KEY,
                           secret_key=settings.MINIO_SECRET_KEY,
                           secure=settings.MINIO_SECURE
                           )

