```tree
service/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── dependencies.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── embedder.py
│   │   ├── vector_store.py
│   │   ├── llm_client.py
│   │   ├── chunker.py
│   │   └── document_processor.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   └── db_models.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── indexing.py
│   │   ├── query.py
│   │   └── tasks.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── logger.py
│       └── validators.py
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
├── data/
│   ├── qdrant_storage/
│   ├── minio_data/
│   ├── pg_data/
│   └── redis_data/
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_indexing.py
│   ├── test_embedder.py
│   └── conftest.py
├── scripts/
│   ├── download_models.py
│   └── seed_test_data.py
├── .venv/
├── pyproject.toml
├── uv.lock
├── .env
└── .gitignore
```