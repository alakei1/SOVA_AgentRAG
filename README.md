<!-- ======================================================== -->
<!--  ПЕЧАТАЮЩИЙ ЗАГОЛОВОК (индиго, без фона, английский)      -->
<!-- ======================================================== -->
<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Dancing+Script&weight=400&size=44&pause=1000&color=4B0082&center=true&vCenter=true&width=900&height=100&lines=Intelligent+Knowledge+Base" alt="Intelligent Knowledge Base" />
</p>

<p align="center" style="font-size: 1.2rem; color: #4B0082; font-family: 'Dancing Script', cursive; margin-top: -10px;">
  RAG-pipeline for corporate documents
</p>

<!-- ======================================================== -->
<!--  БЕЙДЖИ СТАТУСА (замените ссылки на свои)                 -->
<!-- ======================================================== -->
<p align="center">
  <a href="https://github.com/ваш_username/ваш_repo/stargazers">
    <img src="https://img.shields.io/github/stars/ваш_username/ваш_repo?style=for-the-badge&logo=github&color=yellow" alt="Stars" />
  </a>
  <a href="https://github.com/ваш_username/ваш_repo/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/ваш_username/ваш_repo?style=for-the-badge&logo=opensourceinitiative" alt="License" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.14%2B-blue?style=for-the-badge&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" alt="Status" />
</p>

<hr style="border: 1px solid #e1e4e8; margin: 30px 0;" />

<!-- ======================================================== -->
<!--  ОПИСАНИЕ ПРОЕКТА (красиво оформленное)                   -->
<!-- ======================================================== -->
<h2 align="left" style="border-bottom: 2px solid #4B0082; padding-bottom: 6px;">📋 Описание проекта</h2>

<div style="font-size: 16px; line-height: 1.8; color: #1a1a1a;">

  <p style="text-align: justify; margin-bottom: 20px;">
    <span style="font-size: 1.4rem; font-weight: 700; color: #4B0082;">🚀 Что это?</span><br />
    Проект представляет собой полноценный <strong style="color: #4B0082;">RAG-пайплайн</strong> (Retrieval-Augmented Generation) для интеллектуальной обработки корпоративных документов. Система построена на <strong>асинхронном Python 3.10+</strong> и интегрирует несколько высокопроизводительных сервисов:
  </p>

  <ul style="list-style-type: '⚡ '; padding-left: 20px; margin-bottom: 20px; color: #2c3e50;">
    <li><strong style="color: #4B0082;">MinIO</strong> – S3-совместимое хранилище файлов</li>
    <li><strong style="color: #4B0082;">Qdrant</strong> – векторная база данных с дисковыми индексами (экономия RAM)</li>
    <li><strong style="color: #4B0082;">Redis</strong> – кэш для эмбеддингов с настраиваемым TTL</li>
  </ul>

  <hr style="border: 1px dashed #d0d0d0; margin: 25px 0;" />

  <!-- ====== БЛОК 1: DocumentProcessor ====== -->
  <h3 style="color: #4B0082; margin-top: 0;">📄 DocumentProcessor – работа с файлами</h3>
  <p style="text-align: justify; margin-bottom: 15px;">
    Модуль подключается к MinIO через асинхронный клиент <code>miniopy_async</code> и умеет загружать, проверять существование и удалять файлы. Извлечение текста из <strong>PDF</strong> (PyMuPDF) и <strong>DOCX</strong> (python-docx) выполняется в отдельном потоке, чтобы не блокировать событийный цикл. Размер загружаемого файла ограничен <strong>50 МБ</strong> (настраивается).
  </p>

  <!-- ====== БЛОК 2: Чанкинг ====== -->
  <h3 style="color: #4B0082;">✂️ Чанкинг – разбиение на смысловые блоки</h3>
  <p style="text-align: justify; margin-bottom: 10px;">
    Извлечённый текст передаётся в <code>RAGPipeline</code>, где первым этапом происходит чанкинг. Доступны два режима:
  </p>
  <ul style="list-style-type: '✅ '; padding-left: 20px; margin-bottom: 15px;">
    <li>
      <strong style="color: #4B0082;">Быстрый (по умолчанию)</strong> – рекурсивный сплиттер из LangChain (<code>RecursiveCharacterTextSplitter</code>) с настраиваемыми разделителями (абзацы, предложения, слова) и параметрами <code>chunk_size</code> и <code>chunk_overlap</code>. Работает без GPU.
    </li>
    <li>
      <strong style="color: #4B0082;">Семантический (опционально)</strong> – <code>LocalChunker</code> загружает локальную генеративную модель (например, Qwen/Qwen2-7B-Instruct) в 4‑битной квантизации через <code>BitsAndBytesConfig</code> с типом bfloat16. Использует <code>torch.inference_mode()</code>, автоматически очищает кэш CUDA и собирает мусор. Требует GPU и 4–6 ГБ ОЗУ, но выделяет самостоятельные смысловые части (например, пункты приказа) и возвращает JSON с чанками.
    </li>
  </ul>

  <!-- ====== БЛОК 3: Embedder ====== -->
  <h3 style="color: #4B0082;">🧠 Embedder – векторное представление</h3>
  <p style="text-align: justify; margin-bottom: 15px;">
    Каждый чанк преобразуется в вектор с помощью предобученной модели Sentence‑Transformers (по умолчанию <code>sergeyzh/LaBSE-ru-sts</code>, оптимизированной для русского языка). Эмбеддинги кэшируются в Redis по <strong>MD5-хешу</strong> текста и имени модели – при повторных запросах векторы мгновенно получаются из кэша, что ускоряет и индексацию, и поиск. Если Redis недоступен, кэширование прозрачно отключается.
  </p>

  <!-- ====== БЛОК 4: VectorStore ====== -->
  <h3 style="color: #4B0082;">💾 VectorStore – сохранение в Qdrant</h3>
  <p style="text-align: justify; margin-bottom: 15px;">
    Векторы вместе с метаданными (текст чанка, источник, <code>tenant_id</code>, имя коллекции) сохраняются в Qdrant. При создании коллекции включается <code>on_disk=True</code> для хранения векторов и индексов на диске, а также создаётся индекс по полю <code>tenant_id</code> для быстрой фильтрации. Это позволяет работать с большими объёмами данных без переполнения RAM.
  </p>

  <!-- ====== БЛОК 5: Поиск и генерация ====== -->
  <h3 style="color: #4B0082;">🔍 Поиск и генерация ответа</h3>
  <p style="text-align: justify; margin-bottom: 15px;">
    Поисковый запрос пользователя также эмбеддится, выполняется поиск ближайших чанков в Qdrant с фильтром по <code>tenant_id</code> и порогом схожести (<code>score_threshold</code>). Найденные фрагменты объединяются в контекст и отправляются в <code>LLMClient</code>, который через HTTP-клиент <code>httpx</code> с пулом соединений обращается к <strong>DeepSeek API</strong>. Автоматические повторные попытки (через <code>tenacity</code>) при таймаутах и ошибках HTTP, экспоненциальная задержка, стриминг для длинных ответов. Генерируется финальный ответ на русском языке, строго придерживаясь предоставленного контекста; логируется количество использованных токенов.
  </p>

  <!-- ====== БЛОК 6: Управление документами ====== -->
  <h3 style="color: #4B0082;">🗂️ Управление актуальностью базы знаний</h3>
  <p style="text-align: justify; margin-bottom: 15px;">
    В <code>RAGPipeline</code> реализована функция удаления всех чанков конкретного документа по его имени – это позволяет поддерживать базу знаний в актуальном состоянии.
  </p>

  <!-- ====== БЛОК 7: Переменные окружения ====== -->
  <h3 style="color: #4B0082;">⚙️ Переменные окружения</h3>
  <ul style="list-style-type: '🔹 '; padding-left: 20px; margin-bottom: 15px; column-count: 2; column-gap: 30px;">
    <li><strong>Redis:</strong> <code>REDIS_HOST</code>, <code>REDIS_PORT</code>, <code>REDIS_PASSWORD</code></li>
    <li><strong>Qdrant:</strong> <code>QDRANT_HOST</code>, <code>QDRANT_PORT</code>, <code>QDRANT_GRPC_PORT</code></li>
    <li><strong>MinIO:</strong> <code>MINIO_ENDPOINT</code>, <code>MINIO_ACCESS_KEY</code>, <code>MINIO_SECRET_KEY</code>, <code>MINIO_SECURE</code></li>
    <li><strong>DeepSeek:</strong> <code>DEEPSEEK_API_KEY</code>, <code>DEEPSEEK_MODEL</code></li>
    <li><strong>Чанкинг:</strong> <code>CHUNK_SIZE</code>, <code>CHUNK_OVERLAP</code>, <code>CHUNK_SEPARATORS</code></li>
    <li><strong>Модель эмбеддинга:</strong> <code>EMBEDDER_MODEL</code></li>
  </ul>

  <!-- ====== БЛОК 8: Производственные требования ====== -->
  <h3 style="color: #4B0082;">🏭 Производственная надёжность</h3>
  <p style="text-align: justify; margin-bottom: 15px;">
    Система построена с учётом производственных требований:
  </p>
  <ul style="list-style-type: '✅ '; padding-left: 20px; margin-bottom: 15px;">
    <li>Асинхронность и пулы соединений</li>
    <li>Обработка ошибок с логированием через предварительно настроенный <code>logger</code></li>
    <li>Защита от утечек памяти при работе с локальными LLM (очистка CUDA, сборка мусора)</li>
    <li>Гибкость в выборе режимов чанкинга</li>
    <li>Код модулей (<code>ai_chunker</code>, <code>chunker</code>, <code>document_processor</code>, <code>embedder</code>, <code>llm_client</code>, <code>rag_pipeline</code>, <code>vector_store</code>) написан с аннотациями типов, наследованием и синглтонами для LLM-клиента – это упрощает расширение и поддержку.</li>
  </ul>

  <!-- ====== БЛОК 9: Пример использования ====== -->
  <h3 style="color: #4B0082;">💡 Пример использования</h3>
  <div style="background: #f6f8fa; padding: 16px 20px; border-radius: 8px; border-left: 4px solid #4B0082; margin-bottom: 15px;">
    <p style="margin: 0; font-family: 'Courier New', monospace; font-size: 15px; color: #1e1e1e;">
      <span style="color: #4B0082; font-weight: 700;">>>> </span>Создаётся экземпляр <code>RAGPipeline</code>.<br />
      <span style="color: #4B0082; font-weight: 700;">>>> </span>Вызывается метод <code>run_from_text</code> с текстом, именем источника, коллекцией и <code>tenant_id</code> – после чанкинга, эмбеддинга и вставки в Qdrant возвращается количество созданных чанков.<br />
      <span style="color: #4B0082; font-weight: 700;">>>> </span>Метод <code>query</code> принимает вопрос, коллекцию, <code>tenant_id</code> и <code>top_k</code>, выполняет поиск, формирует контекст и через <code>LLMClient</code> возвращает словарь с полями:
      <br />
      <code>answer, sources, context, total_chunks, average_score, top_k_requested, processing_time_ms</code>.
    </p>
  </div>

  <!-- ====== ЗАКЛЮЧЕНИЕ ====== -->
  <div style="background: linear-gradient(135deg, #f0edff 0%, #ffffff 100%); padding: 18px 22px; border-radius: 12px; border: 1px solid #d4c9f0; margin-top: 20px;">
    <p style="margin: 0; font-size: 17px; color: #2c3e50;">
      🎯 <strong style="color: #4B0082;">Готов к развёртыванию</strong> – как внутри корпоративного контура, так и в облаке. Предоставляет сотрудникам интеллектуального ассистента для работы с внутренними документами (приказы, заявления, инструкции) и обеспечивает точные ответы исключительно на основе проверенных данных.
    </p>
  </div>

</div>

<!-- ======================================================== -->
<!--  КЛЮЧЕВЫЕ ФИЧИ (обновлены под реальный функционал)        -->
<!-- ======================================================== -->
<h2 align="left" style="border-bottom: 2px solid #4B0082; padding-bottom: 6px;">⚡ Ключевые фичи</h2>

<ul style="font-size: 16px; line-height: 1.8; list-style-type: '✅ ';">
  <li><strong>Асинхронный пайплайн</strong> – неблокирующая обработка документов с пулами соединений к сервисам.</li>
  <li><strong>Гибкий чанкинг</strong> – рекурсивный сплиттер LangChain для скорости или семантический чанкинг с локальной LLM (Qwen) для точности.</li>
  <li><strong>Кэширование эмбеддингов</strong> – Redis с TTL для мгновенного повторного использования векторов.</li>
  <li><strong>Векторный поиск с фильтрацией</strong> – Qdrant с дисковыми индексами и индексом по tenant_id для многопользовательских сценариев.</li>
  <li><strong>Интеграция с DeepSeek API</strong> – генерация ответов с ретраями, стримингом и логированием токенов.</li>
  <li><strong>Управление документами</strong> – загрузка/удаление из MinIO и синхронизация чанков в Qdrant.</li>
  <li><strong>Производственная надёжность</strong> – обработка ошибок, защита от утечек памяти (CUDA), настраиваемые переменные окружения.</li>
</ul>

<!-- ======================================================== -->
<!--  СТЕК ТЕХНОЛОГИЙ (реальные иконки)                       -->
<!-- ======================================================== -->
<h2 align="left" style="border-bottom: 2px solid #4B0082; padding-bottom: 6px;">🛠️ Стек технологий</h2>

<p align="left">
  <a href="https://skillicons.dev">
    <img src="https://skillicons.dev/icons?i=python,docker,redis,postgres,github,git,pytorch,linux" alt="Tech Stack" />
  </a>
  <br />
  <span style="font-size: 14px; color: #555;">
    <strong>Дополнительно:</strong> MinIO, Qdrant, LangChain, HuggingFace Transformers, PyMuPDF, python-docx, httpx, tenacity
  </span>
</p>

<!-- ======================================================== -->
<!--  УСТАНОВКА И ЗАПУСК                                       -->
<!-- ======================================================== -->
<h2 align="left" style="border-bottom: 2px solid #4B0082; padding-bottom: 6px;">📦 Установка и запуск</h2>

<details style="background: #f6f8fa; padding: 12px 18px; border-radius: 8px; margin: 12px 0;">
  <summary style="font-weight: 600; cursor: pointer; font-size: 16px;">
    📂 Нажмите, чтобы открыть инструкцию по установке
  </summary>
  <div style="margin-top: 12px;">
    <pre style="background: #1e1e1e; color: #d4d4d4; padding: 16px; border-radius: 6px; overflow-x: auto; font-size: 14px;">
      <code>
# Клонируйте репозиторий
git clone https://github.com/ваш_username/ваш_repo.git

# Перейдите в папку проекта
cd ваш_repo

# Создайте виртуальное окружение
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Установите зависимости
pip install -r requirements.txt

# Настройте переменные окружения (создайте .env файл)
# Пример содержимого .env:
# REDIS_HOST=localhost
# REDIS_PORT=6379
# QDRANT_HOST=localhost
# QDRANT_PORT=6333
# MINIO_ENDPOINT=localhost:9000
# MINIO_ACCESS_KEY=minioadmin
# MINIO_SECRET_KEY=minioadmin
# DEEPSEEK_API_KEY=ваш_ключ
# DEEPSEEK_MODEL=deepseek-chat
# CHUNK_SIZE=1024
# CHUNK_OVERLAP=200

# Запустите проект
python main.py
      </code>
    </pre>
  </div>
</details>

<!-- ======================================================== -->
<!--  КОНТАКТЫ                                                 -->
<!-- ======================================================== -->
<h2 align="left" style="border-bottom: 2px solid #4B0082; padding-bottom: 6px;">🤝 Контакты</h2>

<p align="left">
  <a href="https://t.me/ваш_телеграм">
    <img src="https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white" alt="Telegram" />
  </a>
  <a href="mailto:ваш_email@example.com">
    <img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email" />
  </a>
</p>

<hr style="border: 1px solid #e1e4e8; margin: 30px 0;" />

<p align="center" style="color: #888; font-size: 14px;">
  ✨ Сделано с ❤️ и <a href="https://github.com" style="color: #4B0082;">GitHub</a>
</p>
