import logging
import os
import sys


class FolderFileFilter(logging.Filter):
    def filter(self, record):
        path_parts = record.pathname.split(os.sep)

        if len(path_parts) >= 2:
            record.folder_file = f"{path_parts[-2]}/{path_parts[-1]}"
        else:
            record.folder_file = record.pathname

        return True


# Используем новую переменную %(folder_file)s вместо %(pathname)s
log_format = "[%(asctime)s] [%(levelname)s] [%(folder_file)s:%(lineno)d] -> %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger("sova_logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(fmt=log_format, datefmt=date_format))

# ВАЖНО: подключаем наш фильтр к хэндлеру
console_handler.addFilter(FolderFileFilter())
logger.addHandler(console_handler)
