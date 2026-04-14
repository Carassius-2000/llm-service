"""
Модуль предоставляет для логирования.

Attributes
----------
logger : logging.Logger
    Экземпляр логгера для сообщений.
"""

import logging

console_logger = logging.getLogger(__name__)
console_logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - INFO: %(message)s"))
console_logger.addHandler(console_handler)
