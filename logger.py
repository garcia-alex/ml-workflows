import logging
from logging.config import fileConfig

LOGGING_NOTSET = logging.NOTSET
LOGGING_INFO = logging.INFO
LOGGING_DEBUG = logging.DEBUG
LOGGING_WARNING = logging.WARNING
LOGGING_ERROR = logging.ERROR
LOGGING_DEBUG = logging.DEBUG

fileConfig('logging.ini')

logger = logging.getLogger('ml')