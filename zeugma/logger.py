# -*- coding:utf-8 -*-
"""
Created on the 01/25/2018
@author: Nicolas Thiebaut
@email: nicolas@visage.jobs
"""
import logging

from logging import StreamHandler
from logging import Formatter

LOG_FORMAT = (
    "%(asctime)s [%(levelname)s]: %(message)s in %(pathname)s:%(lineno)d")

DEFAULT_LOG_LEVEL = logging.WARNING


DEFAULT_HANDLER = StreamHandler()
DEFAULT_HANDLER.setFormatter(Formatter(LOG_FORMAT))

PACKAGE_LOGGER = logging.getLogger("zeugma")
PACKAGE_LOGGER.setLevel(DEFAULT_LOG_LEVEL)
PACKAGE_LOGGER.addHandler(DEFAULT_HANDLER)
