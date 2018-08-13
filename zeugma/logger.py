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

package_logger = logging.getLogger("zeugma")
package_logger.setLevel(DEFAULT_LOG_LEVEL)
package_logger.addHandler(DEFAULT_HANDLER)
