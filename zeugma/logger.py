# -*- coding:utf-8 -*-
"""
Created on the 01/25/2018
@author: Nicolas Thiebaut
@email: nicolas@visage.jobs
"""
from logging import Formatter, getLogger, StreamHandler, WARNING

LOG_FORMAT = (
    "%(asctime)s [%(levelname)s]: %(message)s in %(pathname)s:%(lineno)d")

DEFAULT_LOG_LEVEL = WARNING


DEFAULT_HANDLER = StreamHandler()
DEFAULT_HANDLER.setFormatter(Formatter(LOG_FORMAT))

package_logger = getLogger("zeugma")
package_logger.setLevel(DEFAULT_LOG_LEVEL)
package_logger.addHandler(DEFAULT_HANDLER)
