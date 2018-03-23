# -*- coding:utf-8 -*-
"""
Created on the 21/11/17
@author: Nicolas Thiebaut
@email: nicolas@visage.jobs
"""

import os
import pytest
from zeugma.conf import MODELS_DIR


def test_models_dir_existence():
    assert os.path.exists(MODELS_DIR)

