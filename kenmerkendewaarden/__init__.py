# -*- coding: utf-8 -*-
"""
.. include:: ../README.md
"""

__author__ = """Jelmer Veenstra"""
__email__ = "jelmer.veenstra@deltares.nl"
__version__ = "0.1.0"

from kenmerkendewaarden.slotgemiddelden import *
from kenmerkendewaarden.havengetallen import *
from kenmerkendewaarden.gemgetij import *
from kenmerkendewaarden.overschrijding import *

import warnings
warnings.filterwarnings(action='always', category=DeprecationWarning)
