# -*- coding : utf-8 -*-
"""
@authors Quentin Delamea <quentin.delamea@atos.net>
@intern
@copyright 2020  Bull S.A.S.  -  All rights reserved.
           This is not Free or Open Source software.
           Please contact Bull SAS for details about its license.
           Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois
@file qat/__init__.py
@brief
@namespace qat
"""

# Try to find other packages in other folders (with separate build directory)
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
