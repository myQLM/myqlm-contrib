# -*- coding: utf-8 -*-

"""
@authors   Quentin Delamea <quentin.delamea@atos.net>
@intern
@copyright 2020 Bull S.A.S.  -  All rights reserved.
@file qat/plugins/hook_sabre.py
@brief A simple hook to import plugins of qat.sabre
@namespace qat.plugins

    Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.
"""

__qlm_importables__ = {
    "Sabre": "qat.sabre:Sabre",
}
