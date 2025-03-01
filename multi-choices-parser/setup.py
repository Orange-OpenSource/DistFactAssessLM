# Software Name : DistFactAssessLM
# SPDX-FileCopyrightText: Copyright (c) 2025 Orange SA
# SPDX-License-Identifier: GPL-2.0-or-later

# This software is distributed under the GNU General Public License v2.0 or later,
# see the "LICENSE.txt" file for more details or GNU General Public License v2.0 or later

# Authors: Hichem Ammar Khodja
# Software description: A factual knowledge assessment method for large language models using distractors

from setuptools import setup, find_packages

setup(
    name='multi-choices-parser',
    version='0.9.57',
    author='Hichem Ammar Khodja',
    author_email='hichem5696@gmail.com',
    packages=find_packages(),
    description='An efficient incremental parser for multi-choices grammars.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/HichemAK/multi-choices-parser',
    install_requires=[],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ]
)
