# Software Name : DistFactAssessLM
# SPDX-FileCopyrightText: Copyright (c) 2025 Orange SA
# SPDX-License-Identifier: GPL-2.0-or-later

# This software is distributed under the GNU General Public License v2.0 or later,
# see the "LICENSE.txt" file for more details or GNU General Public License v2.0 or later

# Authors: Hichem Ammar Khodja
# Software description: A factual knowledge assessment method for large language models using distractors

set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
conda create --name wfd_build python=3.10 -y
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate wfd_build
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install -c conda-forge mlconjug3
pip install -U pip setuptools wheel
pip install -U 'spacy[cuda12x]' # Change cuda version 
python -m spacy download en_core_web_sm
pip install -r $SCRIPT_DIR/wfd_requirements.txt

# Install transformers-cfg for grammar contrained decoding
pip install git+https://github.com/epfl-dlab/transformers-CFG.git@main
# echo "$SCRIPT_DIR/.." | tee $(python -c "import site;print(site.getsitepackages()[0])")/workspace.pth > /dev/null

# install marius
# cd /root
# git clone https://github.com/marius-team/marius.git
# cd marius
# pip3 install . --no-build-isolation