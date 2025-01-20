# Software Name : DistFactAssessLM
# SPDX-FileCopyrightText: Copyright (c) 2025 Orange SA
# SPDX-License-Identifier: GPL-2.0-or-later

# This software is distributed under the GNU General Public License v2.0 or later,
# see the "LICENSE.txt" file for more details or GNU General Public License v2.0 or later

# Authors: Hichem Ammar Khodja
# Software description: A factual knowledge assessment method for large language models using distractors

import re
from kb.wikidata import TempWikidata, WikidataPrepStage
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--date',
    type=str,
    required=True,
    help="The date of the Wikidata dump to download and push to MongoDB",
    # default="20210104"
)

args = parser.parse_args()

date = args.date.strip()
assert re.match(r'^[0-9]{8}$',date), "The date must be in format YYYYMMDD"

wd = TempWikidata(date, WikidataPrepStage.PREPROCESSED)
wd.build(confirm=False)