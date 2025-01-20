# Software Name : DistFactAssessLM
# SPDX-FileCopyrightText: Copyright (c) 2025 Orange SA
# SPDX-License-Identifier: GPL-2.0-or-later

# This software is distributed under the GNU General Public License v2.0 or later,
# see the "LICENSE.txt" file for more details or GNU General Public License v2.0 or later

# Authors: Hichem Ammar Khodja
# Software description: A factual knowledge assessment method for large language models using distractors

from kb.wikidata import Wikidata
import sys

stdout = sys.stdout
sys.stdout = None
list_dates = Wikidata.available_dumps(refresh=True, verbose=False)
sys.stdout = stdout
print('List of Wikidata dump dates for download are:')
for d in list_dates:
    print(d)