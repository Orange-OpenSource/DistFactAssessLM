# Software Name : DistFactAssessLM
# SPDX-FileCopyrightText: Copyright (c) 2025 Orange SA
# SPDX-License-Identifier: GPL-2.0-or-later

# This software is distributed under the GNU General Public License v2.0 or later,
# see the "LICENSE.txt" file for more details or GNU General Public License v2.0 or later

# Authors: Hichem Ammar Khodja
# Software description: A factual knowledge assessment method for large language models using distractors

from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
import re
import subprocess
import os.path as osp
from globals import STORAGE_FOLDER
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', 
                    required=True, 
                    type=str, 
                    choices=['compare_retrieval_strategies', 'compare_retrieval_strategies_dist_temp', 'compare_kms'], 
                    help='Name of the experiment to launch')
parser.add_argument('--num_workers', 
                    type=int, default=2,
                    help='Maximum number of jobs run in parallel')
parser.add_argument('--date', 
                    type=str, required=True,
                    help='Date of the Wikidata dump to use.')
args = parser.parse_args()
commands_type = args.experiment
max_workers = args.num_workers
date = args.date.strip()
assert re.match(r'^[0-9]{8}$',date), "The date must be in format YYYYMMDD"


commands = []

if commands_type in ('compare_retrieval_strategies', 'compare_retrieval_strategies_dist_temp'):
    strategies = ('temp_idf', 'idf', 'random', 'AoID', 'pure_random')
    if commands_type == 'compare_retrieval_strategies_dist_temp':
        strategies = ('temp_idf', 'idf', 'random')
        models = ('EleutherAI/pythia-6.9b',)
        path = osp.join(STORAGE_FOLDER, 'facts_balanced_5000_temp_dist.pkl')
    else:
        path = osp.join(STORAGE_FOLDER, 'facts_balanced_5000.pkl')
    for strategy in strategies:
        commands.append(f'python scripts/general_eval_know_measure/compare_retrieval_strategies.py --model "EleutherAI/pythia-6.9b" --strategy {strategy} --facts_path {path} --date {date}')
    for model in models:
        commands.append(f'python scripts/general_eval_know_measure/compare_retrieval_strategies.py --model "{model}" --strategy AoID --facts_path {path} --date {date}')


elif commands_type == 'compare_kms':
    for i in range(45):
        commands.append(f'python scripts/optimize_know_evals/compute_measures.py --iteration {i} --date {date}')


def run_commands_in_batches(commands, batch_size):
    for i in range(0, len(commands), batch_size):
        batch = commands[i:i + batch_size]
        print('Progress : %s/%s' % (i, len(commands)))
        print(batch)
        processes = []
        for cmd in batch:
            # Start the process and store the process object
            process = subprocess.Popen(cmd, shell=True)
            processes.append(process)
        
        # Wait for all processes in the current batch to finish
        for process in processes:
            process.wait()

def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        pass

# Using ThreadPoolExecutor to run commands in parallel
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit all commands to the exec utor
    future_to_command = {executor.submit(run_command, cmd): cmd for cmd in commands}
    
    # Process results as they complete
    for future in as_completed(future_to_command):
        command = future_to_command[future]
        try:
            cmd, stdout, stderr = future.result()
            print(f"Command: {cmd}\nOutput: {stdout}\nError: {stderr}\n")
        except Exception as exc:
            print(f"Command {command} generated an exception: {exc}")