# Software Name : DistFactAssessLM
# SPDX-FileCopyrightText: Copyright (c) 2025 Orange SA
# SPDX-License-Identifier: GPL-2.0-or-later

# This software is distributed under the GNU General Public License v2.0 or later,
# see the "LICENSE.txt" file for more details or GNU General Public License v2.0 or later

# Authors: Hichem Ammar Khodja
# Software description: A factual knowledge assessment method for large language models using distractors

from collections import Counter
from contextlib import contextmanager
import datetime
from functools import wraps
import hashlib
from itertools import groupby
import json
import math
import os
import pickle
import random
import re
import sys
import time
from typing import ContextManager, Iterable, Union
import bson
import os.path as osp

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pymongo import MongoClient
from scipy import stats
import torch
from scipy.stats import pearsonr
import hashlib
import seaborn as sns

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util import Retry
import zipfile
import os
from io import BytesIO


class PrintableException(Exception):
    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return self.__class__.__name__ + ": " + self.args[0]


def all_equal(iterable : Iterable) -> bool:
    """Checks if all elements of iterable are equal.

    Args:
        iterable (Iterable)

    Returns:
        bool
    """
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

def str2number(number_string : str) -> Union[int, float]:
    """Convert a string representing a number to either an integer or a float (starting with integer conversion).

    Args:
        number_string (str): The number as a string

    Raises:
        ValueError: when number_string is not a number

    Returns:
        Union[int, float]: Number
    """
    try:
        return int(number_string)
    except ValueError:
        pass

    try:
        return float(number_string)
    except ValueError:
        raise ValueError('"%s" is neither an integer nor a float!' % number_string)


def get_print_verbose(verbose = False):
    def nothing(*args, **kwargs) -> None:
        pass
    if verbose:
        return print
    return nothing

DATE_PAT = re.compile(r"^[0-9]{8}$")
def date_is_correct(date : str) -> bool:
    """Check if a date in this format YYYYMMDD is correct and exists

    Args:
        date (str): Text representing a date

    Returns:
        bool
    """
    m = re.match(DATE_PAT, date)
    if not m:
        return False
    
    try:
        datetime.datetime.strptime(date, "%Y%m%d")
        return True
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYYMMDD. Date found : %s" % date)
    





def mongodb_dump(collections, conn, db_name, path):
    """
    MongoDB Dump

    :param collections: Database collections name
    :param conn: MongoDB client connection
    :param db_name: Database name
    :param path:
    :return:
    
    >>> DB_BACKUP_DIR = '/path/backups/'
    >>> conn = MongoClient("mongodb://admin:admin@127.0.0.1:27017", authSource="admin")
    >>> db_name = 'my_db'
    >>> collections = ['collection_name', 'collection_name1', 'collection_name2']
    >>> dump(collections, conn, db_name, DB_BACKUP_DIR)
    """

    db = conn[db_name]
    for coll in collections:
        with open(os.path.join(path, f'{coll}.bson'), 'wb+') as f:
            for doc in db[coll].find():
                f.write(bson.BSON.encode(doc))

def mongodb_exists(path, coll_name):
    return os.path.exists(os.path.join(path, coll_name + '.bson'))


def mongodb_restore(path, conn, db_name, coll_name):
    """
    MongoDB Restore

    :param path: Database dumped path
    :param conn: MongoDB client connection
    :param db_name: Database name
    :param coll: Collection name
    :return:
    
    >>> DB_BACKUP_DIR = '/path/backups/'
    >>> conn = MongoClient("mongodb://admin:admin@127.0.0.1:27017", authSource="admin")
    >>> db_name = 'my_db'
    >>> restore(DB_BACKUP_DIR, conn, db_name)
    
    """
    found = False
    db = conn[db_name]
    for coll in os.listdir(path):
        if coll.endswith('.bson') and coll.startswith(coll_name):
            with open(os.path.join(path, coll), 'rb') as f:
                db[coll.split('.')[0]].insert_many(bson.decode_all(f.read()))
            found = True
    if not found:
        raise FileNotFoundError


def dump_json(filepath : str, obj):
    if filepath.endswith('.json'):
        data = json.dumps(obj)
    elif filepath.endswith('.jsonl'):
        data = ''.join(json.dumps(x) + '\n' for x in obj)
    with open(filepath, 'w') as f:
        f.write(data)

def load_json(filepath : str) -> Union[dict, list]:
    with open(filepath) as f:
        if filepath.endswith('.json'):
            content = json.load(f)
        elif filepath.endswith('.jsonl'):
            content = [json.loads(x) for x in f]
    return content

def read_list(path : str) -> list[str]:
    with open(path, 'r') as f:
        l = [x.strip() for x in f]
    return l

def save_list(l : list[str], path : str):
    with open(path, 'w') as f:
        for x in l:
            f.write(x + '\n')

def inf_none_gen():
    """Infinite generatore of "None" values
    """
    while True:
        yield None

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_class(path : str) -> type:
    """Get class specified in the path , e.g. "os.path"
    
    The module is imported if needed

    Args:
        path (str): Path to the module just like you were doing a regular import

    Raises:
        NameError: If the module or the class inside the module could not be found

    Returns:
        type: The class
    """
    module_name, class_name = osp.splitext(path)
    class_name = class_name.lstrip('.')
    module = sys.modules.get(module_name)
    if module is None:
        try:
            module = __import__(module_name)
        except ImportError:
            raise NameError("The module %s could not be imported" % module_name)
    try:
        cls = getattr(module, class_name)
    except AttributeError:
        raise NameError("The class %s could not be imported" % path)
    return cls

MONGODB_CLIENTS = {}

def get_mongodb_client(mongodb_url : str = None) -> MongoClient:
    """Get MongoDB client given the MongoDB URL.
    This functions caches the mongodb clients to avoid creating a new connection each time a Wikidata object is instanciated for example.
    This is the recommended way of instantiating a MongoDB client.

    Args:
        mongodb_url (str): The URL that points to a Mongo database

    Returns:
        MongoClient
    """

    mongodb_url = os.getenv('MONGO_URL', "mongodb://127.0.0.1")
    client = MONGODB_CLIENTS.get(mongodb_url, None)
    if client is not None:
        return client

    client = MongoClient(mongodb_url)
    MONGODB_CLIENTS[mongodb_url] = client
    return client

def create_switch_contextmanager(cls : type, var_name : str) -> ContextManager:
    @contextmanager
    def manager():
        """Enables caching of credibility results by LMs on text
        """
        # Set the class variable to True to indicate we're inside the context
        cp = getattr(cls, var_name)
        setattr(cls, var_name, True)
        try:
            yield
        finally:
            # Recover the class variable when exiting the context
            setattr(cls, var_name, cp)
    return manager

def is_subseq(subseq : list, seq : list) -> bool:
    for i in range(len(seq)- len(subseq)+1):
        for j in range(len(subseq)):
            if subseq[j] != seq[i+j]:
                break
        else:
            return True
    return False

def concat(l : list[list | np.ndarray | torch.Tensor]):
    if len(l) == 0:
        return []
    first = l[0]
    if isinstance(first, np.ndarray):
        return np.concatenate(l, axis=0)
    elif isinstance(first, torch.Tensor):
        # NOTE: EXPERIMENTAL
        return torch.stack(l, axis=0)
    elif isinstance(first, list):
        return [y for x in l for y in x]
    else:
        return l

def pearson_r(x,y) -> tuple[float, float]:
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    x_isnan = np.isnan(x) | np.isinf(x)
    y_isnan = np.isnan(y) | np.isinf(y)
    mask = ~x_isnan & ~y_isnan
    return pearsonr(x[mask], y[mask])



def plot_mean_with_ci(data, x, y, n_bins=10, ci=0.95, groupby=None, y_limit=None, ax=None):
    """
    Plots mean of y_col per bin of x_col with confidence interval,
    optionally grouping by an additional column.

    Parameters:
    - data: pd.DataFrame
    - x: str
    - y: str
    - n_bins: int, optional
    - ci: float, optional
    - groupby: str, optional
    - ax: specify custom axes here
    """
    if ax is None:
        ax = plt
    set_xticks = lambda ax : plt.xticks if ax is plt else ax.set_xticks
    # Create bins for x_col
    data['bin'] = pd.cut(data[x], bins=n_bins, precision=1)
    min_y, max_y = float('inf'), -float('inf')
    if groupby:
        # Compute the mean and standard error of y_col per bin and group
        summary = data.groupby(['bin', groupby])[y].agg(['mean', 'sem', 'count']).reset_index()
        
        # Getting unique groups and bins for plotting and legend
        groups = summary[groupby].unique()
        bins = summary['bin'].unique()

        # Colors
        colors = sns.color_palette("deep", n_colors=len(bins))

        
        # Plotting
        for idx, group in enumerate(groups):
            group_data = summary[summary[groupby] == group]
            degrees_freedom = group_data['count'] - 1
            conf_int = group_data['sem'] * stats.t.ppf((1 + ci) / 2, degrees_freedom)
            min_y, max_y = min(min_y, -conf_int.min()+summary['mean'].min())-0.1, max(max_y, conf_int.max()+summary['mean'].max())+0.1
            ax.bar(x=np.arange(len(bins)) + idx*0.1, 
                    height=group_data['mean'], 
                    yerr=conf_int, 
                    alpha=0.7, 
                    capsize=5, 
                    color=colors[idx],
                    width=0.1,
                    label=group)
        
        set_xticks(ax)(ticks=np.arange(len(bins)), labels=bins, rotation=45)
        ax.legend(title=groupby)
    else:
        # Compute the mean and standard error of y_col per bin
        summary = data.groupby('bin')[y].agg(['mean', 'sem', 'count'])
        
        # Compute the confidence interval
        degrees_freedom = summary['count'] - 1
        conf_int = summary['sem'] * stats.t.ppf((1 + ci) / 2, degrees_freedom)
        min_y, max_y = min(min_y, -conf_int.min()+summary['mean'].min())-0.05, max(max_y, conf_int.max()+summary['mean'].max())+0.05
        
        # Plotting
        ax.bar(x=np.arange(len(summary)), height=summary['mean'], yerr=conf_int, alpha=0.7, capsize=5)
        set_xticks(ax)(ticks=np.arange(len(summary)), labels=summary.index, rotation=45)
    
    ylabel = lambda ax : plt.ylabel if ax is plt else ax.set_ylabel
    xlabel = lambda ax : plt.xlabel if ax is plt else ax.set_xlabel
    title = lambda ax : plt.title if ax is plt else ax.set_title
    ylim = lambda ax : plt.ylabel if ax is plt else ax.set_ylim

    ylabel(ax)(f'Mean of {y}')
    xlabel(ax)(f'Bins of {x}')
    title(ax)(f'Mean of {y} per Bin of {x} with {int(ci*100)}% CI')
    if ax is None:
        plt.tight_layout()
    ax.grid(axis='y')
    if y_limit:
        ylim(ax)(y_limit)
    else:
        ylim(ax)((min_y, max_y))

def uniquifier(seq : Iterable) -> list:
    """Remove duplicates while keeping order (after first one is filtered)

    Source : https://stackoverflow.com/questions/480214/how-do-i-remove-duplicates-from-a-list-while-preserving-order

    Args:
        seq (Iterable): Iterable

    Returns:
        list: Iterable without duplicates
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def hash_dict(d : dict):
    return commutative_hash(pickle.dumps(x) for x in d.items())

def commutative_hash(*args):
    # Initialize an integer to store the combined hash
    combined_hash = 0
    
    # Hash each argument and combine the result using XOR
    for arg in args:
        # Create a new hash object for each argument
        hash_obj = hashlib.sha256()
        hash_obj.update(arg)
        
        # Convert the hash digest to an integer and XOR it with the combined hash
        current_hash = int(hash_obj.hexdigest(), 16)
        combined_hash ^= current_hash
    
    # Convert the combined integer hash back to hexadecimal
    combined_hash_hex = format(combined_hash, 'x').zfill(64)
    return combined_hash_hex


def confidence_interval_prec(prec : int = 1):
    def f(frame : pd.DataFrame):
        m = frame.mean(axis=0, skipna=True)
        s = frame.std(axis=0, skipna=True)
        c = len(frame) - frame.isna().sum(axis=0)
        d = 1.96*s/math.sqrt(c)
        return f'%.{prec}f ± %.{prec}f' % (m,d)
    return f

class DownloadError(Exception):
    """Custom exception for download errors."""
    pass

# Function to download and unzip a file from a URL with retry logic
def download_and_unzip(url, extract_to, retries=3, backoff_factor=0.3):
    def is_zip_file(response, url):
        """
        Determine if the response is a zip file either by the Content-Disposition header
        or by the file extension in the URL.
        """
        content_disposition = response.headers.get('Content-Disposition', '')
        if 'zip' in content_disposition:
            return True
        if url.lower().endswith('.zip'):
            return True
        return False

    # Create a session object
    session = requests.Session()
    # Define the retry parameters
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
        # method_whitelist=["HEAD", "GET", "OPTIONS"]
    )
    # Mount it for both http and https usage
    session.mount('http://', HTTPAdapter(max_retries=retry_strategy))
    session.mount('https://', HTTPAdapter(max_retries=retry_strategy))

    try:
        # Send a HTTP request to the URL
        print(f"Downloading file from {url}")
        with session.get(url, stream=True) as response:
            # Raise an exception if the download failed
            response.raise_for_status()
            
            # Check if the response content is a zip file
            if is_zip_file(response, url):
                # Create a BytesIO object to hold the chunks of data
                zip_file_bytes = BytesIO()
                total_size = int(response.headers.get('content-length', 0))
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as progress_bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        # Filter out keep-alive new chunks
                        if chunk:
                            zip_file_bytes.write(chunk)
                            progress_bar.update(len(chunk))
                
                # Extract the zip file
                zip_file_bytes.seek(0)  # Move to the beginning of the BytesIO object
                with zipfile.ZipFile(zip_file_bytes, 'r') as zip_ref:
                    # Extract the zip file to the specified directory
                    print(f"Extracting files to '{extract_to}' folder")
                    zip_ref.extractall(extract_to)
                    print("Extraction complete.")
            else:
                raise DownloadError("The URL does not contain a zip file.")
    except requests.exceptions.HTTPError as http_err:
        raise DownloadError(f"HTTP error occurred: {http_err}") from http_err
    except Exception as err:
        raise DownloadError(f"An error occurred: {err}") from err
    finally:
        # Close the session
        session.close()

class TimeItContextManager:
    def __init__(self, name : str) -> None:
        self.name = name

    def __enter__(self):
        print(f'{self.name}', end=' ')
        self.time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.execution_time = time.time() - self.time
        print(f': {self.execution_time} sec')


def sample_from_list_of_collections(list_of_lists, k):
    """Memory-and-compute-efficient sampling from a list of collections, 
    i.e., without creating a big intermediate collection to sample from.
    This sampling is with replacement and is equivalent (if I'm not wrong) to uniform random sampling with replacement
    from the concatenation of all collections.
    """
    # Step 1: Calculate the total number of elements
    total_elements = sum(len(sublist) for sublist in list_of_lists)
    
    if k > total_elements:
        raise ValueError("k cannot be greater than the total number of elements")
    
    # Step 2: Randomly select k unique positions
    sampled_positions = random.sample(range(total_elements), k)
    sampled_positions.sort()  # Sorting helps in efficiently locating elements
    
    # Step 3: Locate the elements
    sampled_elements = []
    current_position = 0
    pos_index = 0
    
    for sublist in list_of_lists:
        sublist_length = len(sublist)
        
        while pos_index < k and sampled_positions[pos_index] < current_position + sublist_length:
            element_index = sampled_positions[pos_index] - current_position
            sampled_elements.append(sublist[element_index])
            pos_index += 1
        
        current_position += sublist_length
        
        if pos_index >= k:
            break
    
    return sampled_elements


def singleton(cls):
    """Singleton class decorator"""
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

def singleton_by_args(cls):
    """Singleton by arguments decorator. 
    It ensures the same instance is returned for the same __init__ arguments"""
    instances = {}
    
    @wraps(cls)
    def get_instance(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in instances:
            instances[key] = cls(*args, **kwargs)
        return instances[key]
    
    return get_instance

def topk_with_indices(arr : np.ndarray, k : int) -> tuple[np.ndarray]:
    """Find the top-k values in array and returns these values + the indices where these values appear"""
    # Avoid index issues
    k = min(arr.shape[0], k)
    # Get the indices of the top k elements
    indices = np.argpartition(arr, -k)[-k:]
    # Get the top k elements
    topk_elements = arr[indices]
    # Sort the top k elements and their indices
    sorted_indices = indices[np.argsort(topk_elements)[::-1]]
    topk_elements_sorted = arr[sorted_indices]
    return topk_elements_sorted, sorted_indices




def logsumexp_by_group(x, b):
    """
    Compute the logsumexp of elements in x indexed by counts in b in a numerically stable way.
    
    Args:
    x (torch.Tensor): Input tensor of size N.
    b (list or torch.Tensor): tensor of counts for each group.
    
    Returns:
    torch.Tensor: Resulting tensor of size equal to the length of b.
    """
    # Convert b to a tensor if it's a list
    
    # Generate the indices from the counts using repeat_interleave
    indices = torch.repeat_interleave(torch.arange(len(b)), b)
    
    # Number of unique indices in b
    num_classes = len(b)

    # Step 1: Compute the maximum value for each group
    max_vals = torch.zeros(num_classes, dtype=x.dtype).scatter_reduce(0, indices, x, reduce='amax')

    # Step 2: Subtract the maximum value from each element in x
    x_stable = x - max_vals[indices]

    # Step 3: Compute the exponentials of the stabilized x
    x_exp = torch.exp(x_stable)

    # Step 4: Initialize a tensor to hold the sums of exponentials
    sum_exp = torch.zeros(num_classes, dtype=x.dtype).scatter_add(0, indices, x_exp)

    # Step 5: Compute the log of the summed exponentials and add the max values back
    logsumexp = torch.log(sum_exp) + max_vals

    return logsumexp


import os
import time
from openai import AzureOpenAI

# Set your Azure OpenAI API key and other configurations here
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_URL")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Initialize Azure OpenAI client with key-based authentication
if subscription_key is not None:
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version=api_version,
    )
else:
    print('To use OpenAI API you must provide AZURE_OPENAI_API_KEY as an environment variable')
    clinet = None

def get_chatgpt_response(prompt, model="gpt-35-turbo-16k-0613", max_retries=3, retry_delay=5):
    """
    Get a response from ChatGPT for a given prompt with error handling.

    Args:
    - prompt (str): The input prompt to send to ChatGPT.
    - model (str): The model to use (default is the deployment name).
    - max_retries (int): The maximum number of retries in case of failure (default is 3).
    - retry_delay (int): The delay between retries in seconds (default is 5).

    Returns:
    - str: The response from ChatGPT.
    """
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.0,  # No randomness
                top_p=1.0,        # No randomness
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")

        if attempt < max_retries - 1:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    raise Exception('ERROR: Azure OpenAI API is not working as intended (maybe down?)')

