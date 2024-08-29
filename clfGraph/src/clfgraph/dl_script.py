'''
Download your dataset if it is in different files

'''
import logging
import os
import timeit
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Union

import dask as dd
import ray
import requests
from clfgraph.constants import DATA_PATH, PROJECT_NAME
from clfgraph.custom_logging import configure_logging
from clfgraph.sklearn_baseline.models import init_wandb, params
from clfgraph.test_data import test_data
from dotenv import load_dotenv
from tqdm import tqdm
import sentry_sdk
sentry_sdk.init(dsn="")

import wandb

logger=configure_logging(level="DEBUG")
if logger.level == "DEBUG":
    test_data = test_data()

logger.info("Logger works")

load_dotenv() 

##### SET CONFIG #####
LOG_LEVEL= logger.setLevel("DEBUG")
DL_TYPE="train"
DEST_FOLDER=DATA_PATH

# List of Files to Download
# Can call all or this list
file_list = ["TCP_IP-DDoS-UDP5_train.pcap.csv",
    "TCP_IP-DDoS-UDP6_train.pcap.csv",
    "TCP_IP-DDoS-UDP7_train.pcap.csv",
    "TCP_IP-DDoS-UDP8_train.pcap.csv",
    "TCP_IP-DoS-ICMP1_train.pcap.csv",
    "TCP_IP-DoS-ICMP2_train.pcap.csv",
    "TCP_IP-DoS-ICMP3_train.pcap.csv",
    "TCP_IP-DoS-ICMP4_train.pcap.csv",
    "TCP_IP-DoS-SYN1_train.pcap.csv",
    "TCP_IP-DoS-SYN2_train.pcap.csv",
    "TCP_IP-DoS-SYN3_train.pcap.csv",
    "TCP_IP-DoS-SYN4_train.pcap.csv",
    "TCP_IP-DoS-TCP1_train.pcap.csv",
    "TCP_IP-DoS-TCP2_train.pcap.csv",
    "TCP_IP-DoS-TCP3_train.pcap.csv",
    "TCP_IP-DoS-TCP4_train.pcap.csv",
    "TCP_IP-DoS-UDP1_train.pcap.csv",
    "TCP_IP-DoS-UDP2_train.pcap.csv",
    "TCP_IP-DoS-UDP3_train.pcap.csv",
    "TCP_IP-DoS-UDP4_train.pcap.csv"
]


'''
Represents a download session for a specific project and dataset.

Parameters:
- project_name (str): The name of the project.
- file_list (Union[List[str], List[int]): List of files to download.
- dataset_name (str): The name of the dataset.
- max_workers (int): Maximum number of workers for parallel downloading (default is 5).
- base_url (str): The base URL for downloading the files.
- destination_folder (str): The folder where the files will be saved.
- chunk_size (int): Size of each download chunk (default is 3).
- max_retries (int): Maximum number of retries in case of download failure (default is 3).
- parallel (bool): Flag to enable parallel downloading (default is True).
- train_test (Union[str, str, str]): Train and test dataset identifiers.
'''
@dataclass
class DownloadSession:
    project_name: str
    base_url: str
    train_test: Union[str, str]
    file_list: Union[List[str], List[int]]
    dest_folder: str
    dataset_name: str
    max_workers: int = 5
    chunk_size: int = 3
    max_retries: int = 3
    parallel: bool = True
    
    



download_session = DownloadSession(
    project_name="ClfGraph",
    base_url="http://205.174.165.80/IOTDataset/CICIoMT2024/Dataset/WiFI_and_MQTT/attacks/CSV/train",
    file_list=file_list,
    dest_folder=DEST_FOLDER,
    dataset_name="CICIMT2024_0x0",
    max_workers=5,
    chunk_size=3,
    max_retries=3, 
    parallel=True,
    train_test="train"
)
wandb.config=params
dataset_path="data/dataset/train/"
if os.path.exists(f"{dataset_path}{download_session.train_test}"):
    logger.info(f"{dataset_path}{download_session.train_test} already exists")
else:
    os.makedirs(f"{dataset_path}{download_session.train_test}")
    logger.info(f"{dataset_path}{download_session.train_test} created")


'''
Download files in chunks in parallel using ThreadPoolExecutor.

Parameters:
- file_list (List): List of files to download in chunks.

Returns:
- None
'''
def download_chunk_parallel(self, dl_session=download_session):
    chunk_run = init_wandb(project=PROJECT_NAME, config=params, name=f"downloading_dataset: {download_session.dataset_name}", job_type="dataset")
    logger.info("Downloading chunks in parallel")
    logger.info("###Downloading all files from:\n {file_list}")
    #total_files = len(file_list)
    for i, file_name in tqdm(iterable=enumerate(file_list),colour="green", ascii=True, timeit=True):
        retries = 3
        while retries < dl_session.max_retries:
            try:
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = []
                    for i in tqdm(range(0, len(file_list) + 1, download_session.chunk_size), colour="green", ascii=True):
                        start_index = i
                        end_index = min(i + dl_session.chunk_size, len(file_list + 1))
                        futures.append(executor.submit(download_chunk_parallel, start_index, end_index))
                    for future in futures:
                        future.result()  # Wait for all downloads to complete
                        
                        retries += 1
                        logger.error(f"Failed to download {file_name}. Retry {retries}/{download_session.max_retries}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download {file_name}. Error: {e}")
            if retries == dl_session.max_retries:
                logger.error(f"Failed to download {file_name} after {self.max_retries} retries.")
    wandb.log({"futures":futures})
'''
Download all files at base_url in parallel using ThreadPoolExecutor.

Parameters:
- file_list (List): List of files to download in chunks.

Returns:
- None
'''


def download_all_parallel(self, file_names):
    with ThreadPoolExecutor(max_workers=download_session.max_workers) as executor:
        futures = {
            executor.submit(download_all_parallel, file_name): file_name
            for file_name in file_list
        }
        for future in tqdm(as_completed(futures), colour="green", ascii=True, mininterval=0.1):
            file_name = futures[future]
            try:
                result = future.result()
                if result:
                    logger.info(f"Successfully downloaded {file_name} in parallel.")
                else:
                    logger.error(f"Failed to download {file_name} in parallel.")
            except Exception as e:
                logger.error(f"Exception occurred while downloading {file_name}: {e}")






if __name__ == "__main__":
    if DL_TYPE=="train":
        download_chunk_parallel(file_list)
    elif DL_TYPE=="test":
        download_chunk_parallel(file_list)
    elif DL_TYPE=="all":
        download_session.download_all_parallel(file_list)
    else:
        print("Invalid download type")
