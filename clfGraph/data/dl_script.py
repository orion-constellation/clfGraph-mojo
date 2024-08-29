'''
Download your dataset in parts

'''
from dataclasses import dataclass
import requests
import os
from src.custom_logging import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Union 
from clfGraph.src.constants import PROJECT_NAME  
from dotenv import load_dotenv
load_dotenv() 

##### SET CONFIG #####
LOG_LEVEL= os.environ("LOG_LEVEL", "debug")

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
    file_list: Union[List[str], List[int]]
    dataset_name: str
    max_workers: int = 5
    base_url: str
    destination_folder: str
    chunk_size: int = 3
    max_retries: int = 3
    parallel: bool = True
    train_test: Union[str, str, str]

'''
Download files in chunks in parallel using ThreadPoolExecutor.

Parameters:
- file_list (List): List of files to download in chunks.

Returns:
- None
'''
def download_chunk_parallel(self, file_list):
    logger.info("Downloading chunks in parallel")
    for i, file_name in enumerate(self.file_list):
        retries = 3
        while retries < self.max_retries:
            try:
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = []
                    for i in range(0, len(file_list) + 1, self.chunk_size):
                        start_index = i
                        end_index = min(i + self.chunk_size, len(file_list + 1))
                        futures.append(executor.submit(self.download_chunk, start_index, end_index))
                    for future in futures:
                        future.result()  # Wait for all downloads to complete
                        
                        retries += 1
                        print(f"Failed to download {file_name}. Retry {retries}/{self.max_retries}. Error: {e}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download {file_name}. Error: {e}")
            if retries == self.max_retries:
                print(f"Failed to download {file_name} after {self.max_retries} retries.")

'''
Download all files at base_url in parallel using ThreadPoolExecutor.

Parameters:
- file_list (List): List of files to download in chunks.

Returns:
- None
'''
def download_all_parallel(self, file_list=file_list):
    logger.info("###Downloading all files from:\n {file_list}")
    total_files = len(file_list)
    def download_all_parallel(self):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.download_file, file_name): file_name
                for file_name in self.file_list
            }
            for future in as_completed(futures):
                file_name = futures[future]
                try:
                    result = future.result()
                    if result:
                        logger.info(f"Successfully downloaded {file_name} in parallel.")
                    else:
                        logger.error(f"Failed to download {file_name} in parallel.")
                except Exception as e:
                    logger.error(f"Exception occurred while downloading {file_name}: {e}")



DL_TYPE="train"
dest_folder='./{PROJECT}/raw/'

download_session = DownloadSession(
    project_name="ClfGraph",
    file_list=file_list,
    dataset_name="CICIMT2024",
    max_workers=5,
    base_url="http://205.174.165.80/IOTDataset/CICIoMT2024/Dataset/WiFI_and_MQTT/attacks/CSV/train",
    file_list=file_list,
    dest_folder='./{PROJECT}/raw/',
    chunk_size=3, 
    parallel=True,
    train_test="train"
)

if os.path.exists(f"../dataset/{PROJECT}/{download_session.train_test}"):
    logger.info(f"Folder {PROJECT}/{download_session.train_test} already exists")
else:
    os.makedirs(f"{download_session.dest_folder}/{download_session.train_test}")
    logger.info(f"{dest_folder}/{download_session.train_test} created")



if __name__ == "__main__":
    if DL_TYPE=="train" | DL_TYPE=="test":
        download_session.download_chunk_parallel(file_list)
    elif DL_TYPE=="test":
        download_session.download_chunk_parallel(file_list)
    elif DL_TYPE=="all":
        download_session.download_all_parallel(file_list)
    else:
        print("Invalid download type")
