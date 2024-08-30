'''
Handles HuggingFace Repo and Dataset

'''
import logging
import os
import shutil
from pathlib import Path

import logging
from constants import __VERSION__, PROJECT_NAME
from custom_logging import configure_logging
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, Repository, login

load_dotenv()
logger = configure_logging(level="DEBUG")

# Initialize logger

'''
Uploads project files and dataset to the Hugging Face Hub.

:param params: Parameters for the project.
:param __VERSION__: Version of the project.
'''
def upload_to_hf_hub(params, name, __VERSION__=__VERSION__):
    
    try:
        # Log in to Hugging Face Hub using the token from the .env file
        HF_HUB_TOKEN = os.getenv("HF_HUB_TOKEN")
        if not HF_HUB_TOKEN:
            logger.error("HF_HUB_TOKEN is not found in the .env file.")
            return

        login(token=HF_HUB_TOKEN)
        logger.info("Successfully logged in to Hugging Face Hub.")

        # Initialize Hugging Face API
        api = HfApi()
        model_name = name
        # Check if the repository exists
        repo_name = "CICIoMT2024_Attacks_Orion_v0.1.0_0x0"
        username = "synavate"
        repo_url = f"https://huggingface.co/{username}/{repo_name}"

        repos = api.list_repos_objs(token=HF_HUB_TOKEN)
        repo_exists = any(repo.repo_id == f"{username}/{repo_name}" for repo in repos)

        if not repo_exists:
            logger.info(f"Repository {repo_name} does not exist. Creating a new one.")
            api.create_repo(repo_name, token=HF_HUB_TOKEN)
        else:
            logger.info(f"Repository {repo_name} already exists.")

        # Clone the repository locally
        local_dir = os.path.join("./", repo_name)
        repo = Repository(local_dir, clone_from=repo_url)

        # Copy the .gitignore file
        if Path(".gitignore").exists():
            shutil.copy(".gitignore", local_dir / ".gitignore")
            logger.info(".gitignore copied to the repository.")

        # Apply .gitignore to all subfolders and files
        repo.git_add([".gitignore"])

        # Add required files to the repository
        files_to_add = [
            "result", "data_files", "README.md", "LICENSE",
            "data/dataset", "sklearn_baseline/"
        ]

        for file in files_to_add:
            full_path = Path(file)
            if full_path.exists():
                repo.git_add([str(full_path)])
                logger.info(f"Added {file} to the repository.")
            else:
                logger.warning(f"{file} does not exist and was not added.")

        # Commit and push changes
        repo.git_commit("Initial commit with project files.")
        repo.git_push()
        logger.info("Changes committed and pushed to the repository.")


        logger.info("Operation completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

