from pathlib import Path
import os
from logging import INFO

RANDOM_SEED = 521

ROOT_DIR = Path(".").resolve()
DATASET_DIR = ROOT_DIR / "Dataset"
OUTPUT_DIR = ROOT_DIR / "output"
LOG_DIR = OUTPUT_DIR / "log"
LOG_LEVEL = INFO

INS_CORE_FULFIL = False

init_dir_list = [OUTPUT_DIR, LOG_DIR]


def init_dir():
    for path in init_dir_list:
        if not os.path.exists(path):
            os.mkdir(path)


init_dir()
