import os.path
from pathlib import Path
from os import remove
from shutil import rmtree

if __name__ == '__main__':
    # files to be included in the extensions
    paths = [
        'Algorithm/FoCore',
        'MLGraph/MLGraph',
        'Utils/Metrics',
        'Utils/Timer',
        'Utils/log',
        'main',
        'constant',
    ]

    pwd_path = Path("./")
    build_tree_path = pwd_path / "build"
    if os.path.exists(build_tree_path):
        rmtree(build_tree_path)
    for path in paths:
        to_remove = pwd_path / f"{path}.c"
        if os.path.exists(to_remove):
            remove(to_remove)
        to_remove = pwd_path / f"{path}.pyx"
        if os.path.exists(to_remove):
            remove(to_remove)
        to_remove = pwd_path / f"{path}.cpython-38-darwin.so"
        if os.path.exists(to_remove):
            remove(to_remove)
        to_remove = pwd_path / f"{path}.cpython-39-x86_64-linux-gnu.so"
        if os.path.exists(to_remove):
            remove(to_remove)