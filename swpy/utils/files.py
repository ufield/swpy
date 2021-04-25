import os

def make_dirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)