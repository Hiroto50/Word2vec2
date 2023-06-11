from src.constants.path import ROOT_PATH

def read_file(file, path=f"{ROOT_PATH}/src/data/"):
    with open(f"./src/data/{file}") as f:
        return f.readlines()