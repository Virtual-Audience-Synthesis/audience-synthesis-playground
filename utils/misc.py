import pandas as pd
import numpy as np
import random
import math
import os
from tqdm import tqdm
from pathlib import Path


def change_docker_cwd(target_dir):
    """Change the current working dir if in docker"""
    if f"{os.getenv('BASE_DIR')}" == str(Path.cwd()):
        os.chdir(Path(target_dir))
        print(f"Running in docker -> Change cwd to {target_dir}")

# read csv file as data frame
def readCSV(path):
    return pd.read_csv(path)


# get random gender distribution
def getRandomDistribution(df, N):
    idx_list = [None] * N
    for i in np.arange(N):
        idx_list[i] = random.choice(list(df.index))
    return idx_list


# get requested gender distribution
def getGenderDistribution(df, ratio, N):
    # Get female idxs
    idx_female = df.index[df["Gender"] == "female"]
    # females = df.iloc[idx_female]
    # Get male idxs
    idx_male = df.index[df["Gender"] == "male"]
    # males = df.iloc[idx_male]

    female_num = int(ratio * N)
    male_num = N - female_num

    idx_list = random.choices(idx_female, k=female_num)
    idx_list = idx_list + random.choices(idx_male, k=male_num)

    # idx_list = random.shuffle(idx_list)
    return idx_list


# get female lists
def getFemaleList(df, N):
    idx_list = [None] * N
    # Get female idxs
    idx_female = df.index[df["Gender"] == "female"]

    for i in np.arange(N):
        # print(np.mod(i, len(df)))
        idx_list[i] = df.index[idx_female[np.mod(i, len(idx_female))]]

    # idx_list = random.shuffle(idx_list)
    return idx_list


# get male lists
def getMaleList(df, N):
    idx_list = [None] * N
    # Get female idxs
    idx_male = df.index[df["Gender"] == "male"]

    for i in np.arange(N):
        # print(np.mod(i, len(df)))
        idx_list[i] = df.index[idx_male[np.mod(i, len(idx_male))]]

    # idx_list = random.shuffle(idx_list)
    return idx_list

# create chunks with sliding window
def slidingWindow(x, window, stride):
    # Warning: this function changes the view of the array but the locations in memory is the same!
    shape = x.shape[:-1] + (math.floor((x.shape[-1] - window) / stride) + 1, window)
    strides = x.strides[:-1] + (
        x.strides[-1] * stride,
        x.strides[-1],
    )
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

# load precomputed feature vectors
def loadFeatureDatabase(path):
    files = os.listdir(path)
    feat_database = []
    for f in tqdm(files):
        feat_database.append(np.load(PATH + "features/" + f))
    return np.asarray(feat_database)
