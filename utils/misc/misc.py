import pandas as pd
import numpy as np
import random

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
    idx_female = df.index[df['Gender'] == 'female']
    #females = df.iloc[idx_female]
    # Get male idxs
    idx_male = df.index[df['Gender'] == 'male']
    #males = df.iloc[idx_male]
    
    female_num = int(ratio * N)
    male_num = N - female_num
    
    idx_list = random.choices(idx_female, k=female_num)
    idx_list = idx_list + random.choices(idx_male, k=male_num)
        
    #idx_list = random.shuffle(idx_list)
    return idx_list
