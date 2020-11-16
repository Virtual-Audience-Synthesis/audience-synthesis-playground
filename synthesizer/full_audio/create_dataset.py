import os
import numpy as np
from multiprocessing import Pool
import time

import sys
from pathlib import Path

try:
    sys.path.append(str(Path().cwd().parent.parent))
    sys.path.append(str(Path().cwd()))
except IndexError:
    pass

# import utils.audio as a
# import utils.plots as plots
import utils as utils


def main():
    SCRIPT_DIR = Path(__file__).parent
    LABELS = SCRIPT_DIR.joinpath("../../datasets/dataset_laughter_single_22050.csv")
    input_df = utils.misc.readCSV(LABELS)
    input_df.head()

    # get female laughter
    idx_female = input_df.index[input_df["Gender"] == "female"]
    females = input_df.iloc[idx_female]
    print(females)
    print(females.iloc[0])
    print(input_df.iloc[0].Filename)

    # get male laughter
    idx_male = input_df.index[input_df["Gender"] == "male"]
    males = input_df.iloc[idx_male]
    print(males)
    print(males.iloc[0])
    print(input_df.iloc[0].Filename)

    PATH = SCRIPT_DIR.joinpath("../../datasets/dataset_laughter_single_22050/")
    SR = 22050

    dataset = []
    for i in np.arange(len(input_df)):
        audio, sr = utils.audio.loadAudio(
            PATH.joinpath(input_df.iloc[i].Filename).with_suffix(".wav"),
            sr=SR,
            fix_length=True,
            length=10,
        )
        dataset.append(audio)

    # ### Create N Female audio files without alpha, beta

    N = 500

    t = time.time()
    PATH = SCRIPT_DIR.joinpath("female_audios")
    PATH.mkdir(parents=True, exist_ok=True)

    len_females = len(females)

    # in percent
    min_speed, max_speed = 0.9, 1.2
    speed_range = max_speed - min_speed  # 2
    max_noise = 0.001
    min_pitch, max_pitch = -4, 4
    audio = np.zeros_like(dataset[0])

    idx_list = utils.misc.getFemaleList(input_df, N)

    # add radial decay
    alpha = np.random.rand(N)
    beta = np.reciprocal(np.sqrt(np.arange(1, N + 1)))

    # pdb.set_trace()
    noise_factor = list(np.random.rand(N) * max_noise)
    shift_max = np.random.rand(N) * 3  # in seconds
    shift = [np.random.randint(0, int(SR * i)) for i in shift_max]
    direction = ["left" if np.random.randint(0, 2) else "right" for i in np.arange(N)]
    pitch_factor = list(
        np.random.randint(min_pitch, max_pitch, N)
    )  # pitch > 0 higher pitch, else deeper voice
    speed_factor = list(np.random.rand(N) * speed_range + min_speed)
    data = [dataset[idx] for idx in idx_list]

    # create arguments as list for parallel computing
    func_args = [
        (
            data[i],
            SR,
            noise_factor[i],
            shift[i],
            direction[i],
            pitch_factor[i],
            speed_factor[i],
        )
        for i in np.arange(N)
    ]

    # multiprocessing
    with Pool(8) as p:
        out = p.starmap(utils.audio.augmentAudio, func_args)

    # pdb.set_trace()
    np.save(PATH.joinpath("female_audios_500.npy"), np.array(out))  # two channel audio
    elapsed = time.time() - t
    print(elapsed)

    # ### Create N Male audio files without alpha, beta
    #

    N = 500

    t = time.time()
    PATH = SCRIPT_DIR.joinpath("male_audios")
    PATH.mkdir(parents=True, exist_ok=True)

    len_males = len(males)

    # in percent
    min_speed, max_speed = 0.9, 1.2
    speed_range = max_speed - min_speed  # 2
    max_noise = 0.001
    min_pitch, max_pitch = -4, 4
    audio = np.zeros_like(dataset[0])

    idx_list = utils.misc.getMaleList(input_df, N)

    # add radial decay
    alpha = np.random.rand(N)
    beta = np.reciprocal(np.sqrt(np.arange(1, N + 1)))

    # pdb.set_trace()
    noise_factor = list(np.random.rand(N) * max_noise)
    shift_max = np.random.rand(N) * 3  # in seconds
    shift = [np.random.randint(0, int(SR * i)) for i in shift_max]
    direction = ["left" if np.random.randint(0, 2) else "right" for i in np.arange(N)]
    pitch_factor = list(
        np.random.randint(min_pitch, max_pitch, N)
    )  # pitch > 0 higher pitch, else deeper voice
    speed_factor = list(np.random.rand(N) * speed_range + min_speed)
    data = [dataset[idx] for idx in idx_list]

    # create arguments as list for parallel computing
    func_args = [
        (
            data[i],
            SR,
            noise_factor[i],
            shift[i],
            direction[i],
            pitch_factor[i],
            speed_factor[i],
        )
        for i in np.arange(N)
    ]

    # multiprocessing
    with Pool(8) as p:
        out = p.starmap(utils.audio.augmentAudio, func_args)

    # pdb.set_trace()
    np.save(PATH.joinpath("male_audios_500.npy"), np.array(out))  # two channel audio
    elapsed = time.time() - t
    print(elapsed)


if __name__ == "__main__":
    main()
