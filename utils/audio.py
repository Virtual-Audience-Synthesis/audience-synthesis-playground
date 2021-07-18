import librosa
import numpy as np
import IPython.display as ipd
import os
from tqdm import tqdm
import noisereduce as nr
import random
import sys

if "../../" not in sys.path:
    sys.path.insert(1, "../../")

from utils import audio, misc, plots, whistle, clap
import pdb


def spawnLaughter(
    n_person,
    ratio_female,
    female_laughter_path,
    male_laughter_path,
    stereo=True,
    fs=22050,
    t_len=10,
):
    """Spawn audio from pre-recorded laughter files of 10 secs, and fs, SR=22050
    TODO: Adapt length of audio if t_len is set
    TODO: Adapt fs of audio if fs is set
    TODO: Return audio and sampling rate SR, fs
    Make sure full_audio/female_audios and full_audio/male_audios are provided

    Args:
        n_person (int): Number of persons laughing
        ratio_female (float): Ratio of female persons in percent (0.0-1.0)
        stereo (boolean): Synthesize stereo audio if true, mono if false
        fs (int): Sampling rate of audio
        t_len (float): Seconds of audio recording

    Returns:
        numpy.ndarray: (1, int(t_len*fs)) if mono else (2, int(t_len*fs))

    """
    # load (fe)male audio sequences without alpha, beta
    female_audios = np.load(female_laughter_path)
    male_audios = np.load(male_laughter_path)

    n_female = int(n_person * ratio_female)
    n_male = n_person - n_female

    # get male indices and audio signals
    male_idx = np.random.choice(male_audios.shape[0], n_male)
    males = [male_audios[i] for i in male_idx]

    # get female indices and audio signals
    female_idx = np.random.choice(female_audios.shape[0], n_female)
    females = [female_audios[i] for i in female_idx]

    # create list for beta fallof
    beta = np.reciprocal(np.sqrt(np.arange(1, n_person + 1)))
    # concatenate both lists and shuffle
    mixed_audio = np.array(random.sample(males + females, n_person))
    if stereo:
        # create list of alpha values for stereo
        alpha = np.random.rand(n_person)
        # synthesize left audio
        left_audio = n_person * alpha[:, np.newaxis] * beta[:, np.newaxis] * mixed_audio
        # synthesize right audio
        right_audio = (
            n_person * (1 - alpha[:, np.newaxis]) * beta[:, np.newaxis] * mixed_audio
        )
        # concatenate and compute mean
        audio = np.array([left_audio.mean(0), right_audio.mean(0)])
    else:
        audio = beta[:, np.newaxis] * mixed_audio

    return audio


# create whistles
def spawnWhistles(
    fs,
    t_len=1,
    can_radius=200,
    pea_radius=5,
    bump_radius=160,
    norm_can_loss=0.97,
    gravity=15.0,
    norm_tick_size=0.004,
    env_rate=0.001,
    sample_rate=44100,
    fipple_freq_mod=0.25,
    fipple_gain_mod=0.5,
    blow_freq_mod=0.2,
    noise_gain=0.3,
    base_freq=3000,
    sine_rate=2600,
    pole=0.95,
    load=True,
    load_path=None,
):
    if load:
        assert load_path is not None, "Load path must be defined!"

    whistle_len = int(t_len * fs)
    output = np.zeros(whistle_len)
    if not load:
        w = whistle.Whistle(
            can_radius=can_radius,
            pea_radius=pea_radius,
            bump_radius=bump_radius,
            norm_can_loss=norm_can_loss,
            gravity=gravity,
            norm_tick_size=norm_tick_size,
            env_rate=env_rate,
            sample_rate=sample_rate,
            fipple_freq_mod=fipple_freq_mod,
            fipple_gain_mod=fipple_gain_mod,
            blow_freq_mod=blow_freq_mod,
            noise_gain=noise_gain,
            base_freq=base_freq,
            sine_rate=sine_rate,
            pole=pole,
        )
        for i in range(whistle_len):
            w.tick()
            output[i] = w.last_frame
    else:
        # TODO: load
        output = loadAudio(load_path, sr=44100)[0]

    return output


def spawnClaps(n_persons, fs, t_len):
    # fs=8192                            # Sampling frequency in Hz
    n_samples = round(fs * t_len)  # Total length of audio clip generated in seconds
    # NClap=round(Fs*0.1);               # Length of audio signal for individual clap
    # RiseTime=round(0.032*Fs);          # Exponentially rising attack segment of envelope 3.2 ms
    # Base=0.99^(44100/Fs);              # Make shape of envelope independent from Fs
    # BaseDecay=Base^0.1;                # Slower decay of envelope to fake reverb
    # R=0.9;                             # Pole radius of cavity resonator
    claps_n = fs * 1  # 1 second
    theta = (
        np.pi / 2.5
    )  # Resonant frequency of cavity resonator: adjust for different hand clapping styles; higher for flat hands
    theta_std = np.pi / 6  # Vary hand clapping styles
    # b=1;                               # Cavity transfer function numerator - arbitary
    # env=min(Base.^(RiseTime-t),BaseDecay.^(t-RiseTime)); % Generate envelope - fast rise, slow decay
    avgOOI = 0.4  # Time between clap in secs. Natural 0.4 Enthusiastic 0.3 Bored 0.6
    stdOOI = avgOOI / 16  # Time between claps varies with this st dev
    swell = 1  # Applause increases over Swell seconds at the beginning
    fade = 5  # Applause fades over Fade seconds at the end
    out_left = np.zeros(n_samples)  # Left channel
    out_right = np.zeros_like(out_left)
    # row_n=50                        # Number of rows
    # pp_per_row=1                    # People per row
    n_person = (
        n_persons  # Number of persons clapping -------------------------------------
    )
    #     if (row_n*pp_per_row)<n_person:       # Check if enough seats are available
    #         print("More people than seats available")
    is_whistle = True
    row = 0

    t = np.linspace(1, claps_n, claps_n)  # create time vector
    env = audio.clap.envelope(t, fs)

    for i in range(1, n_person + 1):
        # a=[1 -2*R*cos(Theta+(rand-0.5)*ThetaStD) R*R];  # Cavity transfer function denominator
        #   Onset=round(rand*AverageOOI*Fs)+1;              # first clap onset uniformly distributed without clapping interval
        onset = (
            round(np.random.random() * fs * swell) + 1
        )  # first clap onset uniformly between 0 and Swell seconds
        end_clap = (
            n_samples - np.random.random() * fs * fade
        )  # clapping stops, uniformly distributed over Fade seconds at the end
        alpha = np.random.random()  # random distribute between left and right channel

        #         if np.mod(i-1, pp_per_row):                         # Fill rows; increment index when row is full 10 persons / row
        #             row = row + 1

        beta = np.power(i, -0.5)
        # beta = max_gain-0.2*row/row_n

        OOI = avgOOI + np.random.random() * stdOOI  # Random OOI, no (a)synch

        while onset + claps_n < end_clap:

            # Filter single clap
            y = clap.clap(claps_n, env, fs)

            # reverb = reverberator('PreDelay', 0.5, 'WetDryMix', 1, 'SampleRate', Fs);
            # y = reverb(y);
            # pdb.set_trace()
            out_left[onset : onset + claps_n] = (
                out_left[onset : onset + claps_n] + beta * alpha * y
            )
            out_right[onset : onset + claps_n] = (
                out_right[onset : onset + claps_n] + beta * (1 - alpha) * y
            )

            onset = onset + round(fs * OOI)
    return np.stack([out_left, out_right])


# create adsr envelope
def adsr(x, t, a, d, s_val, r):
    """
    x input vector
    t signal total len
    a attack duration
    d decay duration
    s_val sustain level
    r release duration
    """
    m0 = 1 / a
    b0 = 0
    m1 = (s_val - 1) / d
    b1 = 1 - a * m1

    s = t - r
    m2 = s_val / (s - t)
    b2 = -t * m2
    y = []
    for i in x:
        if i < a:
            y.append(m0 * i)
        if (i >= a) & (i < a + d):
            y.append(m1 * i + b1)
        if (i >= a + d) & (i < s):
            y.append(s_val)
        if i > s:
            y.append(m2 * i + b2)
    return np.asarray(y)


# load audio signal
def loadAudio(path, sr=22050, fix_length=False, length=10):
    audio, sr = librosa.load(path, sr=sr)
    if fix_length:
        audio = librosa.util.fix_length(
            audio, int(sr * length)
        )  # 10.24s to get a mel spec . x 64
    return audio, sr


# play audio signal
def playSingleAudio(audio, sr=22050):
    return ipd.Audio(audio, rate=sr)


# play audio signal
def playMultiAudio(audio, sr=22050):
    for a in audio:
        display(ipd.Audio(a, rate=sr))


# load audio files as dataset
def loadAudioFiles(path, sr):
    files = os.listdir(path)
    dataset = []
    for f in tqdm(files):
        if f.split(".")[-1] != "wav":
            continue
        # pdb.set_trace()
        audio, sr = loadAudio(path + f, sr=sr)
        dataset.append(audio)
    return np.array(dataset)


# create chunks
def createAudioChunks(dataset, chunk_len, sr):  # chunk_len in ms
    dataset_chunks = []
    chunk_size = int(sr * chunk_len / 1000)
    for chunk in dataset:
        # pdb.set_trace()
        w = misc.slidingWindow(chunk, chunk_size, chunk_size)
        if len(w) != 0:  # if chunks zero (= less than Sr*x sec chunks)
            dataset_chunks.append(w)
    return np.array(dataset_chunks)


# Adding white noise
def addNoise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data


# shift directions right, left
def addShift(data, shift, shift_direction):

    if shift_direction == "right":
        shift = -shift

    augmented_data = np.roll(data, shift)

    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data


# change pitch in fractional half-steps
def changePitch(audio, sr, pitch_factor):
    return librosa.effects.pitch_shift(audio, sr, n_steps=pitch_factor)


# change pitch in fractional half-steps


def changeSpeed(audio, speed_factor, keep_dim=True):
    # pdb.set_trace()
    output = np.zeros_like(audio)
    audio_stretched = librosa.effects.time_stretch(audio, speed_factor)
    if keep_dim:
        if speed_factor > 1:
            idx = np.random.randint(0, len(audio) - len(audio_stretched))
            output[idx : idx + len(audio_stretched)] = audio_stretched
        else:
            idx = np.random.randint(0, len(audio_stretched) - len(audio))
            output = audio_stretched[idx : idx + len(output)]
    else:
        output = audio_stretched
    return output


def augmentAudio(
    audio, sr, noise_factor, shift, shift_direction, pitch_factor, speed_factor
):
    # white noise
    audio = addNoise(audio, noise_factor=noise_factor)
    # time shift
    audio = addShift(audio, shift, shift_direction)
    # change pitch
    audio = changePitch(audio, sr, pitch_factor)
    # change speed
    audio = changeSpeed(audio, speed_factor, keep_dim=True)
    return audio


def augmentAudioStereo(
    audio,
    alpha,
    beta,
    sr,
    noise_factor,
    shift,
    shift_direction,
    pitch_factor,
    speed_factor,
):
    # white noise
    audio = addNoise(audio, noise_factor=noise_factor)
    # time shift
    audio = addShift(audio, shift, shift_direction)
    # change pitch
    audio = changePitch(audio, sr, pitch_factor)
    # change speed
    audio = changeSpeed(audio, speed_factor, keep_dim=True)
    # create left audio
    left = alpha * beta * audio
    # create right audio
    right = (1 - alpha) * beta * audio

    return left, right


# create and apply envelope (swell, fade in seconds)
def createEnvelope(len_signal, sr, swell, fade):

    env = np.ones(len_signal)
    t = np.linspace(0, int(len_signal / sr), len_signal)
    # pdb.set_trace()
    # swell
    t_swell = np.linspace(0, swell, int(swell * sr))
    swell_func = np.power(2, t_swell)
    # print(len(swell_func))
    env_swell = np.clip(
        (swell_func - np.min(swell_func)) / (np.max(swell_func) - np.min(swell_func)),
        0,
        1,
    )
    env[: len(t_swell)] = env_swell

    # fade
    t_fade = np.linspace(0, fade, int(fade * sr))
    fade_func = np.power(2, -t_fade)
    env_fade = np.clip(
        (fade_func - np.min(fade_func)) / (np.max(fade_func) - np.min(fade_func)), 0, 1
    )
    env[-len(t_fade) :] = env_fade

    # apply envelope
    return env


# hmm model, noise cluster ID, Chunk Size: chunk length in s * SR, audio as np.array
def removeNoise(hmm, noise_clusterID, chunk_sz, chunks, audio):
    # pdb.set_trace()
    # hmm prediction into cluster IDs
    pred = hmm.predict(chunks.reshape(chunks.shape[0], -1))
    # check which IDs are desired cluster ID
    clusterID_lst = (pred != noise_clusterID).astype(int)
    clusterID_lst_diff = np.diff(clusterID_lst)

    cnt = 0
    idx = 0
    res = []
    flag = False
    for i, item in enumerate(clusterID_lst_diff):
        if i == 0 and clusterID_lst[i] == 0:
            idx = 0
        if item == 1:
            flag = False
            cnt += 1
            res.append((idx, cnt))
        if item == -1:
            flag = True
            idx = i + 1
            cnt = 0
        if item == 0:
            cnt += 1
        if i == len(clusterID_lst_diff) - 1 and flag:
            cnt += 1
            res.append((idx, cnt))

    # get tuple with highest key value
    try:
        z_idx, z_len = max(res, key=lambda x: x[1])
    except ValueError:
        return audio

    # select section of data that is noise
    noisy_part = audio[(z_idx - 1) * chunk_sz : (z_idx + z_len - 1) * chunk_sz]

    # if no noise class detected return original audio
    if not len(noisy_part):
        return audio
    # perform noise reduction
    reduced_noise = nr.reduce_noise(
        audio_clip=audio, noise_clip=noisy_part, verbose=False
    )
    return reduced_noise


def spawnBoos(Nperson, Fs, t_len, load_path):
    boos = np.load(load_path, allow_pickle=True)
    Nsamples = round(Fs * t_len)  #   # Total length of audio clip generated in seconds
    NClap = round(Fs * 1)  #    # Length of audio signal for individual clap
    RiseTime = round(
        0.032 * Fs
    )  #  # Exponentially rising attack segment of envelope 3.2 ms
    DecayTime = round(
        0.052 * Fs
    )  #  # Exponentially rising attack segment of envelope 3.2 ms
    Base = 0.99 ** (44100 / Fs)  ## Make shape of envelope independent from Fs
    BaseDecay = Base ** 0.1  # # Slower decay of envelope to fake reverb
    R = 0.9  ## Pole radius of cavity resonator
    Theta = (
        np.pi / 2.5
    )  #   # Resonant frequency of cavity resonator: adjust for different hand clapping styles higher for flat hands
    ThetaStD = np.pi / 6  #  # Vary hand clapping styles
    b = np.array([1])  ## Cavity transfer function numerator - arbitary
    t = np.arange(1, NClap + 1).T  #        # Index of signal for individual calp
    opt1 = Base ** (RiseTime - t)
    opt2 = BaseDecay ** (t - DecayTime)
    env = np.where(
        opt1 < opt2, opt1, opt2
    )  # # Generate envelope - fast rise, slow decay
    AverageOOI = (
        0.4  # # Time between clap in secs. Natural 0.4 Enthusiastic 0.3 Bored 0.6
    )
    StDevOOI = AverageOOI / 16  # Time between claps varies with this st dev
    Swell = 1  # Applause increases over Swell seconds at the beginning
    Fade = 5  # Applause fades over Fade seconds at the end
    outleft = np.zeros((Nsamples,))
    outright = np.zeros((Nsamples,))
    # create triangular distribution between for 150ms < OOI < 290ms
    def tri():
        return np.random.triangular(0.3, 0.4, 0.5)

    affinity = 0.0  # 0 asynch, 1 in synch
    K = 1.0 - affinity
    c1_async = 1.3
    c2_async = -0.25
    c1_sync = 3
    c2_sync = 4
    LeadOOI = 2  # s
    KProfile = [0.5, 0.5, 0.75, 0.9, 0.9, 0.75, 0.5, 0.5, 0]
    SyncProfile = [0, 0, 1, 1, 1, 1, 0, 0, 0]
    Nsec = 0

    def rand():
        return np.random.uniform()

    for i in range(1, Nperson + 1):
        a = np.array(
            [1, -2 * R * np.cos(Theta + (rand() - 0.5) * ThetaStD), R ** 2]
        )  # Cavity transfer function denominator
        #  Onset=round(rand*AverageOOI*Fs)+1       # first clap onset uniformly distributed without clapping interval
        Onset = (
            round(rand() * Fs * Swell) + 1
        )  # first clap onset uniformly between 0 and Swell seconds
        EndClap = (
            Nsamples - rand() * Fs * Fade
        )  # clapping stops, uniformly distributed over Fade seconds at the end
        alpha = rand()  # random distribute between left and right channel
        beta = (i) ** (-0.5)  # decay with distance.
        # OOI = round(Fs*(AverageOOI+randn*StDevOOI))#
        OOI = 1.0 / (c1_async + K * c2_async) * tri()  #
        while Onset + NClap < EndClap:
            #     x=np.random.random(size=(NClap,))*env # Generate single clap
            #     y=signal.lfilter(b,a,x)   # Filter single clap
            boo = boos[np.random.randint(0, len(boos))]
            start = np.random.randint(0, max(1, len(boo) - Fs))
            if len(boo) < Fs:
                temp_boo = np.zeros((Fs,))
                temp_boo[: len(boo)] = boo
                boo = temp_boo

            y = boo[start : start + Fs]  # *env
            outleft[Onset : Onset + NClap] = (
                outleft[Onset : Onset + NClap] + beta * alpha * y
            )
            outright[Onset : Onset + NClap] = (
                outright[Onset : Onset + NClap] + beta * (1 - alpha) * y
            )
            delta_phase = OOI - LeadOOI / 2
            if (Onset + NClap) % (1 * Fs) == 0:  # every second
                Nsec = Nsec + 1
                K = KProfile[Nsec]
                if SyncProfile[Nsec]:
                    if delta_phase > 0:
                        OOI = (
                            LeadOOI
                            + K / 2 * (OOI - LeadOOI)
                            - 1 / (c1_sync + c2_sync * K) * delta_phase
                        )  # accelerate
                    if delta_phase < 0:
                        OOI = (
                            LeadOOI
                            + K / 2 * (OOI - LeadOOI)
                            + 1 / (c1_sync + c2_sync * K) * (LeadOOI - abs(delta_phase))
                        )  # decelerate
                else:
                    OOI = 1 / (c1_async + K * c2_async) * tri()

            Onset = Onset + round(Fs * OOI)
            # Onset=Onset+round(1/(c1_async+K*c2_async)*random(tri,1)*Fs)

    OutStereo = np.array([outleft, outright])

    return OutStereo
