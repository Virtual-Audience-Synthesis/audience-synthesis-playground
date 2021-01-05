import librosa
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd

import pdb

# load audio signal
def loadAudio(path, sr=22050, fix_length=False, length=10):
    audio, sr = librosa.load(path, sr=sr)
    if fix_length:
        audio = librosa.util.fix_length(audio, int(sr * length))  # 10.24s to get a mel spec . x 64
    return audio, sr


# play audio signal
def playAudio(audio, sr=22050):
    return ipd.Audio(audio, rate=sr) 


# Plot audio signal
def plotAudio(audio, figsize=(6, 4), ylim=True):
    fig = plt.figure(figsize=figsize)
    plt.title('Raw wave ')
    plt.ylabel('Amplitude')
    plt.plot(np.linspace(0, 1, len(audio)), audio)
    if ylim:
        plt.ylim((-1,1))
    plt.show()

def plotAudioSub(audio1, audio2, figsize=(4, 4), ylim=True, horizontal=True):
    if horizontal:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    ax1.plot(np.linspace(0, 1, len(audio1)), audio1)
    ax1.set_title("Original")
    ax1.set(xlabel='Time', ylabel='Amplitude')
    
    ax2.plot(np.linspace(0, 1, len(audio2)), audio2)
    ax2.set_title("Augmented")
    ax2.set(xlabel='Time', ylabel='Amplitude')
    
    if ylim:
        ax1.set_ylim((-1,1))
        ax2.set_ylim((-1,1))
    
    fig.tight_layout()
    #fig.show()

    
# Adding white noise 
def addNoise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data


# shift directions right, left
def addShift(data, shift, shift_direction):
   
    if shift_direction == 'right':
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
    #pdb.set_trace()
    output = np.zeros_like(audio)
    audio_stretched = librosa.effects.time_stretch(audio, speed_factor)
    if keep_dim:
        if speed_factor > 1:
            idx = np.random.randint(0, len(audio)-len(audio_stretched))
            output[idx:idx+len(audio_stretched)] = audio_stretched
        else:
            idx = np.random.randint(0, len(audio_stretched)-len(audio))
            output = audio_stretched[idx:idx+len(output)]
    else:
        output = audio_stretched
    return output

def augmentAudio(audio, sr, noise_factor, shift, shift_direction, pitch_factor, speed_factor): 
    # white noise
    audio = addNoise(audio, noise_factor=noise_factor)
    # time shift
    audio = addShift(audio, shift, shift_direction)
    # change pitch
    audio = changePitch(audio, sr, pitch_factor)
    #change speed
    audio = changeSpeed(audio, speed_factor, keep_dim=True)
    return audio


# create and apply envelope (swell, fade in seconds)
def createEnvelope(len_signal, sr, swell, fade):

    env = np.ones(len_signal)
    t = np.linspace(0, int(len_signal/sr), len_signal)
    #pdb.set_trace()
    # swell
    t_swell = np.linspace(0, swell, int(swell*sr))
    swell_func = np.power(2, t_swell)
    #print(len(swell_func))
    env_swell = np.clip((swell_func-np.min(swell_func))/(np.max(swell_func)-np.min(swell_func)), 0, 1)
    env[:len(t_swell)] = env_swell

    # fade
    t_fade = np.linspace(0, fade, int(fade*sr))
    fade_func = np.power(2, -t_fade)
    env_fade = np.clip((fade_func-np.min(fade_func))/(np.max(fade_func)-np.min(fade_func)), 0, 1)
    env[-len(t_fade):] = env_fade

    # apply envelope
    return env