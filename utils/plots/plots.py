import numpy as np
import matplotlib.pyplot as plt

# Plot audio signal
def plotAudio(audio, figsize=(6, 4), ylim=True):
    fig = plt.figure(figsize=figsize)
    plt.title("Raw wave ")
    plt.ylabel("Amplitude")
    plt.plot(np.linspace(0, 1, len(audio)), audio)
    if ylim:
        plt.ylim((-1, 1))
    plt.show()


def plotAudioPair(audio1, audio2, figsize=(4, 4), ylim=True, horizontal=True):
    if horizontal:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    ax1.plot(np.linspace(0, 1, len(audio1)), audio1)
    ax1.set_title("Original")
    ax1.set(xlabel="Time", ylabel="Amplitude")

    ax2.plot(np.linspace(0, 1, len(audio2)), audio2)
    ax2.set_title("Augmented")
    ax2.set(xlabel="Time", ylabel="Amplitude")

    if ylim:
        ax1.set_ylim((-1, 1))
        ax2.set_ylim((-1, 1))

    fig.tight_layout()
    # fig.show()
