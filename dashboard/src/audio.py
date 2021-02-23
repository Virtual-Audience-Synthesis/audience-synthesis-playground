import librosa

def read_audio(path):
    return librosa.load(path)

