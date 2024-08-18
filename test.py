import torch
import torch.nn as nn
from torchaudio import transforms

from model import UnetGenerator
from data import AudioDataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import pyaudio
import librosa
import soundfile as sf

def reconstruct_signal(S_db, ref=1.0):
    """Builds an audio signal (numpy array) from a spectogram."""
    sample_rate = 44100
    n_fft = 2048
    hop_length = 518
    
    S = librosa.db_to_power(S_db, ref=ref)
    
    audio = librosa.feature.inverse.mel_to_audio(
        M=S,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    return audio

data_path = 'data/'
source_voice = 'SF2'
target_voice = 'TM1'

dataset = AudioDataset(data_path, source_voice, target_voice)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

sample = next(iter(dataloader))
real = sample[0]
target = sample[1]
G = UnetGenerator()
G.load_state_dict(torch.load('runs/generator.pt'))
out = G(real)

plt.subplot(3, 1, 1)
plt.imshow(real.squeeze().numpy(), cmap='viridis')

plt.subplot(3, 1, 2)
plt.imshow(target.squeeze().numpy(), cmap='viridis')

plt.subplot(3, 1, 3)
plt.imshow(out.squeeze().detach().numpy(), cmap='viridis')

plt.show()

audio_signal = out.squeeze().cpu().detach().numpy()
audio_signal = reconstruct_signal(out.squeeze().cpu().detach().numpy())

plt.subplot(2, 1, 1)
plt.plot(audio_signal)

plt.subplot(2, 1, 2)
plt.plot(reconstruct_signal(target.squeeze().numpy()))

plt.show()
sf.write('out.wav', audio_signal, samplerate=44100)

