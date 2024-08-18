import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader

from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, path, source_voice, target_voice, sample_rate=44100, n_fft=2048, hop_length=518, n_mels=256):
        self.source_voice = source_voice
        self.target_voice = target_voice
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        filenames = glob((Path(path) / Path('**')).as_posix(), recursive=True)

        self.filenames = self._filter_filenames(filenames)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename_a, filename_b = self.filenames[idx]

        audio_a, _ = torchaudio.load(filename_a, normalize=True)
        audio_b, _ = torchaudio.load(filename_b, normalize=True)

        # Ensure audio length is at least 3 seconds
        if audio_a.size(1) < 3 * self.sample_rate:
            audio_a = F.pad(audio_a, (0, 3 * self.sample_rate - audio_a.size(1)))
        if audio_b.size(1) < 3 * self.sample_rate:
            audio_b = F.pad(audio_b, (0, 3 * self.sample_rate - audio_b.size(1)))
        
        #Ensure audio A and B are equal length
        if audio_a.size(1) < audio_b.size(1):
            audio_a = F.pad(audio_a, (0, audio_b.size(1) - audio_a.size(1)))
        else:
            audio_b = F.pad(audio_b, (0, audio_a.size(1) - audio_b.size(1)))

        # Randomly select a 3-second segment
        start = torch.randint(audio_a.size(1) - 3 * self.sample_rate + 1, (1,))
        audio_a = audio_a[:, start:start+3*self.sample_rate]
        audio_b = audio_b[:, start:start+3*self.sample_rate]

        # Compute mel-spectrograms
        S_a = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)(audio_a)
        S_b = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels)(audio_b)

        # Convert to dB
        S_db_a = torchaudio.transforms.AmplitudeToDB()(S_a)
        S_db_b = torchaudio.transforms.AmplitudeToDB()(S_b)

        return S_db_a, S_db_b

    def _filter_filenames(self, filenames):
        """Filter filenames based on source and target voices."""
        return [(f_a, f_b) for f_a in filenames for f_b in filenames if self.source_voice in Path(f_a).parent.name and self.target_voice in Path(f_b).parent.name]

if __name__ == "__main__":
    data_path = 'data/'
    source_voice = 'SF2'
    target_voice = 'TM1'

    batch_size = 1
    epochs = 2000
    checkpoint_freq = 100

    dataset = AudioDataset(data_path, source_voice, target_voice)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    im = next(iter(dataloader))

    plt.subplot(2, 1, 1)
    plt.imshow(im[0].squeeze().numpy(), cmap='viridis')

    plt.subplot(2, 1, 2)
    plt.imshow(im[1].squeeze().numpy(), cmap='viridis')
    
    plt.show()
    

    