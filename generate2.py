import torch
import torchaudio
import torch.nn.functional as F
from pathlib import Path
from soundstream import SoundStream
from data import SoundDataset,get_dataloader
from trainer import cycle
from scipy.signal import find_peaks
import numpy as np
import math
import matplotlib.pyplot as plt

class Generator():
    def __init__(
            self,
            model_path="./results/soundstream.best.pt",
            file_path='/home/nhatz/Documents/University/thesis/code/data/TAU-urban-acoustic-scenes-2022-mobile-development/test',
            batch_size=1,
            ):
        self.soundstream = SoundStream( #encoder decoder only
            emb_size=256,
            strides=(2,4,5,8),
            target_sample_hz=24000,
        )

        self.soundstream.load(model_path)
        self.batch_size = batch_size
        
        self.ds = SoundDataset(
            folder = file_path,
            max_length=24000, #1 second
            target_sample_hz=self.soundstream.target_sample_hz,
            seq_len_multiple_of=self.soundstream.seq_len_multiple_of
        )

        self.dl = get_dataloader(self.ds,batch_size=batch_size,num_workers=0)
        self.dl_iter = cycle(self.dl)
        self.results_folder = Path("./results")
        self.results_folder.mkdir(parents=True,exist_ok=True)
    
    def prepare_wave(self,wave):
        log_nrg = self.short_term_log_energy(wave,k=100)
        peaks,_ = find_peaks(log_nrg,prominence=3)
        print(len(peaks))
        plt.plot(peaks, log_nrg[peaks], "xr"); plt.plot(log_nrg); plt.legend(['prominence'])
        plt.show()

    def short_term_log_energy(self,wave,k):
        return torch.tensor([torch.log10(torch.sum(torch.square(wave[i:i+k]))) for i in range(len(wave)-k+1)])

    def _short_term_log_energy(self,wave,k):
        for i in range(len(wave)-k+1):
            yield torch.log(torch.sum(torch.square(wave[i:i+k])))

    def test_peaks(self):
        wave, = next(self.dl_iter)
        self.prepare_wave(wave.squeeze(0))
    
    def generate(self,num_samples):
        count = math.ceil(num_samples/self.batch_size)
        self.soundstream.eval()
        with torch.no_grad():
            for _ in range(count):
                wave, = next(self.dl_iter)
                wave = self.prepare_wave(wave)
                recons = self.soundstream(wave)
                for ind,recon in enumerate(recons.unbind(dim=0)):
                    filename = str(self.results_folder / f'long_{ind}.flac')
                    torchaudio.save(filename,recon.cpu().detach(),self.soundstream.target_sample_hz)
                    print(f'{ind}: saving to {str(self.results_folder)}')
    

if __name__=="__main__":
    gen = Generator()
    gen.test_peaks()






