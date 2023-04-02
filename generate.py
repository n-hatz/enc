import torch
import torchaudio
import torch.nn.functional as F
from pathlib import Path
from soundstream import SoundStream
from data import SoundDataset,get_dataloader
from trainer import cycle
import matplotlib.pyplot as plt

#soundstream = SoundStream.init_and_load_from("./results/soundstream.regmask.pt")

soundstream = SoundStream( #encoder decoder only
    emb_size=256,
    strides=(2,4,5,8),
    target_sample_hz=24000,
)

soundstream.load("./results/soundstream.curr.pt")
#soundstream.load("./results/soundstream.mask.1000.pt")

ds = SoundDataset(
    folder = '/home/nhatz/Documents/University/thesis/code/data/TAU-urban-acoustic-scenes-2022-mobile-development/test',
    max_length=24000, #1 second
    target_sample_hz=soundstream.target_sample_hz,
    seq_len_multiple_of=soundstream.seq_len_multiple_of
)
'''
ds2 = SoundDataset(
    folder = '/home/nhatz/Documents/University/thesis/code/bsc-thesis/results',
    max_length=24000, #1 second
    target_sample_hz=soundstream.target_sample_hz,
    seq_len_multiple_of=soundstream.seq_len_multiple_of
)
'''

dl = get_dataloader(ds,batch_size=1,num_workers=0)
dl_iter = cycle(dl)

#dl2 = get_dataloader(ds2,batch_size=1,num_workers=0)
#dl2_iter = cycle(dl2)

results_folder = Path("./results")
results_folder.mkdir(parents=True,exist_ok=True)

wave, = next(dl_iter)
#wave2, = next(dl2_iter)

#filename = str(results_folder / f'resample_{0}.flac')
#torchaudio.save(filename,wave.cpu().detach(),soundstream.target_sample_hz)
#print("saved")

#'''
soundstream.eval()

with torch.no_grad():
    #nb=4
    recons = soundstream(wave)#,new_blocks=nb)
    print(wave.size())
    print(recons.size())
    print(100*F.l1_loss(wave.unsqueeze(1),recons))
    for ind, recon in enumerate(recons.unbind(dim=0)):
        filename = str(results_folder / f'long_{ind}.flac')
        torchaudio.save(filename,recon.cpu().detach(),soundstream.target_sample_hz)
        print(f'{ind}: saving to {str(results_folder)}')
#'''



plt.plot(wave.flatten().cpu().numpy())
plt.plot(recons.flatten().cpu().numpy())
plt.show()
