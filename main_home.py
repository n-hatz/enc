import torch
from soundstream import SoundStream
from trainer import SoundStreamTrainer

soundstream = SoundStream( #encoder decoder only
    emb_size=256,
    target_sample_hz=24000,
    strides=(2,4,5,8),
)

#soundstream.load("./results/soundstream.gaussian14.pt")

# '/data2/nchatz/Documents/thesis/data/development'
trainer = SoundStreamTrainer(
    soundstream,
    folder = '/home/nhatz/Documents/University/thesis/code/data/TAU-urban-acoustic-scenes-2022-mobile-development/test',
    batch_size = 8,
    grad_accum_every = 1,       # effective batch size of 8
    data_max_length_seconds = 1,
    num_epochs=1,
    use_mask=True,
)

#torch.autograd.set_detect_anomaly(True)
trainer.train()