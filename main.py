import torch
import random
from soundstream import SoundStream
from trainer import SoundStreamTrainer

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

#soundstream.load("./results/model1.curr.pt")

soundstream = SoundStream( #encoder decoder only
    emb_size=256,
    target_sample_hz=24000,
    strides=(2,4,5,8),
)

torch.cuda.set_device(0)

# '/data2/nchatz/Documents/thesis/data/development'
trainer = SoundStreamTrainer(
    soundstream,
    folder = '/data2/nchatz/Documents/thesis/data/development',
    batch_size = 48,
    grad_accum_every = 1,       # effective batch size of 48
    data_max_length_seconds = 1,
    num_epochs=100,
    valid_frac=0.2,
    use_mask=True
).cuda()

trainer.train()