from math import sqrt
import copy
from random import choice
from pathlib import Path
from shutil import rmtree

from tqdm.auto import tqdm

from beartype.typing import Union, List, Optional, Tuple
from typing_extensions import Annotated

from beartype import beartype
from beartype.door import is_bearable
from beartype.vale import Is

import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from einops import rearrange

from optimizer import get_optimizer

from soundstream import SoundStream

from data import SoundDataset, get_dataloader

from accelerate import Accelerator, DistributedType
from accelerate.utils import DistributedDataParallelKwargs

from early_stopping import EarlyStopping

# helpers

def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

# auto data to module keyword argument routing functions

def has_duplicates(tup):
    counts = dict()
    for el in tup:
        if el not in counts:
            counts[el] = 0
        counts[el] += 1
    return any(filter(lambda count: count > 1, counts.values()))

def determine_types(data, config):
    output = []
    for el in data:
        for name, data_type in config.items():
            if is_bearable(el, data_type):
                output.append(name)
                break
        else:
            raise TypeError(f'unable to determine type of {data}')

    return tuple(output)

# sound stream trainer

class SoundStreamTrainer(nn.Module):
    def __init__(
        self,
        soundstream: SoundStream,
        *,
        batch_size,
        data_max_length = None,
        data_max_length_seconds = None,
        folder,
        lr = 2e-4,
        grad_accum_every = 4,
        wd = 0.,
        max_grad_norm = 0.5,
        discr_max_grad_norm = None,
        save_results_every = 100,
        save_model_every = 1000,
        log_losses_every = 1,
        results_folder = './results',
        valid_frac = 0.05,
        random_split_seed = 42,
        apply_grad_penalty_every = 4,
        dl_num_workers = 0,
        accelerate_kwargs: dict = dict(),
        force_clear_prev_results = None,  # set to True | False to skip the prompt
        num_epochs = 1,
        use_mask = False
    ):
        super().__init__()

        kwargs = DistributedDataParallelKwargs(find_unused_parameters = True)
        self.accelerator = Accelerator(kwargs_handlers = [kwargs], **accelerate_kwargs)

        self.soundstream = soundstream

        self.use_mask = use_mask

        self.register_buffer('steps', torch.Tensor([0]))
        self.batch_size = batch_size
        self.grad_accum_every = grad_accum_every
        self.epochs = num_epochs

        # optimizers

        self.optim = get_optimizer(soundstream.non_discr_parameters(), lr = lr, wd = wd)

        # max grad norm

        self.max_grad_norm = max_grad_norm
        self.discr_max_grad_norm = discr_max_grad_norm

        # create dataset

        assert not (exists(data_max_length) and exists(data_max_length_seconds))

        if exists(data_max_length_seconds):
            data_max_length = data_max_length_seconds * soundstream.target_sample_hz

        self.ds = SoundDataset(
            folder,
            max_length = data_max_length,
            target_sample_hz = soundstream.target_sample_hz,
            seq_len_multiple_of = soundstream.seq_len_multiple_of
        )

        # split for validation

        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # dataloader

        self.dl = get_dataloader(self.ds, batch_size = batch_size, num_workers = dl_num_workers, shuffle = True)

        self.valid_dl = get_dataloader(self.valid_ds, batch_size = batch_size, num_workers = dl_num_workers, shuffle = True)

        # prepare with accelerator
        
        (
            self.soundstream,
            self.optim,
            self.dl
        ) = self.accelerator.prepare(
            self.soundstream,
            self.optim,
            self.dl
        )

        # prepare the multiscale discriminators with accelerator

        # dataloader iterators

        self.dl_iter = cycle(self.dl)
        self.valid_dl_iter = cycle(self.valid_dl)

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every
        self.log_losses_every = log_losses_every

        self.apply_grad_penalty_every = apply_grad_penalty_every

        self.results_folder = Path(results_folder)

        if self.is_main and force_clear_prev_results is True or (not exists(force_clear_prev_results) and len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?')):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents = True, exist_ok = True)

        hps = {"num_epochs": num_epochs, "data_max_length": data_max_length, "learning_rate": lr}
        self.accelerator.init_trackers("soundstream", config=hps)        

    def save(self, path):
        pkg = dict(
            model = self.accelerator.get_state_dict(self.soundstream),
            optim = self.optim.state_dict()
        )

        torch.save(pkg, path)

    @property
    def unwrapped_soundstream(self):
        return self.accelerator.unwrap_model(self.soundstream)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')

        # if loading from old version, make a hacky guess

        if len(pkg.keys()) > 20:
            self.unwrapped_soundstream.load_state_dict(pkg)
            return

        # otherwise load things normally

        self.unwrapped_soundstream.load_state_dict(pkg['model'])

        self.optim.load_state_dict(pkg['optim'])

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def train_step(self,loss_fn = nn.L1Loss()):
        device = self.device
        
        self.soundstream.train()
        train_loss = 0
        
        for i, batch in enumerate(self.dl):
            wave, = batch    
            wave = wave.to(device)
            rec_wave = self.soundstream(wave, use_mask=self.use_mask).squeeze(1)

            loss = loss_fn(rec_wave,wave)

            train_loss += loss.item()
            
            self.accelerator.backward(loss / self.grad_accum_every)

            if ((i+1) % self.grad_accum_every == 0) or (i+1 == len(self.dl)):
                
                if exists(self.max_grad_norm):
                    self.accelerator.clip_grad_norm_(self.soundstream.parameters(), self.max_grad_norm)

                self.optim.step()
                self.optim.zero_grad()

        return train_loss / len(self.dl)
        
    def test_step(self,loss_fn = nn.L1Loss()):
        device = self.device
        self.soundstream.eval()
        test_loss = 0
        
        with torch.inference_mode():
            for i,batch in enumerate(self.valid_dl):
                wave, = batch
                wave = wave.to(device)
                rec_wave = self.soundstream(wave, use_mask=self.use_mask).squeeze(1)
                loss = loss_fn(rec_wave,wave)
                test_loss += loss.item()

        return test_loss / len(self.valid_dl)
    
    def train(self):
        best_test_loss = float('inf')
        #early_stopping = EarlyStopping(tolerance=5, min_delta=0.001)
        for epoch in tqdm(range(self.epochs)):
            train_loss = self.train_step()
            test_loss = self.test_step()
            
            self.print(
              f"Epoch: {epoch+1} | "
              f"train_loss: {train_loss:.4f} | "
              f"test_loss: {test_loss:.4f} | "
            )
            
            #save best model
            self.accelerator.wait_for_everyone()
            if self.is_main:
                model_path = str(self.results_folder / f'soundstream.curr.pt')
                self.save(model_path)

                if test_loss<best_test_loss:
                    best_test_loss = test_loss
                    model_path = str(self.results_folder / f'soundstream.best.pt')
                    self.save(model_path)
                    self.print(f'{epoch+1}: saving model to {str(self.results_folder)}')
            
            #early stopping
            '''
            early_stopping(train_loss,test_loss)
            if early_stopping.early_stop:
                self.print(f'stopping at epoch {epoch+1}')
                break
            '''
        
        self.print('training complete')	