from math import sqrt
import copy
from random import choice
from pathlib import Path
from shutil import rmtree

from tqdm.auto import tqdm
import random

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

import matplotlib.pyplot as plt

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
        force_clear_prev_results = False,  # set to True | False to skip the prompt
        num_epochs = 1,
        use_mask = False,
        use_mask_sparse=False,
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

        if torch.cuda.is_available():
            self.print("Running on: "+torch.cuda.get_device_name(torch.cuda.current_device))

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

    def mask_waveform_continuous(self,wave,pct=0.3):
        mask_size = int(pct*wave.size(1))
        for b in range(wave.size(0)):
            mask = torch.zeros(mask_size)
            mask_index = random.randint(0,wave.size(1)-mask_size-1)
            if torch.cuda.is_available():
                mask = mask.cuda()
            wave[b,mask_index:mask_index+mask_size] = mask
        return wave
    
    def mask_waveform_sparse(self,wave,pct=0.3):
        mask_size = int(pct*wave.size(1))

    def train_step(self,shorten=None,loss_fn = nn.L1Loss()):
        device = self.device
        
        self.soundstream.train()
        train_loss = 0
        
        for i, batch in enumerate(tqdm(self.dl,desc="Train batches",position=1,leave=True)):
            wave, = batch    
            wave = wave.to(device)

            if shorten:
                new_wavesize = int(shorten*wave.shape[1])
                short_wave = torch.zeros(wave.shape[0],new_wavesize)
                for b in range(wave.shape[0]):
                    index = random.randint(int(0.4*wave.shape[1]),int(0.6*wave.shape[1]))
                    short_wave[b,:] = wave[b,index:index+new_wavesize]
                wave = short_wave.clone()

            rec_wave = self.soundstream(wave, use_mask_sparse=True,mask_pct=0.05).squeeze(1)

            loss = 100*loss_fn(rec_wave,wave)

            train_loss += loss.item()
            
            self.accelerator.backward(loss / self.grad_accum_every)

            if ((i+1) % self.grad_accum_every == 0) or (i+1 == len(self.dl)):
                
                if exists(self.max_grad_norm):
                    self.accelerator.clip_grad_norm_(self.soundstream.parameters(), self.max_grad_norm)

                self.optim.step()
                self.optim.zero_grad()

        return train_loss / len(self.dl)
        
    def test_step(self,shorten=None,loss_fn = nn.L1Loss()):
        device = self.device
        self.soundstream.eval()
        test_loss = 0
        
        with torch.inference_mode():
            for i,batch in enumerate(tqdm(self.valid_dl,desc="Test batches",position=2,leave=True)):
                wave, = batch
                wave = wave.to(device)

                if shorten:
                    new_wavesize = int(shorten*wave.shape[1])
                    short_wave = torch.zeros(wave.shape[0],new_wavesize)
                    for b in range(wave.shape[0]):
                        index = random.randint(int(0.4*wave.shape[1]),int(0.6*wave.shape[1]))
                        short_wave[b,:] = wave[b,index:index+new_wavesize]
                    wave = short_wave.clone()

                rec_wave = self.soundstream(wave, use_mask_sparse=True,mask_pct=0.05).squeeze(1)
                loss = 100*loss_fn(rec_wave,wave)
                test_loss += loss.item()

        return test_loss / len(self.valid_dl)
    
    def train(self,shorten=None):
        best_test_loss = float('inf')
        train_losses = []
        test_losses = []
        #early_stopping = EarlyStopping(tolerance=5, min_delta=0.001)
        for epoch in tqdm(range(self.epochs),desc="Epochs",position=0):
            train_loss = self.train_step(shorten=shorten)
            test_loss = self.test_step(shorten=shorten)
            
            self.print(" ")
            self.print(
              f"Epoch: {epoch+1} | "
              f"train_loss: {train_loss:.4f} | "
              f"test_loss: {test_loss:.4f} | "
            )

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            
            #save best model
            self.accelerator.wait_for_everyone()
            if self.is_main:
                model_path = str(self.results_folder / f'model1.50.short.curr.pt')
                self.save(model_path)

                if test_loss<best_test_loss:
                    best_test_loss = test_loss
                    model_path = str(self.results_folder / f'model1.50.short.best.pt')
                    self.save(model_path)
                    self.print(f'{epoch+1}: saving model to {str(self.results_folder)}')
            self.print(" ")
            #early stopping
            '''
            early_stopping(train_loss,test_loss)
            if early_stopping.early_stop:
                self.print(f'stopping at epoch {epoch+1}')
                break
            '''
        #plt.plot(train_losses,label="train loss")
        #plt.plot(test_losses,label="test loss")
        #plt.legend(loc="upper right")
        #plt.show()
        self.print('training complete')
    
    def train2(self):
        best_test_loss = float('inf')
        #early_stopping = EarlyStopping(tolerance=5, min_delta=0.001)
        for epoch in tqdm(range(self.epochs),desc="Epochs",position=0):
            train_loss = self.train_step2()
            test_loss = self.test_step2()
            
            self.print(
              f"Epoch: {epoch+1} | "
              f"train_loss: {train_loss:.4f} | "
              f"test_loss: {test_loss:.4f} | "
            )
            
            #save best model
            self.accelerator.wait_for_everyone()
            if self.is_main:
                model_path = str(self.results_folder / f'model2.curr.pt')
                self.save(model_path)

                if test_loss<best_test_loss:
                    best_test_loss = test_loss
                    model_path = str(self.results_folder / f'model2.best.pt')
                    self.save(model_path)
                    self.print(f'{epoch+1}: saving model to {str(self.results_folder)}')
            self.print(" ")
            #early stopping
            '''
            early_stopping(train_loss,test_loss)
            if early_stopping.early_stop:
                self.print(f'stopping at epoch {epoch+1}')
                break
            '''
        
        self.print('training complete')
    
    def train_step2(self,loss_fn = nn.L1Loss()):
        device = self.device
        
        self.soundstream.train()
        train_loss = 0
        
        for i, batch in enumerate(tqdm(self.dl,desc="Train batches",position=1,leave=True)):
            wave, = batch    
            wave = wave.to(device)

            masked_wave = wave.clone()
            masked_wave = self.mask_waveform_continuous(masked_wave)

            rec_wave = self.soundstream(masked_wave, use_mask=False).squeeze(1)

            loss = 100*loss_fn(rec_wave,wave)

            train_loss += loss.item()
            
            self.accelerator.backward(loss / self.grad_accum_every)

            if ((i+1) % self.grad_accum_every == 0) or (i+1 == len(self.dl)):
                
                if exists(self.max_grad_norm):
                    self.accelerator.clip_grad_norm_(self.soundstream.parameters(), self.max_grad_norm)

                self.optim.step()
                self.optim.zero_grad()

        return train_loss / len(self.dl)
        
    def test_step2(self,loss_fn = nn.L1Loss()):
        device = self.device
        self.soundstream.eval()
        test_loss = 0
        
        with torch.inference_mode():
            for i,batch in enumerate(tqdm(self.valid_dl,desc="Test batches",position=2,leave=True)):
                wave, = batch
                wave = wave.to(device)
                masked_wave = wave.clone()
                masked_wave = self.mask_waveform_continuous(masked_wave)
                rec_wave = self.soundstream(masked_wave, use_mask=False).squeeze(1)
                loss = 100*loss_fn(rec_wave,wave)
                test_loss += loss.item()

        return test_loss / len(self.valid_dl)