import functools
from itertools import cycle
from functools import partial
from pathlib import Path
import random

import torch
from torch import nn
import torch.nn.functional as F

import torchaudio.transforms as T
from torchaudio.functional import resample

from einops import rearrange, reduce, pack, unpack

from utils import curtail_to_multiple

import pickle

from version import __version__
from packaging import version
parsed_version = version.parse(__version__)

# helper functions

def exists(val):
    return val is not None

def filter_by_keys(fn, d):
    return {k: v for k, v in d.items() if fn(k)}

def map_keys(fn, d):
    return {fn(k): v for k, v in d.items()}

# better sequential

def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))

# sound stream

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class CausalConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, **kwargs):
        super().__init__()
        kernel_size = kernel_size
        dilation = kwargs.get('dilation', 1)
        stride = kwargs.get('stride', 1)
        self.causal_padding = dilation * (kernel_size - 1) + (1 - stride)

        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, **kwargs)

    def forward(self, x):
        x = F.pad(x, (self.causal_padding, 0))#,mode="reflect")
        return self.conv(x)

class CausalConvTranspose1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, stride, **kwargs):
        super().__init__()
        self.upsample_factor = stride
        self.padding = kernel_size - 1
        self.conv = nn.ConvTranspose1d(chan_in, chan_out, kernel_size, stride, **kwargs)

    def forward(self, x):
        n = x.shape[-1]

        out = self.conv(x)
        out = out[..., :(n * self.upsample_factor)]

        return out

def ResidualUnit(chan_in, chan_out, dilation, kernel_size = 7):
    return Residual(Sequential(
        CausalConv1d(chan_in, chan_out, kernel_size, dilation = dilation),
        nn.ELU(),
        CausalConv1d(chan_out, chan_out, 1),
        nn.ELU()
    ))

def EncoderBlock(chan_in, chan_out, stride, cycle_dilations = (1, 3, 9)):
    it = cycle(cycle_dilations)
    residual_unit = partial(ResidualUnit)

    return nn.Sequential(
        residual_unit(chan_in, chan_in, next(it)),
        residual_unit(chan_in, chan_in, next(it)),
        residual_unit(chan_in, chan_in, next(it)),
        CausalConv1d(chan_in, chan_out, 2 * stride, stride = stride)
    )

def DecoderBlock(chan_in, chan_out, stride, cycle_dilations = (1, 3, 9)):
    even_stride = (stride % 2 == 0)
    padding = (stride + (0 if even_stride else 1)) // 2
    output_padding = 0 if even_stride else 1

    residual_unit = partial(ResidualUnit)

    it = cycle(cycle_dilations)
    return nn.Sequential(
        CausalConvTranspose1d(chan_in, chan_out, 2 * stride, stride = stride),
        residual_unit(chan_out, chan_out, next(it)),
        residual_unit(chan_out, chan_out, next(it)),
        residual_unit(chan_out, chan_out, next(it)),
    )

class SoundStream(nn.Module):
    def __init__(
        self,
        *,
        channels = 32,
        strides = (2, 4, 5, 8),
        channel_mults = (2, 4, 8, 16),
        emb_size = 512, #codebook dim
        input_channels = 1,
        enc_cycle_dilations = (1, 3, 9),
        dec_cycle_dilations = (1, 3, 9),
        target_sample_hz = 24000,
        use_disc = True,
        use_rq = True,
    ):
        super().__init__()
        self.use_disc = use_disc
        self.use_rq = use_rq

        # for autosaving the config

        _locals = locals()
        _locals.pop('self', None)
        _locals.pop('__class__', None)
        self._configs = pickle.dumps(_locals)

        # rest of the class

        self.target_sample_hz = target_sample_hz # for resampling on the fly

        self.single_channel = input_channels == 1
        self.strides = strides

        layer_channels = tuple(map(lambda t: t * channels, channel_mults))
        layer_channels = (channels, *layer_channels)
        chan_in_out_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))

        encoder_blocks = []

        for ((chan_in, chan_out), layer_stride) in zip(chan_in_out_pairs, strides):
            encoder_blocks.append(EncoderBlock(chan_in, chan_out, layer_stride, enc_cycle_dilations))

        self.encoder = nn.Sequential(
            CausalConv1d(input_channels, channels, 7),
            *encoder_blocks,
            CausalConv1d(layer_channels[-1], emb_size, 3)
        )
        decoder_blocks = []

        for ((chan_in, chan_out), layer_stride) in zip(reversed(chan_in_out_pairs), reversed(strides)):
            decoder_blocks.append(DecoderBlock(chan_out, chan_in, layer_stride, dec_cycle_dilations))

        self.decoder = nn.Sequential(
            CausalConv1d(emb_size, layer_channels[-1], 7),
            *decoder_blocks,
            CausalConv1d(channels, input_channels, 7)
        )

        self.register_buffer('zero', torch.tensor([0.]), persistent = False)

    @property
    def configs(self):
        return pickle.loads(self._configs)

    def save(self, path):
        path = Path(path)
        pkg = dict(
            model = self.state_dict(),
            config = self._configs,
            version = __version__
        )

        torch.save(pkg, str(path))

    @classmethod
    def init_and_load_from(cls, path, strict = True):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')

        assert 'config' in pkg, 'model configs were not found in this saved checkpoint'

        config = pickle.loads(pkg['config'])
        soundstream = cls(**config)
        soundstream.load(path, strict = strict)
        return soundstream

    def load(self, path, strict = True):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')

        # check version

        if 'version' in pkg and version.parse(pkg['version']) < parsed_version:
            print(f'soundstream model being loaded was trained on an older version of audiolm-pytorch ({pkg["version"]})')

        has_ema = 'ema_model' in pkg
        model_pkg = pkg['ema_model'] if has_ema else pkg['model']

        if has_ema:
            model_pkg = filter_by_keys(lambda k: k.startswith('ema_model.'), model_pkg)
            model_pkg = map_keys(lambda k: k[len('ema_model.'):], model_pkg)

        self.load_state_dict(model_pkg, strict = strict)

    def load_from_trainer_saved_obj(self, path):
        path = Path(path)
        assert path.exists()
        obj = torch.load(str(path))
        self.load_state_dict(obj['model'])

    def non_discr_parameters(self):
        return [
            *self.encoder.parameters(),
            *self.decoder.parameters()
        ]

    @property
    def seq_len_multiple_of(self):
        return functools.reduce(lambda x, y: x * y, self.strides)

    def forward(
        self,
        x,
        return_encoded = False,
        input_sample_hz = None,
        use_mask = False,
        use_mask_sparse=False,
        mask_pct=0.05,
        new_blocks=0,
        #use_gaussian = False,
    ):
        x, ps = pack([x], '* n')

        if exists(input_sample_hz):
            x = resample(x, input_sample_hz, self.target_sample_hz)

        x = curtail_to_multiple(x, self.seq_len_multiple_of)

        if x.ndim == 2:
            x = rearrange(x, 'b n -> b 1 n')

        orig_x = x.clone()

        x = self.encoder(x)

        x = rearrange(x, 'b c n -> b n c')


        if use_mask_sparse:
            #enc(x) = [batch_size,emb_count,emb_size] [32,50,256] [0,1,2]
            mask_number = int(mask_pct*x.size(1))+1
            for b in range(x.size(0)): #for each wave in batch
                u2 = torch.max(x[b,:,:]).clone().detach()
                u1 = torch.min(x[b,:,:]).clone().detach()
                mask_indexes = [random.randint(0,x.size(1)-1) for _ in range(mask_number)]
                #real_ind = [m*480 for m in mask_indexes]
                #print("Mask positions: ",real_ind)
                for mi in mask_indexes:
                    if torch.cuda.is_available():
                        mask = (u2-u1)*torch.rand(1,x.size(2)).cuda()+u1
                    else:
                        mask = (u2-u1)*torch.rand(1,x.size(2))+u1
                    x[b,mi,:] = mask

        if use_mask:
            #enc(x) = [batch_size,emb_count,emb_size] [32,75,256] [0,1,2]
            mask_size = int(mask_pct*x.size(1))
            for b in range(x.size(0)): #for each wave in batch
                u2 = torch.max(x[b,:,:]).clone().detach()
                u1 = torch.min(x[b,:,:]).clone().detach()
                mask_index = random.randint(0,x.size(1)-mask_size-1)
                if torch.cuda.is_available():
                    mask = (u2-u1)*torch.rand(mask_size,x.size(2)).cuda()+u1
                else:
                    mask = (u2-u1)*torch.rand(mask_size,x.size(2))+u1
                x[b,mask_index:mask_index+mask_size,:] = mask
        
        if new_blocks>0:
            '''
            Block size is always rounded down. With strides (3,4,5,8)
            for 1 second at 24kHz we get 50 embeddings. If we use
            mask_pct of 25% this means block size is 12 (12.5 becomes 12).
            So adding 4 blocks does not result in exactly 2 seconds of audio,
            but a bit less.
            '''
            block_size = int(mask_pct*x.size(1))
            ext = torch.zeros(x.size(0),x.size(1)+(new_blocks*block_size),x.size(2)) #preallocate exended wave
            for b in range(x.size(0)): #for each wave in batch
                #c = x[b,:,:].clone()
                u2 = torch.max(x[b,:,:]).clone().detach()
                u1 = torch.min(x[b,:,:]).clone().detach()
                ixs = sorted(random.sample(range(x.size(1)-1),new_blocks)) #random indexes to insert new random blocks
                ixs.append(x.size(1)) #ensure trailing part is added
                c = x[b,:ixs[0],:].clone() #first part
                for ix in range(1,new_blocks+1):
                    block = (u2-u1)*torch.rand(block_size,x.size(2))+u1
                    c = torch.cat([c,block,x[b,ixs[ix-1]:ixs[ix],:]],0) #block and next part
                ext[b,:,:] = c.clone()
            x = ext.clone()

        x = rearrange(x, 'b n c -> b c n')

        if return_encoded:
            return x

        recon_x = self.decoder(x)

        recon_x, = unpack(recon_x, ps, '* c n')
        return recon_x

