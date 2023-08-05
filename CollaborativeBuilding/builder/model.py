import math
import torch, sys
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from builder.dataloader_with_glove import BuilderDataset, RawInputs

sys.path.append('..')
from utils import *

color2id = {
    "orange": 0,
    "red": 1,
    "green": 2,
    "blue": 3,
    "purple": 4,
    "yellow": 5
}
id2color = {v:k for k, v in color2id.items()}


class Builder(nn.Module):
    def __init__(self, con