from torch.functional import Tensor
from torch.utils.data import Dataset, DataLoader
import torch
import argparse
import tqdm
import numpy as np
import pickle
import os, sys
sys.path.append('..')
from utils import type2id, load_pkl_data, x_min, x_max, y_min, y_max, z_min, z_max, BuilderAction, f2, stop_action_label
from builder.data_loader_with_glove import BuilderActionExample, Region
from builder.vocab import Vocabulary
from builder.utils_builder import get_feasibile_location

color2id = {
	"orange": 0,
	"red": 1,
	"green": 2,
	"blue": 3,
	"purple": 4,
	"yellow": 5
}

id2color = {v: k for k, v in type2id.items()}

action2id = {
	"placement": 0,
	"removal": 1
}

class BuilderDataset(Dataset):
    def __init__(self, args, split, encoder_vocab):
        if args.load_items:
            self.dataset_dir = args.json_data_dir
            self.split = split
            self.samples = False
            with open(os.path.join(self.dataset_dir, self.split + "-items.pkl"), 'rb') as f:
                self.items = torch.load(f, map_location="cpu") 
            self.include_empty_channel = True
            self.use_builder_actions = False
            self.add_action_history_weight = True
            self.action_history_weighting_scheme = "step"
            self.concatenate_action_history_weight = True
            self.add_perspective_coords = False
        else:
            self.include_empty_channel = args.include_empty_channel
            self.use_builder_actions = args.use_builder_actions
            self.add_action_history_weight = args.add_action_history_weight
            self.action_history_weighting_scheme = args.action_history_weighting_scheme
            self.concatenate_action_history_weight = args.concatenate_action_history_weight
            self.num_prev_utterances = args.num_prev_utterances
            self.use_builder_actions = args.use_builder_actions
            self.add_perspective_coords = args.add_perspective_coords
            self.encoder_vocab = encoder_vocab

            self.split = split
            self.max_length = args.max_length
            self.dataset_dir = args.json_data_dir
            self.samples = load_pkl_data(self.dataset_dir + "/"+ self.split + "-samples.pkl")
            self.items = []
            for i in tqdm.tqdm(range(len(self))):
                self.items.append(self.preprocess(i))
            torch.save(self.items, os.path.join(self.dataset_dir, self.split + "-items.pkl")) 
            print('{} data has been stored at {}'.format(self.split, os.path.join(self.dataset_dir, self.split + "-items.pkl")))

    def __len__(self):
        if self.samples:
            return len(self.samples)
        else:
            return len(self.items)

    def __