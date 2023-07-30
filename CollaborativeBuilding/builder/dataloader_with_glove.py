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

    def __getitem__(self, idx):
        return self.items[idx]

    def preprocess(self, idx):
        """ Computes the tensor representations of a sample """
        orig_sample = self.samples[idx]

        all_actions = orig_sample["next_builder_actions"]
        perspective_coords = orig_sample["perspective_coordinates"]

        initial_prev_config_raw = all_actions[0].prev_config
        initial_action_history_raw = all_actions[0].action_history
        end_built_config_raw = all_actions[-1].built_config

        all_actions_reprs = list(map(lambda x: self.get_repr(x, perspective_coords), all_actions))
        all_grid_repr_inputs = list(map(lambda x: x[0], all_actions_reprs))
        all_outputs = list(map(lambda x: x[1], all_actions_reprs))
        all_location_mask = list(map(lambda x: x[2], all_actions_reprs))

        all_actions_repr_inputs = list(map(f2, all_actions))

        start_action_repr = torch.Tensor([0] * 11)
        
        stop_action = BuilderActionExample(
            action = None,
            built_config = all_actions[-1].built_config,
            prev_config = all_actions[-1].built_config,
            action_history = all_actions[-1].action_history + [all_actions[-1].action]
        )
        stop_action_repr = self.get_repr(stop_action, perspective_coords)
        stop_action_grid_repr_input = stop_action_repr[0]
        stop_action_output_label = stop_action_repr[1]
        stop_action_location_mask = stop_action_repr[2]

        dec_inputs_1 = all_grid_repr_inputs + [stop_action_grid_repr_input]
        dec_inputs_2 = [start_action_repr] + all_actions_repr_inputs
        location_mask = all_location_mask + [stop_action_location_mask]

        dec_outputs = all_outputs + [stop_action_output_label]
        
        # Encoder inputs
        utterances_to_add = []

        '''
        an example of orig_sample["prev_utterances"]:
        
        [{'speaker': 'Builder', 'utterance': ['<dialogue>']}, 
        {'speaker': 'Builder', 'utterance': ['Mission has started.']}, 
        {'speaker': 'Builder', 'utterance': ['hi']}, 
        {'speaker': 'Architect', 'utterance': ["hello, it looks like we are building a cube where the sides are made of the letters 'a', 'b', 'c', 'd'"]}, 
        {'speaker': 'Builder', 'utterance': ['sounds cool']}, 
        {'speaker': 'Architect', 'utterance': ['First, we will need