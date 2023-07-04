import tqdm
import sys, torch, json, copy, pickle, re, os, numpy as np, pprint as pp, cProfile, pstats, io, traceback, itertools, random
sys.path.append('..')
from builder.diff import diff, get_diff, build_region_specs, dict_to_tuple, is_feasible_next_placement

from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, Counter
from operator import itemgetter
from utils import *
from builder.vocab import Vocabulary


class CwCDataset(Dataset):
	""" CwC Dataset compatible with torch.utils.data.DataLoader. """

	def __init__(
		self, split, lower=False, compute_perspective=True,
		data_dir="../../data/logs/", gold_configs_dir="../../data/gold-configurations/", save_dest_dir="../builder_data_with_glove/", saved_dataset_dir="../builder_data_with_glove/", vocab_dir="../../data/vocabulary/",
		encoder_vocab=None, dump_dataset=False, load_dataset=False,
		add_augmented_data=False, aug_data_dir="../../data/augmented/logs/", aug_gold_configs_dir="../../data/augmented/gold-configurations/",
        aug_sampling_strict=False
	):
		"""
		Instantiates a dataset
			- If dump_dataset and load_dataset are both un-set, generates the dataset
			- If dump_dataset is set, also writes the generated dataset to file
			- If load_dataset is set, loads an existing dataset instead of generating (needed most often)

		By dataset, we mean self.samples and self.jsons -- the former being actual train/test examples, the latter being the json log files used