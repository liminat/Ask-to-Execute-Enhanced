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

		By dataset, we mean self.samples and self.jsons -- the former being actual train/test examples, the latter being the json log files used to obtain these samples

		"""

		self.split = split
		self.lower = lower
		self.compute_perspective = compute_perspective
		self.add_augmented_data = add_augmented_data

		self.encoder_vocab = encoder_vocab # is a Class Vocabulary, word2id is __call__ method 

		self.num_prev_utterances = 1
		self.include_empty_channel = False

		self.aug_sampling_strict = aug_sampling_strict

		cwc_datasets_path = save_dest_dir

		lower_str = "lower" if self.lower else ""
		pers_str = '-no_perspective_coords' if not self.compute_perspective else ""
		aug_str = "-augmented" if self.add_augmented_data else ""

		if load_dataset:
			dataset_dir = saved_dataset_dir

			print("Loading dataset ...\n")

			print("Loading self.samples ...")
			self.samples = load_pkl_data(dataset_dir + "/"+ self.split + "-samples.pkl")

			print("Loading self.jsons ...")
			self.jsons = load_pkl_data(dataset_dir + "/"+ self.split + "-jsons.pkl")

			print("Done! Loaded dataset of size", len(self.samples))

		else:
			self.jsons = list(
				map(
					remove_empty_states,
					map(
						reorder,
						get_logfiles_with_gold_config(data_dir, gold_configs_dir, split)
					)
				)
			) # TODO: Move the extra maps to a postprocesing step for the dataset?

			if self.add_augmented_data:
				print(timestamp(), "Adding augmented dataset...")

				def reformat_utterances(aug_observations_json):
					"""
						Joins tokens back with a space
					"""
					for world_state in aug_observations_json["WorldStates"]:
						world_state["ChatHistoryTokenized"] = list(map(
							lambda x: " ".join(x), world_state["ChatHistoryTokenized"]
						))
						world_state["ChatHistory"] = world_state.pop("ChatHistoryTokenized")

					return aug_observations_json

				self.jsons += list(
					map(
						remove_empty_states,
						map(
							reorder,
							map(
								reformat_utterances,
								get_logfiles_with_gold_config(aug_data_dir, aug_gold_configs_dir, split, from_aug_data=True)
							)
						)
					)
				)

			print(timestamp(), 'Started processing jsons to get samples...')
			self.samples = self.process_samples(lower, compute_perspective=self.compute_perspective)
			print(timestamp(), 'Done processing jsons to get samples.')

			if self.add_augmented_data:
				samples_split = {'orig': [], 'aug': []}
				for sample in self.samples:
					samples_split['orig'].append(sample) if not sample.get('from_aug_data') else samples_split['