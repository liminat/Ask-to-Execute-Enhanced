from enum import unique
import tqdm
import sys, torch, json, copy, pickle, re, os, numpy as np, pprint as pp, cProfile, pstats, io, traceback, itertools, random
sys.path.append('..')
from builder.diff import diff, get_diff, build_region_specs, dict_to_tuple, is_feasible_next_placement

from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, Counter
from operator import itemgetter
from utils import *
from builder.utils_builder import is_a_id, split_line

# MAIN CLASSES

class CwCDataset(Dataset):
	""" CwC Dataset compatible with torch.utils.data.DataLoader. """

	def __init__(
		self, split, compute_perspective=True,
		data_dir="../../data/logs/", gold_configs_dir="../../data/gold-configurations/", save_dest_dir="../builder_with_questions_data", saved_dataset_dir="../builder_with_questions_data",
		dump_dataset=False, load_dataset=False,
		add_augmented_data=False, aug_data_dir="../../data/augmented/logs/", aug_gold_configs_dir="../../data/augmented/gold-configurations/",
        aug_sampling_strict=False, lower=False
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
					samples_split['orig'].append(sample) if not sample.get('from_aug_data') else samples_split['aug'].append(sample)
				print('\nOriginal dataset contains', len(samples_split['orig']), 'original samples and', len(samples_split['aug']), 'augmented samples ('+str(len(samples_split['orig'])+len(samples_split['aug'])), 'total samples).')

				augmented_data_fractions = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]
				augmented_data_fractions = list(map(lambda x: x/2, augmented_data_fractions))
				mixed_data_size = ["2x", "4x", "6x", "8x", "10x", "12x", "14x", "16x", "18x", "20x"]
				num_aug_samples_per_orig = list(range(1, 20, 2))

				frac2size = dict(zip(augmented_data_fractions, mixed_data_size))
				frac2data = {}

				if self.aug_sampling_strict:
					grouped_aug_samples, _ = group_samples_by_id(samples_split['aug'])

				for frac, num_samples in zip(augmented_data_fractions, num_aug_samples_per_orig):
					if not self.aug_sampling_strict:
						print('Filtering augmented samples with a fraction of', frac, '...')
						chosen_aug_samples = list(np.random.choice(samples_split['aug'], int(frac*len(samples_split['aug'])), replace=False))
					else:
						print('Filtering augmented samples per group with a num_samples of', num_samples, '...')
						chosen_aug_samples = sample_strictly(grouped_aug_samples, num_samples)

					print('Randomly sampled', len(chosen_aug_samples), 'augmented samples from the full augmented set.')

					mixed_samples = samples_split['orig'] + chosen_aug_samples
					frac2data[frac] = mixed_samples

			print("Current dataset size", len(self.samples))
			print("Done! Loaded vanilla dataset of size", len(self.samples))

			if dump_dataset:
				if self.add_augmented_data:
					for frac, data in frac2data.items():
						# FIXME: Resuse code
						# Generate semi-unique file path based on inputs
						aug_frac_str = "-" + frac2size[frac]
						dataset_dir = lower_str + pers_str + aug_str + aug_frac_str
						dataset_dir = os.path.join(cwc_datasets_path, dataset_dir)

						if not os.path.exists(dataset_dir):
							os.makedirs(dataset_dir)

						print("Saving dataset ...\n")

						print("Saving self.jsons ...") # NOTE: These are ALL vanilla + aug jsons -- does not correspond to the ones used in samples only
						save_pkl_data(dataset_dir + "/"+ self.split + "-jsons.pkl", self.jsons)

						print("Saving self.samples ...")
						save_pkl_data(dataset_dir + "/"+ self.split + "-samples.pkl", data)

						# write which aug dir used
						with open(os.path.join(dataset_dir, "aug_data_dir.txt"), 'w') as f:
							f.write(os.path.abspath(aug_data_dir))

				else:
					# Generate semi-unique file path based on inputs

					print("Saving dataset ...\n")

					print("Saving self.jsons ...")
					if not os.path.exists(save_dest_dir):
						os.makedirs(save_dest_dir)
					save_pkl_data(save_dest_dir + "/"+ self.split + "-jsons.pkl", self.jsons)

					print("Saving self.samples ...")
					save_pkl_data(save_dest_dir + "/"+ self.split + "-samples.pkl", self.samples)

		self.augmentation_factor = 0

	def get_sample(self, idx):
		""" Returns one data sample (utterance) in tokenized form. """
		return self.samples[idx]

	def process_samples(self, lower, compute_diff=True, compute_perspective=True):
		""" Preprocesses the input JSONs and generates a list of data samples. """
		samples = []

		try:
			for j in tqdm.tqdm(range(len(self.jsons))):
                ## This is to preprocess each dialogue example, len(train_jsons),len(val_jsons),len(test_jsons) = (281, 101, 137)

				try:
					js = self.jsons[j]
					unique_id = js['log_dir']
					builder_utterance_labels_unique_id = builder_utterance_labels[unique_id]

					if js["from_aug_data"]:
						orig_log_dir = re.sub(r"_\d+", "", js["log_dir"])
					else:
						orig_log_dir = js["log_dir"]

					# print(js["logfile_path"] + "\n")
					world_states = js["WorldStates"]
					# print(world_states[0])
					# sys.exit(0)
					final_observation = world_states[-1]
					gold_config = js["gold_config_structure"]

					last_world_state = None
					chat_history = []
					chat_with_actions_history = []

					## this for loop is to construct chat_with_actions_history (zshi)
					## each element in chat_with_actions_history will be a sample
					for i in range(1, len(world_states)):
						if i > 1:
							action_history = world_states[i - 1]["ActionHistory"]
						else:
							action_history = []
						observation = world_states[i]
						observation["ActionHistory"] = action_history
						built_config = get_built_config(observation)
						builder_position = get_builder_position(observation)
						prev_builder_position = get_builder_position(world_states[i-1])
						last_action = None

						for k, curr_world_state in enumerate(reversed(world_states[:i+1])):
							original_index = i-k

							# compare blocks with its prev world state
							curr_blocks = curr_world_state["BlocksInGrid"]
							prev_blocks = [] if original_index == 0 else world_states[original_index-1]["BlocksInGrid"]
							last_action = get_last_action(curr_blocks, prev_blocks)

							if last_action:
								break

						if not last_world_state:
							for i2 in range(len(observation["ChatHistory"])):
								chat_history.append(observation["ChatHistory"][i2].strip())

								for block in built_config:
									chat_with_actions_history.append({"idx": i, "action": "putdown", "type": block["type"], "x": block["x"], "y": block["y"], "z": block["z"], "built_config": built_config, "prev_config": None, "builder_position": builder_position, "prev_builder_position": prev_builder_position, "last_action": last_action})

								chat_with_actions_history.append({"idx": i, "action": "chat", "utterance": observation["ChatHistory"][i2].strip(), "built_config": built_config, "prev_config": None, "builder_position": builder_position, "prev_builder_position": prev_builder_position, "last_action": last_action})

						else:
							prev_config = get_built_config(last_world_state)
							config_diff = diff(gold_config=built_config, built_config=prev_config)

							config_diff["gold_minus_built"] = sorted(config_diff["gold_minus_built"], key=itemgetter('x', 'y', 'z', 'type'))
							config_diff["built_minus_gold"] = sorted(config_diff["built_minus_gold"], key=itemgetter('x', 'y', 'z', 'type'))

							delta = {"putdown": config_diff["gold_minus_built"], "pickup": config_diff["built_minus_gold"]}

							for action_type in delta:
								for block in delta[action_type]:
									chat_with_actions_history.append({"idx": i, "action": action_type, "type": block["type"], "x": block["x"], "y": block["y"], "z": block["z"], "built_config": built_config, "prev_config": prev_config, "builder_position": builder_position, "prev_builder_position": prev_builder_position, "last_action": last_action})

							if len(observation["ChatHistory"]) > len(last_world_state["ChatHistory"]):
								for i3 in range(len(last_world_state["ChatHistory"]), len(observation["ChatHistory"])):
									chat_history.append(observation["ChatHistory"][i3].strip())
									chat_with_actions_history.append({"idx": i, "action": "chat", "utterance": observation["ChatHistory"][i3].strip(), "built_config": built_config, "prev_config": prev_config, "builder_position": builder_position, "prev_builder_position": prev_builder_position, "last_action": last_action})

						last_world_state = observation
                    

					# process dialogue line-by-line

					assert observation["ActionHistory"] == []

					for i in range(len(chat_with_actions_history)):
						elem = chat_with_actions_history[i]
						weight = None
						if elem['action'] == 'chat':
							
							utterance = elem["utterance"]
							idx = elem['idx']
							built_config = elem["built_config"]
							prev_config = elem["prev_config"] 

							prev_builder_position = elem["prev_builder_position"]

							if utterance.startswith('<Architect>') and i < len(chat_with_actions_history) - 1: ## i == len(chat_with_actions_history) - 1 means finished
								next_elem = chat_with_actions_history[i+1]
								if next_elem['action'] == 'chat' and next_elem["utterance"].startswith('<Builder>'):

									builder_utterance = next_elem["utterance"]

									prev_utterances = []
									prev_utterances.append({'speaker': 'Builder', 'utterance': ['<dialogue>']})

									for k in range(i):
										prev_elem = chat_with_actions_history[k]

										if prev_elem['action'] != 'chat': ## Action 
											prev_utterances.append({'speaker': 'Builder', 'utterance': ['<builder_'+prev_elem['action']+'_'+prev_elem['type']+'>']})

										else:  ## Utterance 
											prev_utterance = prev_elem['utterance']
											prev_speaker = "Architect" if "Architect" in prev_utterance.split()[0] else "Builder"
											prev_utterance = prev_utterance[len(architect_prefix):] if prev_speaker == 'Architect' else prev_utterance[len(builder_prefix):]
											prev_tokenized, _ = tokenize(prev_utterance.lower() if lower else prev_utterance) ## (zshi)			
											prev_utterances.append({'speaker': prev_speaker, 'utterance': prev_tokenized})
                                    
									perspective_coordinates = None if not compute_perspective else torch.Tensor(get_perspective_coord_repr(prev_builder_position))

									add_dict = {}
									add_dict.update({'builder_action_history': observation["ActionHistory"]})
									add_dict.update(
										{
											'builder_action': builder_utterance_labels_unique_id[builder_utterance.lower()],
											'prev_utterances': prev_utterances, # previous utterances
											'gold_config': gold_config,
											'built_config': built_config,
											'prev_config': prev_config if prev_config else [], # NOTE: CRITICAL FOR PREDICTING BUILDER ACTIONS -- INPUT TO MODEL SHOULD BE THIS AND NOT BUILT CONFIG
											'prev_builder_position': prev_builder_position,
											'perspective_coordinates': perspective_coordinates,
											'from_aug_data': js['from_aug_data'],
											'json_id': j, # ID of the json this sample was obtained from
											'sample_id': idx, # ID assigned to this sample,
											'orig_experiment_id': orig_log_dir # original log dir/experiment ID (for augmented data -- this is the same as the original counterpart)
										}
									)

									samples.append(add_dict)
						
						else:
							if elem['action'] != 'chat':
								if elem['action'] == "putdown":
									weight = 1
								if elem['action'] == "pickup":
									weight = 0

								next_builder_action = BuilderAction(elem["x"], elem["y"], elem["z"], elem["type"], elem["action"], weight)
								#pp.PrettyPrinter(indent=4).pprint(next_builder_action.weight)
								observation["ActionHistory"].append(next_builder_action)

							def get_all_next_actions(start_index):
								all_next_actions = []
								# slicing below to create copies and avoid effects of mutation as observation["ActionHistory"] is updated
								action_history = observation["ActionHistory"][:-1]

								for zzz in range(start_index, len(chat_with_actions_history)):
									next_elem = chat_with_actions_history[zzz]
									if next_elem['action'] == 'chat':
										break
									else:
										weight = None
										all_next_actions.append(
											BuilderActionExample(
												action = BuilderAction(next_elem["x"], next_elem["y"], next_elem["z"], next_elem["type"], next_elem["action"], weight),
												built_config = next_elem["built_config"],
												prev_config = next_elem["prev_config"] if next_elem["prev_config"] else [],
												action_history = action_history
											)
										)
										action_history = action_history + [(
											BuilderAction(next_elem["x"], next_elem["y"], next_elem["z"], next_elem["type"], next_elem["action"], weight)
										)] # NOTE: concatenation to avoid mutation

								return all_next_actions

							if i > 0:
								prev_elem = chat_with_actions_history[i-1]
								if not prev_elem['action'] == 'chat':
									continue
								
							all_next_actions = get_all_next_actions(i)

							idx = elem['idx']
							built_config = elem["built_config"]
							prev_config = elem["prev_config"] ## list of dicts, prev_config is always one elem less than built_config
							prev_builder_position = elem["prev_builder_position"]

							def valid_config(config):
								if not config:
									return True

								for block in config:
									x, y, z = block["x"]-x_min, block["y"]-y_min, block["z"]-z_min
									if x < 0 or x >= x_range or y < 0 or y >= y_range or z < 0 or z >= z_range:
										return False

								return True

							# temporary fix for troublesome configs
							if not valid_config(built_config) or not valid_config(prev_config):
								continue

							skip_sample = False
							for next_action in all_next_actions:
								if not valid_config(next_action.built_co