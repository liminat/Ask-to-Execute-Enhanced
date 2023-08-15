from email.policy import default
import sys, os, nltk, pickle, argparse, gzip, csv, json, torch, numpy as np, torch.nn as nn
from collections import defaultdict

sys.path.append('..')
from utils import Logger, get_logfiles, tokenize, architect_prefix, builder_prefix, type2id, initialize_rngs, write_commit_hashes

class Vocabulary(object):
	"""Simple vocabulary wrapper."""
	def __init__(self, data_path='../data/logs/', vector_filename=None, embed_size=300, use_speaker_tokens=False, use_builder_action_tokens=False, add_words=True, lower=False, threshold=0, all_splits=False, add_builder_utterances=False, builder_utterances_only=False):
		"""
		Args:
			data_path (string): path to CwC official data directory.
			vector_filename (string, optional): path to pretrained embeddings file.
			embed_size (int, optional): size of word embeddings.
			use_speaker_tokens (boolean, optional): use speaker tokens <Architect> </Architect> and <Builder> </Builder> instead of sentence tokens <s> and </s>
			use_builder_action_tokens (boolean, optional): use builder action tokens for pickup/putdown actions, e.g. <builder_pickup_red> and <builder_putdown_red>
			add_words (boolean, optional): whether or not to add OOV words to vocabulary as random vectors. If not, all OOV tokens are treated as unk.
			lower (boolean, optional): whether or not to lowercase all tokens.
			keep_all_embeddings (boolean, optional): whether or not to keep embeddings in pretrained files for all words (even those out-of-domain). Significantly reduces file size, memory usage, and processing time if set.
			threshold (int, optional): rare word threshold (for the training set), below which tokens are treated as unk.
			add_builder_utterances (boolean, optional): whether or not to obtain examples for builder utterances as well
			builder_utterances_only (boolean, optional): whether or not to only include builder utterances
		"""
		# do not freeze embeddings if we are training our own
		self.data_path = data_path
		self.vector_filename = vector_filename
		self.embed_size = embed_size
		self.freeze_embeddings = vector_filename is not None
		self.use_speaker_tokens = use_speaker_tokens
		self.use_builder_action_tokens = use_builder_action_tokens
		self.add_words = add_words
		self.lower = lower
		self.threshold = threshold
		self.all_splits = all_splits
		self.add_builder_utterances = add_builder_utterances
		self.builder_utterances_only = builder_utterances_only

		print("Building vocabulary.\n\tdata path:", self.data_path, "\n\tembeddings file:", self.vector_filename, "\n\tembedding size:", self.embed_size,
			"\n\tuse speaker tokens:", self.use_speaker_tokens, "\n\tuse builder action tokens:", self.use_builder_action_tokens, "\n\tadd