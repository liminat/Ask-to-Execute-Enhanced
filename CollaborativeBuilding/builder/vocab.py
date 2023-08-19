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
			"\n\tuse speaker tokens:", self.use_speaker_tokens, "\n\tuse builder action tokens:", self.use_builder_action_tokens, "\n\tadd words:", self.add_words,
			"\n\tlowercase:", self.lower, "\n\trare word threshold:", self.threshold, "\n")

		# mappings from words to IDs and vice versa
		# this is what defines the vocabulary
		self.word2idx = {}
		self.idx2word = {}

		# mapping from words to respective counts in the dataset
		self.word_counts = defaultdict(int)
		# entire dataset in the form of tokenized utterances
		self.tokenized_data = []
		# words that are frequent in the dataset but don't have pre-trained embeddings
		self.oov_words = set()

		# store dataset in tokenized form and it's properties
		self.get_dataset_properties() # self.word_counts and self.tokenized_data populated

		# initialize word vectors
		self.init_vectors() # self.word_vectors, self.word2idx and self.idx2word populated for aux tokens

		# load pretrained word vectors
		if vector_filename is not None:
			self.load_vectors() # self.word_vectors, self.word2idx and self.idx2word populated for real words

		# add random vectors for oov train words -- words that are in data, frequent but do not have a pre-trained embedding
		if add_words or vector_filename is None: ## True
			self.add_oov_vectors()

		# create embedding variable
		self.word_embeddings = nn.Embedding(self.word_vectors.shape[0], self.word_vectors.shape[1])

		# initialize embedding variable
		self.word_embeddings.weight.data.copy_(torch.from_numpy(self.word_vectors))

		# freeze embedding variable
		if self.freeze_embeddings:
			self.word_embeddings.weight.requires_grad = False

		self.num_tokens = len(self.word2idx)
		self.print_vocab_statistics()

	def get_dataset_properties(self):
		jsons = get_logfiles(self.data_path, split='train')
		if self.all_splits:
			print("Using all three of train, val and test data...")
			jsons += get_logfiles(self.data_path, split='val') + get_logfiles(self.data_path, split='test')
		fixed_tokenizations = set()

		# compute word counts for all tokens in training dataset
		for i in range(len(jsons)):
			js = jsons[i]
			final_observation = js["WorldStates"][-1]

			for i in range(1, len(final_observation["ChatHistory"])):
				line = final_observation["ChatHistory"][i]

				speaker = "Architect" if "Architect" in line.split()[0] else "Builder"
				if speaker == "Architect":
					# Skip over architect if only want builder utterances
					if self.builder_utterances_only:
						continue
					else:
						utterance = line[len(architect_prefix):]
				else:
					# Include builder if either flag is on
					if self.add_builder_utterances or self.builder_utterances_only:
						utterance = line[len(builder_prefix):]
					else:
						continue

				if self.lower:
					utterance = utterance.lower()

				tokenized, fixed = tokenize(utterance)
				fixed_tokenizations.update(fixed)
				self.tokenized_data.append(tokenized) ## store all tokens in dataset into tokenized_data (zshi)

				for word in tokenized:
					self.word_counts[word] += 1

		print("\nTokenizations fixed:", fixed_tokenizations, "\n")

	def init_vectors(self):
		embed_size = self.embed_size

		# pad token
		vector_arr = np.zeros((1, embed_size))
		self.add_word('<pad>') # id 0 # vector of 0's

		if not self.vector_filename:
			vector_arr = np.concatenate((vector_arr, np.random.rand(1, embed_size)), 0)
			self.add_word('<unk>')

		# start & end of sentence symbols
		if not self.use_speaker_tokens: ## False (zshi)
			vector_arr = np.concatenate((vector_arr, np.random.rand(2, embed_size)), 0)
			self.add_word('<s>') # id 1 # random vector
			self.add_word('</s>') # id 2 # random vector

		# speaker tokens
		else:
			vector_arr = np.concatenate((vector_arr, np.random.rand(6, embed_size)), 0)
			self.add_word('<dialogue>') # id 1 # random vector
			self.add_word('</dialogue>') # id 2 # random vector
			self.add_word('<architect>') # id 3 # random vector
			self.add_word('</architect>') # id 4 # random vector
			self.add_word('<builder>') # id 5 # random vector
			self.add_word('</builder>') # id 6 # random vector

		if self.use_builder_action_tokens: ## False (zshi)
			vector_arr = np.concatenate((vector_arr, np.random.rand(12, embed_size)), 0)
			for color_key in type2id:
				self.add_word('<builder_pickup_'+color_key+'>')
				self.add_word('<builder_putdown_'+color_key+'>')

		self.word_vectors = vector_arr

	def load_vectors(self):
		vector_filename = self.vector_filename
		print("\nLoading word embedding vectors from", vector_filename, "...")

		embed_size = self.embed_size
		threshold = self.threshold

		f = None
		if vector_filename.endswith('.gz'): # TODO: this doesn't work. fix loading of word2vec pretrained model
			f = gzip.open(vector_filename, 'rt')
		else:
			f = open(vector_filename, 'r')

		# load word embeddings from file
		data = []
		data_rare = []
		for line in f:
			tokens = line.split()

			if len(tokens) != embed_size+1:
				print("Warning: expe