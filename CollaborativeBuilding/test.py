import os
import json
import argparse 
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from builder.vocab import Vocabulary
from builder.model import Builder
from builder.dataloader_with_glove import BuilderDataset, RawInputs
from builder.utils_builder import evaluate_metrics
from utils import *


def main(args, config):
    testdataset = BuilderDataset(args, split='test', encoder_vocab=None)
    test_items = testdataset.items
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {}'.format(device))
    with open(args.encoder_vocab_path, 'rb') as f:
        encoder_vocab = pickle.load(f)
    model = Builder(config, vocabulary=encoder_vocab).to(device)
    model.load_state_dict(torch.load(os.path.join(args.saved_models_path, "model.pt")))

    model.eval()
    test_loss = 0
    test_pred_seqs = []
    test_raw_inputs = []
    with torch.no_grad():
        test_total_color = 0
        test_total_location = 0
        test_total_color_correct = 0
        test_total_location_correct = 0
        test_total_action_type_correct = 0
        test_total_actions = 0
        for i, data in enumerate(tqdm(test_items)):
            encoder_inputs, grid_repr_inputs, action_repr_inputs, labels, location_mask, raw_input = data
            encoder_inputs, grid_repr_inputs, action_repr_inputs, labels, location_mask = encoder_inputs.unsqueeze(0), grid_repr_inputs.unsqueeze(0), action_repr_inputs.unsqueeze(0), labels.unsqueeze(0), location_mask.unsqueeze(0)
            """
            encoder_inputs: [batch_size, max_length]
            grid_repr_inputs: [batch_size=1, act_len, 8, 11, 9, 11]
            action_repr_inputs: [batch_size=1, act_len, 11]
            location_mask: [batch_size=1, act_len, 1089]
            labels: [batch_size=1, act_len, 7]
            """
            loss, test_acc, test_predicted_seq = model(encoder_inputs.long().to(device), grid_repr_inputs.to(device), action_repr_inputs.to(device), labels.long().to(device), location_mask.to(device), raw_input=raw_input, dataset=testdataset)
            
            test_loss += sum(loss)
            test_total_action_type_correct += test_acc[0]
            test_total_location += test_acc[1]
            test_total_location_correct += test_acc[2]
            test_total_color += test_acc[3]
