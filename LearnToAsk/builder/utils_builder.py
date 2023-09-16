import re
from builder.decoding import update_built_config, is_feasible_action, update_action_history
from builder.diff import diff, get_diff, dict_to_tuple
from utils import action_label2action_repr, BuilderActionExample
import copy

stop_action_label = 7*11*9*11

def is_a_id(string):
    line = re.sub(r'\d+', '', string)
    if line == 'B-A-C-': return True
    return False

def split_line(line):
    line_elems = line.split('\t')
    if len(line_elems) == 2:
        utterance = line_elems[0]
        label = line_elems[1]
        corrected_utterance = None
        return utterance, label, corrected_utterance
    elif len(line_elems) == 3:
        utterance = line_elems[0]
        label = line_elems[1]
        corrected_utterance = line_elems[2]
        return utterance, label, corrected_utterance
    else:
        print('Error', line)
        return None, None, None

def compute_action_prf(fn, fp, tp):
    action_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    action_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    action_f1 = ((2 * action_precision * action_recall) / (action_precision + action_recall)) if (action_precision + action_recall) > 0 else 0.0
    return action_precision, action_recall, action_f1

def compute_metrics(raw_input, action_seq):
    initial_built_config = raw_input.initial_prev_config_raw
    ground_truth_end_built_config = raw_input.end_built_config_raw
    
    built_config_post_last_action = initial_built_config
    for action_label in action_seq:
        built_config_post_last_action = update_built_config(built_config_post_last_action, action_label)

    generated_end_built_config = built_config_post_last_action

    net_change_generated = diff(gold_config=generated_end_built_config, built_config=initial_built_config)
    net_change_gt = diff(gold_config=ground_truth_end