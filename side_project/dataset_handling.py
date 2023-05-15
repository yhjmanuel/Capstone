# Tricks used in the repository https://github.com/morningmoni/HiLAP helps a lot for this implementation

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import statistics
from make_dataset import *
import numpy as np

'''
A data handler that helps finding the hierarchy within a tree
'''
class DataHandler:
    def __init__(self, data, hierarchy, algo):
        # data has format: {bid: {'categories': [], 'reviews': str}
        self.data = data
        # each bid and its corresponding label index
        self.bid2lbl = {}
        self.hierarchy = hierarchy
        # if algo not in ['rl', 'flat']:
        #     raise NotImplementedError('Only reinforcement learning & flat algorithm implemented')
        self.algo = algo
        self.p2c = defaultdict(set)
        self.c2p = defaultdict(set)
        self.p2c_idx = {}
        self.c2p_idx = {}
        # three special labels
        self.lbl2idx = {'[PAD]': 0, '[STOP]': 1, 'root': 2}
        self.idx2lbl = {}
        self._make_mappings()

    def _make_mappings(self):
        # 0 for [PAD], 1 for [STOP], 2 for root
        start_idx = 3
        for key in self.hierarchy:
            if key not in self.lbl2idx:
                self.lbl2idx[key] = start_idx
                start_idx += 1
            self.p2c[key].update(self.hierarchy[key]['children'])
            self.c2p[key].update(self.hierarchy[key]['parents'])
        self.idx2lbl = {self.lbl2idx[key]: key for key in self.lbl2idx}
        # hierarchy in index
        for key in self.p2c:
            self.p2c_idx[self.lbl2idx[key]] = [self.lbl2idx[value] for value in self.p2c[key]]
        for key in self.c2p:
            self.c2p_idx[self.lbl2idx[key]] = [self.lbl2idx[value] for value in self.c2p[key]]
        # map bid to lbl idx
        for bid in self.data:
            self.bid2lbl[bid] = set([self.lbl2idx[lbl] for lbl in self.data[bid]['categories']])

    def idx_batch_p2c(self, p_batch):
        return [self.p2c_idx[item] for item in p_batch]

    def idx_batch_c2p(self, c_batch):
        return [self.c2p_idx[item] for item in c_batch]

    @staticmethod
    def pad_to_max_len(lists_to_pad):
        max_len = max(len(item) for item in lists_to_pad)
        # (batch_size, n_max_labels)
        padded = torch.zeros(len(lists_to_pad), max_len)
        for i in range(len(lists_to_pad)):
            padded[i, :len(lists_to_pad[i])] = torch.tensor(list(lists_to_pad[i]))
        return padded

    def get_next_labels(self, actions_taken, latest_actions, prev_labels):
        # actions_taken: (batch_size, n_labels_assigned)
        # return: next_labels: (batch_size, n_labels_to_select)
        # n_labels_to_select may be different for each instance. Therefore, padding is needed.
        # next labels should be selected via: all selected nodes' children - all selected nodes
        for i in range(len(latest_actions)):
            # add the latest label's children to the next_labels
            prev_labels[i] = set(prev_labels[i])
            if latest_actions[i] != self.lbl2idx['[STOP]']:
                prev_labels[i].update(self.p2c_idx[latest_actions[i]])
            prev_labels[i].update([self.lbl2idx['[STOP]']])
            prev_labels[i] = list(prev_labels[i].difference(actions_taken[i]))
        return self.pad_to_max_len(prev_labels), prev_labels

    # record f1 / recall for each single instance
    def calculate_metric(self, batch_preds, batch_bids, mode, epsilon=1e-32):
        batch_metric = []
        for pred, bid in zip(batch_preds, batch_bids):
            pred = pred[1:]
            lbls = self.bid2lbl[bid]
            if self.lbl2idx['[STOP]'] in pred:
                #remove anything after ['STOP']
                 if len(pred[:pred.index(self.lbl2idx['[STOP]'])]) >= 1:
                    pred = pred[:pred.index(self.lbl2idx['[STOP]'])]
            correct = set(pred).intersection(lbls)
            # correct = correct.difference([self.lbl2idx['[STOP]']])
            correct = len(correct)
            precision = correct / len(set(pred))
            recall = correct / len(lbls)
            # we look at recall for training, f1 for evaluation
            if mode != 'train':
                batch_metric.append(2 * precision * recall / (precision + recall + epsilon))
            else:
                batch_metric.append(recall)

        return batch_metric

    def calculate_reward(self, batch_actions, batch_actions_taken, batch_bids):
        # batch_actions:
        batch_rewards = []
        # calculate the rewards for every instance in a batch
        for actions, actions_taken, bid in zip(batch_actions, batch_actions_taken, batch_bids):
            # labels: labels in the dataset + [STOP]
            lbls = self.bid2lbl[bid]
            # ignore the first action, which is "root"
            # ignore the first action taken set, which is {"root"}
            instance_rewards = []
            for action, action_taken in zip(actions[1:], actions_taken[1:]):
                if action in lbls:
                    instance_rewards.append(1.0)
                # if the action is stop and it should stop (because all label categories have been predicted)
                elif action == self.lbl2idx['[STOP]'] and len(set(action_taken).intersection(set(lbls))) == len(set(lbls)):  
                    instance_rewards.append(1.0)
                else:
                    instance_rewards.append(-1.0)
            # reward normalization
            instance_rewards -= np.mean(instance_rewards)
            instance_rewards /= (np.std(instance_rewards) + 1e-32)
            instance_rewards = list(instance_rewards)
            batch_rewards.append(instance_rewards)

        return batch_rewards


# extend PyTorch's dataset, used for making dataloaders
class TorchDataset(Dataset):
    def __init__(self, data, tokenize_config):
        self.data = [{'bid': bid, 'reviews': data[bid]['reviews']} for bid in data]
        self.config = tokenize_config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        instance = self.data[index]
        encoding = self.tokenizer(instance['reviews'], padding='max_length',
                                  max_length=int(self.config.max_length),
                                  return_token_type_ids=False,
                                  truncation=True, return_tensors='pt')
        return {'bid': instance['bid'], 'encoding': encoding}
