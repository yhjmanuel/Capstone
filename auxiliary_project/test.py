import unittest
import torch
from transformers import AutoTokenizer
from model import RLModel
from make_dataset import *
from dataset_handling import *
import pickle

'''
Some unit tests that test the model and the data handler.
'''
class TestRLModel(unittest.TestCase):
    def setUp(self):
        self.rl_model = RLModel(RLModelConfig)
        self.reviews_case1 = ['Review 1', 'Review 2', 'Review 3']
        self.reviews_case2 = ['Review for a product'] * 3
        self.tokenizer = AutoTokenizer.from_pretrained(RLModelConfig.pretrained_model_name)

    def test_shape_case1(self):
        encoding = self.tokenizer(self.reviews_case1, padding='max_length',
                                  max_length=5,
                                  return_token_type_ids=False,
                                  truncation=True, return_tensors='pt')
        review_batch = {'bid': [10, 11, 12], 'encoding': encoding}
        # imitate the initial state
        next_labels_batch = torch.tensor([[1,2,3], [1,2,3], [1,2,3]])
        result = self.rl_model(review_batch, next_labels_batch)
        self.assertEqual(torch.Size([3, 3]), result.shape)

    def test_shape_case2(self):
        encoding = self.tokenizer(self.reviews_case2, padding='max_length',
                                  max_length=10,
                                  return_token_type_ids=False,
                                  truncation=True, return_tensors='pt')
        review_batch = {'bid': [10, 11, 12], 'encoding': encoding}
        # this time we include padded labels
        next_labels_batch = torch.tensor([[30, 31, 32, 33], [35, 36, 37, 0], [50, 0, 0, 0]])
        result = self.rl_model(review_batch, next_labels_batch)
        self.assertEqual(torch.Size([3, 4]), result.shape)


class TestDataHandler(unittest.TestCase):
    def setUp(self):
        hierarchy_file = 'dataset/Taxonomy_100'
        hierarchy = get_hierarchy(hierarchy_file)

        file = open('dataset/train_data.pickle', 'rb')
        data = pickle.load(file)
        self.data_handler = DataHandler(data=data, hierarchy=hierarchy, algo='rl')
        file.close()

    def test_make_mappings(self):
        self.assertEqual(len(self.data_handler.p2c), len(self.data_handler.c2p))
        root_children = ['Beauty & Spas', 'Active Life', 'Hotels & Travel',
                         'Mass Media', 'Home Services', 'Event Planning & Services',
                         'Pets', 'Arts & Entertainment', 'Nightlife', 'Shopping', 'Food',
                         'Health & Medical', 'Religious Organizations', 'Restaurants',
                         'Professional Services', 'Local Flavor', 'Financial Services', 'Local Services',
                         'Public Services & Government', 'Automotive', 'Education']
        for root_child in root_children:
            self.assertTrue(root_child in self.data_handler.p2c['root'])
            self.assertTrue('root' in self.data_handler.c2p[root_child])
        transportation_children = ['Public Transportation', 'Airlines', 'Limos',
                                   'Taxis', 'Airport Shuttles']
        for t_child in transportation_children:
            self.assertTrue(t_child in self.data_handler.p2c['Transportation'])
            self.assertTrue('Transportation' in self.data_handler.c2p[t_child])
        self.assertEqual(len(self.data_handler.p2c), 540)
        self.assertEqual(len(self.data_handler.lbl2idx), 542)

    def test_pad_next_labels(self):
        next_labels = [set([1, 2, 3]), set([101, 102, 103, 104, 105]), set([16])]
        padded_labels = self.data_handler.pad_to_max_len(next_labels)
        self.assertEqual(padded_labels.shape, torch.Size([3, 5]))

        next_labels = [set([1, 3, 3]), set([106, 107, 108, 109, 120]), set([16])]
        padded_labels = self.data_handler.pad_to_max_len(next_labels)
        self.assertEqual(padded_labels.shape, torch.Size([3, 5]))

    def test_get_next_labels(self):
        map = self.data_handler.lbl2idx
        # test a [batch_size = 1] case
        actions_taken = [set([map['root']])]
        latest_actions = [map['root']]
        prev_labels = [set([self.data_handler.lbl2idx[child] for child in self.data_handler.p2c['root']])]
        next_labels, _ = self.data_handler.get_next_labels(actions_taken, latest_actions, prev_labels)

        actions_taken = [set([map['root'], map['Food']])]
        latest_actions = [map['Food']]
        prev_labels = [set([self.data_handler.lbl2idx[child] for child in self.data_handler.p2c['root']])]
        next_labels, _ = self.data_handler.get_next_labels(actions_taken, latest_actions, prev_labels)
        # all children of root - "Food"
        target = set(self.data_handler.p2c_idx[map['root']]).difference(set([map['Food']]))
        # all children of "Food"
        target.update(self.data_handler.p2c_idx[map['Food']])
        # include [STOP] in target
        target.update([self.data_handler.lbl2idx['[STOP]']])
        self.assertTrue(torch.equal(torch.tensor(next_labels[0], dtype=torch.int32),
                                    torch.tensor(sorted(list(target)), dtype=torch.int32)))

        # increase batch_size to 2
        actions_taken = [set([map['root']]), set([map['root'], map['Food'], map['Specialty Food']])]
        latest_actions = [map['root'], map['Specialty Food']]
        prev_labels = [set([self.data_handler.lbl2idx[child] for child in self.data_handler.p2c['root']])]
        prev_labels.append(set([self.data_handler.lbl2idx[child] for child in self.data_handler.p2c['root']]))
        prev_labels[1].difference(set([map['Food']]))
        prev_labels[1].update(self.data_handler.p2c_idx[map['Food']])

        next_labels, _ = self.data_handler.get_next_labels(actions_taken, latest_actions, prev_labels)
        target = set([map['root']])
        target.update(self.data_handler.p2c_idx[map['root']])
        target.update(self.data_handler.p2c_idx[map['Food']])
        target.update(self.data_handler.p2c_idx[map['Specialty Food']])
        target = target.difference(actions_taken[1])
        target.update([self.data_handler.lbl2idx['[STOP]']])

        self.assertTrue(torch.equal(torch.tensor(list(target), dtype=torch.int32),
                                    torch.tensor(next_labels[1], dtype=torch.int32)))

    def test_calculate_f1(self):
        batch_preds = [[2, 13, 227, 205, 204], [2, 22, 23, 24]]
        batch_bids = ['EsMcGiZaQuG1OOvL9iUFug', 'TGWhGNusxyMaA4kQVBNeew']
        target1 = (2 * 1 * 0.75) / (1 + 0.75)
        target2 = (2 * 0.3333 * 0.5) / (0.3333 + 0.5)
        result = self.data_handler.calculate_metric(batch_preds, batch_bids, mode='eval')
        self.assertAlmostEqual(target1, result[0], 3)
        self.assertAlmostEqual(target2, result[1], 3)

    def test_calculate_reward(self):
        # correct label: [[13, 227, 205], [22, 387]]
        batch_actions = [[0, 12, 13, 227, 205, 1, 204, 202], [0, 22, 23, 1, 385, 1, 386, 1]]
        batch_actions_taken = [[set(batch_actions[0][:i]) for i in range(1, len(batch_actions[0]) + 1)]]
        batch_actions_taken.append([set(batch_actions[1][:i]) for i in range(1, len(batch_actions[1]) + 1)])
        batch_bids = ['EsMcGiZaQuG1OOvL9iUFug', 'TGWhGNusxyMaA4kQVBNeew']
        result = self.data_handler.calculate_reward(batch_actions, batch_actions_taken, batch_bids)
        self.assertAlmostEqual(-1.1547, result[0][0], 4)

        batch_actions = [[0, 13, 12, 227, 1, 205, 204, 202], [0, 22, 23, 1, 387, 1, 386, 1]]
        batch_actions_taken = [[set(batch_actions[0][:i]) for i in range(1, len(batch_actions[0]) + 1)]]
        batch_actions_taken.append([set(batch_actions[1][:i]) for i in range(1, len(batch_actions[1]) + 1)])
        result = self.data_handler.calculate_reward(batch_actions, batch_actions_taken, batch_bids)
        print(result)
        self.assertAlmostEqual(1.1547, result[0][0], 4)