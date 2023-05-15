import random
import torch
from config import *

'''
A trainer that helps in training and evaluation
'''
class RLTrainer:
    def __init__(self, model, optimizer, data_loaders, train_config, data_handler):
        self.model = model
        self.optimizer = optimizer
        self.data_loaders = data_loaders
        self.train_config = train_config
        self.data_handler = data_handler
        self.max_eval_f1 = None
        self.softmax = nn.Softmax(dim=-1)

    def run(self):
        self.model = self.model.to(self.train_config.device)
        for i in range(int(self.train_config.n_epoch)):
            print('******************Epoch {}******************'.format(i + 1))
            self.train()
            self.eval(loader='dev')
        print('******************Training Finished******************')
        if 'test' in self.data_loaders:
            self.eval(loader='test')

    # one epoch of training
    def train(self):
        self.model.train()
        avg_batch_recalls = []
        batch_losses = []
        n_batch = 0
        for idx, batch in enumerate(self.data_loaders['train']):
            batch_loss = []
            n_batch += 1
            # get batch size
            bs = len(batch['bid'])
            if n_batch % 1000 == 0:
                torch.save(self.model.state_dict(), self.train_config.save_model_dir)
            # all_actions_taken reflects the latest "actions_taken" for each instance,
            # while all_actions_taken_at_all_steps reflects:
            # the latest "actions_taken" for each instance AT EACH TIMESTEP
            # it will be used to calculate the rewards
            all_actions_taken = [set([self.data_handler.lbl2idx['root']]) for _ in range(bs)]
            all_actions_taken_at_all_steps = [[set([self.data_handler.lbl2idx['root']])] for _ in range(bs)]
            # all instances will be assigned a default label "root" at the beginning
            latest_actions = [self.data_handler.lbl2idx['root'] for _ in range(bs)]
            # at timestep 0, the "next_labels" for each instance are the children of root
            next_labels = [self.data_handler.p2c_idx[latest_actions[0]] for _ in range(bs)]
            actions_probs = [[] for _ in range(bs)]
            preds, bids = [[self.data_handler.lbl2idx['root']] for _ in range(bs)], batch['bid']
            for i in range(self.train_config.max_train_steps):
                # device transfer
                for item in batch['encoding']:
                    batch['encoding'][item] = batch['encoding'][item].squeeze().to(self.train_config.device)
                next_labels_tensor, next_labels = self.data_handler.get_next_labels(all_actions_taken,
                                                                                    latest_actions,
                                                                                    next_labels)
                # device transfer
                next_labels_tensor = torch.tensor(next_labels_tensor, dtype=torch.int64).to(self.train_config.device)
                latest_actions_tensor = torch.tensor(latest_actions, dtype=torch.int64).to(self.train_config.device)
                # get probs for actions, (batch_size, action_size)
                action_probs = self.get_normalized_probs(batch, next_labels_tensor)

                # actions_probs.append(action_probs) # not used anymore
                # we get the positions of the predicted actions, need to convert them to actions index
                # if "cpu" not used, torch.argmax always give 0 in pytorch 2??
                log_probs = []
                m = torch.distributions.Categorical(action_probs.cpu())
                samples = m.sample()
                limits = torch.tensor([len(next_labels[i]) for i in range(len(next_labels))])
                # although a low probability is assigned
                while torch.any(samples >= limits):
                    samples = m.sample()
                actions_positions = [int(i) for i in samples]

                actions = [next_labels[i][actions_positions[i]] for i in range(len(next_labels))]
                for prob, pos in zip(action_probs, actions_positions):
                    log_probs.append(torch.log(prob[pos]))
                # used for calculating metrics
                # update "actions taken" and "latest actions"
                # actions taken: current action + previous actions
                # actions_taken_at_all_steps: current action until now.
                for j in range(len(actions)):
                    all_actions_taken[j].update([int(actions[j])])
                    all_actions_taken_at_all_steps[j].append(all_actions_taken[j].copy())
                    actions_probs[j].append(log_probs[j])
                    preds[j].append(int(actions[j]))
                    latest_actions[j] = int(actions[j])
            # update parameters every batch (not every step)
            loss = self.episode(preds=preds, bids=batch['bid'],
                                actions_probs=actions_probs,
                                all_actions_taken_at_all_steps=all_actions_taken_at_all_steps)
            batch_losses.append(loss)
            # used for checking overfitting (if the predicted results are very similar, likely overfitting
            # happens. The reasons could be n_steps too small, learning rate too large, lack of randomness, etc.
            # for pred in preds:
            #    print([self.data_handler.idx2lbl[p] for p in pred])
            # for bid in bids:
            #    print(bid)
            #    print(self.data_handler.data[bid]['categories'])

            # return recall during training, and F1 during evaluation.
            avg_batch_recalls.append(sum(self.data_handler.calculate_metric(preds, bids, 'train')) / len(preds))
            pf = self.train_config.print_freq
            if n_batch % pf == 0:
                print('Avg recall for batch {} - {}: {:.3f}'.format(n_batch - pf + 1, n_batch, sum(avg_batch_recalls[-pf: ]) / pf))
                print('Avg loss for batch {} - {}: {:.3f}'.format(n_batch - pf + 1, n_batch, sum(batch_losses[-pf:]) / pf))
                print(sum(avg_batch_recalls) / len(avg_batch_recalls))
                # randomly checks whether the model suffers from overfitting.
                # if predictions for all instances appear to be the same, then overfitting happens
                if random.random() > 0.9:
                    for pred in preds[:10]:
                        print([self.data_handler.idx2lbl[p] for p in pred])
                # for bid in bids:
                #    print(bid)
                #    print(self.data_handler.data[bid]['categories'])
            # save the model that performs best on dev
        # when all batches are processed
        print('Avg {} recall: {:.3f}'.format('train', sum(avg_batch_recalls) / len(avg_batch_recalls)))
        print('Avg {} loss: {:.3f}'.format('train', sum(batch_losses) / len(batch_losses)))

    # one epoch of evaluation / inference
    def eval(self, loader):
        assert loader in ['dev', 'test']
        self.model.eval()
        avg_batch_f1s = []
        n_batch = 0
        with torch.no_grad():
            for idx, batch in enumerate(self.data_loaders[loader]):
                n_batch += 1
                # get batch size
                bs = len(batch['bid'])
                # all_actions_taken reflects the latest "actions_taken" for each instance,
                # while all_actions_taken_at_all_steps reflects:
                # the latest "actions_taken" for each instance AT EACH TIMESTEP
                # it will be used to calculate the rewards
                all_actions_taken = [set([self.data_handler.lbl2idx['root']]) for _ in range(bs)]
                all_actions_taken_at_all_steps = [[set([self.data_handler.lbl2idx['root']])] for _ in range(bs)]
                # all instances will be assigned a default label "root" at the beginning
                latest_actions = [self.data_handler.lbl2idx['root'] for _ in range(bs)]
                # at timestep 0, the "next_labels" for each instance are the children of root
                next_labels = [self.data_handler.p2c_idx[latest_actions[0]] for _ in range(bs)]
                actions_probs = [[] for _ in range(bs)]
                preds, bids = [[self.data_handler.lbl2idx['root']] for _ in range(bs)], batch['bid']
                for i in range(self.train_config.max_eval_steps):
                    # device transfer
                    for item in batch['encoding']:
                        batch['encoding'][item] = batch['encoding'][item].squeeze().to(self.train_config.device)
                    next_labels_tensor, next_labels = self.data_handler.get_next_labels(all_actions_taken,
                                                                                        latest_actions,
                                                                                        next_labels)
                    # device transfer
                    next_labels_tensor = torch.tensor(next_labels_tensor, dtype=torch.int64).to(self.train_config.device)
                    latest_actions_tensor = torch.tensor(latest_actions, dtype=torch.int64).to(self.train_config.device)
                    # get probs for actions, (batch_size, action_size)
                    action_probs = self.get_normalized_probs(batch, next_labels_tensor)

                    # actions_probs.append(action_probs) # not used anymore
                    # we get the positions of the predicted actions, need to convert them to actions index
                    # if "cpu" not used, torch.argmax always give 0 in pytorch 2??
                    log_probs = []
                    # choose max during evaluation, no randomness anymore.
                    actions_positions = torch.argmax(action_probs.cpu(), dim=-1)

                    actions = [next_labels[i][actions_positions[i]] for i in range(len(next_labels))]
                    for prob, pos in zip(action_probs, actions_positions):
                        log_probs.append(torch.log(prob[pos]))
                    # used for calculating metrics

                    # update "actions taken" and "latest actions"
                    # actions taken: current action + previous actions
                    # actions_taken_at_all_steps: current action until now.
                    for j in range(len(actions)):
                        all_actions_taken[j].update([int(actions[j])])
                        all_actions_taken_at_all_steps[j].append(all_actions_taken[j].copy())
                        actions_probs[j].append(log_probs[j])
                        preds[j].append(int(actions[j]))
                        latest_actions[j] = int(actions[j])
            # do not calculate loss during evalution. We only care about avg F1.
            # used for checking overfitting
            # for pred in preds:
            #    print([self.data_handler.idx2lbl[p] for p in pred])
            # for bid in bids:
            #    print(bid)
            #    print(self.data_handler.data[bid]['categories'])

                avg_batch_f1s.append(sum(self.data_handler.calculate_metric(preds, bids, loader)) / len(preds))
                pf = self.train_config.print_freq
                if n_batch % pf == 0:
                    print('Avg F1 for batch {} - {}: {:.3f}'.format(n_batch - pf + 1, n_batch, sum(avg_batch_f1s[-pf: ]) / pf))
                    print(sum(avg_batch_f1s) / len(avg_batch_f1s))
                    if random.random() > 0.9:
                        for pred in preds[:10]:
                            print([self.data_handler.idx2lbl[p] for p in pred])
                # for bid in bids:
                #    print(bid)
                #    print(self.data_handler.data[bid]['categories'])
        # save the model that performs best on dev
        if loader == 'dev':
            if not self.max_eval_f1 or self.max_eval_f1 < sum(avg_batch_f1s) / len(avg_batch_f1s):
                self.max_eval_f1 = sum(avg_batch_f1s) / len(avg_batch_f1s)
                torch.save(self.model.state_dict(), self.train_config.save_model_dir)
                print('Best model saved')
        # when all batches are processed
        print('Avg {} F1: {:.3f}'.format(loader, sum(avg_batch_f1s) / len(avg_batch_f1s)))


    def get_normalized_probs(self, batch, next_labels_tensor):
        logits = self.model(batch, next_labels_tensor)
        # [PAD] as a label should never be selected. Therefore, we minimize its probability
        pad_idx = self.data_handler.lbl2idx['[PAD]']
        logits = (next_labels_tensor == pad_idx).float() * -9999 + (next_labels_tensor != pad_idx).float() * logits

        return self.softmax(logits)

    # loss is computed here, and model is updated here
    def episode(self, preds, bids, actions_probs, all_actions_taken_at_all_steps):
        rewards = self.data_handler.calculate_reward(preds, all_actions_taken_at_all_steps, bids)
        # (batch_size, max_steps)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.train_config.device)
        loss = torch.tensor(.0).to(self.train_config.device)
        self.optimizer.zero_grad()
        # * -1 to calculate loss
        # loss = torch.einsum('i,ij->ij', -rewards, actions_probs).sum()
        for i in range(len(actions_probs)):
            temp_loss = -torch.stack(actions_probs[i]) * rewards[i]
            loss += temp_loss.sum()
        loss.backward()
        self.optimizer.step()
        return loss
