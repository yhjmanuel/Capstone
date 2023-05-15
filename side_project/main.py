from trainer import *
from config import *
from dataset_handling import *
from make_dataset import *
from model import *
import random
from torch.utils.data import DataLoader
import torch


if __name__ == '__main__':
    # read data
    print('Loading Data...')
    hierarchy_file = 'dataset/Taxonomy_100'
    hierarchy = get_hierarchy(hierarchy_file)
    file = open('dataset/train_data.pickle', 'rb')
    data = pickle.load(file)
    #data = {bid: v for bid, v in data.items() if len(data[bid]['categories']) <= 10}
    data_handler = DataHandler(data=data, hierarchy=hierarchy, algo='rl')
    file.close()
    print('Data Loaded!')

    # make datasets
    print('Making Datasets...')
    seed = 19260817
    bids = list(data)
    random.Random(seed).shuffle(bids)
    len_bids = len(bids)
    train_bids, dev_bids, test_bids = set(bids[: int(len_bids * 0.8)]), set(bids[int(len_bids * 0.8): int(len_bids * 0.9)]), set(bids[int(len_bids * 0.9):])

    train_set = TorchDataset({bid: v for bid, v in data.items() if bid in train_bids}, TokenizeConfig)
    dev_set = TorchDataset({bid: v for bid, v in data.items() if bid in dev_bids}, TokenizeConfig)
    test_set = TorchDataset({bid: v for bid, v in data.items() if bid in test_bids}, TokenizeConfig)

    train_loader = DataLoader(train_set, batch_size=TrainConfig.batch_size, shuffle=True, pin_memory=True)
    dev_loader = DataLoader(dev_set, batch_size=TrainConfig.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=TrainConfig.batch_size, shuffle=True, pin_memory=True)
    print('Dataset Made!')

    loaders = {'train': train_loader, 'dev': dev_loader, 'test': test_loader}
    print('start training...')
    # make model, optimizer and trainer
    rl_model = RLModel(RLModelConfig)
    # rl_model.load_state_dict(torch.load('model.pt'))
    optimizer = torch.optim.Adam(rl_model.parameters(), lr=TrainConfig.lr)
    rl_trainer = RLTrainer(model=rl_model, optimizer=optimizer, data_loaders=loaders,
                           train_config=TrainConfig, data_handler=data_handler)

    # start training and evaluation
    rl_trainer.run()
