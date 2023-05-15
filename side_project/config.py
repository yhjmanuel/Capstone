# configs of the project
import torch.nn as nn


class MakeDataConfig:
    hierarchy_file = 'dataset/Taxonomy_100'
    yelp_review_file = 'dataset/yelp_academic_dataset_review.json'
    yelp_bz_file = 'dataset/yelp_academic_dataset_business.json'
    n_max_reviews = 3
    save_dir = 'dataset/train_data.pickle'


class RLModelConfig:
    pretrained_model_name = 'microsoft/deberta-base'
    n_labels = 542
    label_embed_dim = 50
    act = nn.ReLU()


class FlatModelConfig:
    pretrained_model_name = 'microsoft/deberta-base'
    n_labels = 542


class TokenizeConfig:
    pretrained_model_name = 'microsoft/deberta-base'
    max_length = 256

class TrainConfig:
    device = 'cuda'
    n_epoch = 5
    max_train_steps = 12
    max_eval_steps = 3
    save_model_dir = 'model.pt'
    lr = 8e-6
    batch_size = 40
    print_freq = 100
