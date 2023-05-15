import ast
import json
import random
import pickle
from collections import defaultdict
from config import *


# read the hierarchy tree file
def get_hierarchy(hierarchy_file):
    nodes = {}
    with open(hierarchy_file) as f:
        for line in f:
            node = ast.literal_eval(line)
            title = node['title']
            del node['title']
            nodes[title] = node
    return nodes


# read yelp's review dataset
def get_reviews(yelp_review_file):
    bz_reviews = defaultdict(list)
    with open(yelp_review_file) as f:
        for line in f:
            data = json.loads(line)
            bz_reviews[data['business_id']].append(data['text'])
    return bz_reviews


# read yelp's business categories dataset
def get_bz_categories(nodes, yelp_bz_file):
    bz_cats = {}
    with open(yelp_bz_file) as f:
        for line in f:
            data = json.loads(line)
            # delete businesses with no categories
            if data['categories']:
                # delete categories that are not in our hierarchy tree
                cats = [cat for cat in data['categories'].split(', ') if cat in nodes]
                if cats:
                    bz_cats[data['business_id']] = cats
    return bz_cats


# merge the two datasets above
def merge_review_cats(bz_cats, bz_reviews, n_max_reviews, save_dir):
    # based on bz_cats
    data = {}
    for bid in bz_cats:
        if bid in bz_reviews:
            reviews = bz_reviews[bid]
            if len(reviews) > n_max_reviews:
                reviews = random.sample(reviews, n_max_reviews)
            data[bid] = {'categories': bz_cats[bid], 'reviews': ' '.join(reviews)}
    file = open(save_dir, 'wb')
    pickle.dump(data, file)
    print('Preprocessed data saved to {}'.format(save_dir))


if __name__ == '__main__':
    nodes = get_hierarchy(MakeDataConfig.hierarchy_file)
    bz_reviews = get_reviews(MakeDataConfig.yelp_review_file)
    bz_cats = get_bz_categories(nodes, MakeDataConfig.yelp_bz_file)
    merge_review_cats(bz_cats, bz_reviews, MakeDataConfig.n_max_reviews, MakeDataConfig.save_dir)