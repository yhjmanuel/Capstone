###Hierarchical Text Classification Subtask

This is an auxiliary project. Tricks that it involves can be used to optimize the results in the main project experiments.

This project trains a deep reinforcement learning-based hierarchical text classification model. The idea can be applied for query hierarchical category classification, which may be used for improving the query-product-matching task results. 

This project is based on the paper Hierarchical Text Classification with Reinforced Label Assignment (Mao et al., 2020)

###How to run the codes:

Download yelp's 2018 dataset from https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset

Unzip the file and save "yelp_academic_dataset_business.json" and "yelp_academic_dataset_review.json" in the "dataset" folder, which also contains the hierarchy file.

Run

```
pip install -r requirements.txt 
```

Then run 

```
python make_dataset.py
```

To generate the preprocessed data, which will be saved as "train_data.pickle" in the "dataset" folder.

Then run 

```
python main.py
```

to start training. The model will be saved as "model.pt" in this folder. Feel free to change the training hyperparameters in "config.py".

For this project, the model is trained on 1 RTX 4090.

Thank you so much for reading!