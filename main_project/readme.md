### Query-Product Matching Subtask

This project serves as a solution for task 1 (search ranking) and task 2 (relevance level classification) for the ESCI challenge https://amazonkddcup.github.io/ 

### How to run the codes

Download the dataset from https://github.com/amazon-science/esci-data/tree/main/shopping_queries_dataset

Put all the files in the same directory as this readme.md

Run

```
pip install -r requirements.txt 
```

#### Task1
For experimenting with task 1 (the search ranking task), run

```
sh get_task_data.sh 1
```

To generate the preprocessed data, which will be saved in the "task_data" folder.

Then run 

```
sh run_experiment.sh ranking_baseline.py config/task1.config
```

or

```
sh run_experiment.sh ranking_optim_DeltaNDCG.py config/task1.config
```

to start doing experiments (for MSE-based method and Delta-NDCG-based method, respectively). 

Feel free to change the hyperparameters in "config/task1.config". Logs will be printed to console.

The model will be saved in the "models" folder. 

#### Task2
For experimenting with task 2 (the search relevance level classification task), run

```
sh get_task_data.sh 2
```

To generate the preprocessed data, which will be saved in the "task_data" folder.

Then run 

```
sh run_experiment.sh classification_baseline.py config/task2.config
```

to start doing experiments. 

Feel free to change the hyperparameters in "config/task2.config". Logs will be printed to console.

The model will be saved in the "models" folder. 

For task 2, to save time, you can set "cache_available=false" in the config file for the first time, then the encoded data will be saved in the "cache" folder. Then for next experiments, if data tokenization logic is the same, you can set "cache_available=true", then the encoding will be loaded directly.

For this project, the model is trained on 1 RTX 4090.

Thank you so much for reading!
