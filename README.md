# Computational Intelligence Lab: Collaborative Filtering

## Setup
1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ````
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Methods
This section briefly explains how to run the different models, namely:
 - ALS
 - SVD++
 - SVD++ extended
 - NCF
 - Bagged NCF

Each algorithm has a python file (```.py```), and a notebook (```.ipynb```).
The python scripts can be found in the ```./models``` directory, and the notebooks 
can be found in the ```./notebooks``` directory.

After initializing the environment, the models can be run with ```python models/modelName.py```.
We describe how to run each model individually.

### ALS
Running the ```models/ALS.py``` script will perform, over five seeds, 30 iterations, with 15 latent factors, and regularization parameter of 20.
It will print the train and validation score for each iteration, and, at the end, the mean and standard deviation for the train and validation RMSE.
Afterward, in lines 141 to 156 it outputs a submission. These lines are commented.

At the end of the file, there is also the code for the hyperparameter optimization. To run it, uncomment the last five lines, and comment
the lines 110 to 138 running over the five seeds.

The notebook can be found in ```notebooks/ALS.ipynb```.

### SVD++
Running the ```models/svdpp_results.py``` script with the following cli arguments: ```--factors 50 --lr 0.005 --reg 0.05 --epochs 45 --multi_seed```
will run the method over the five seeds, and print the mean and standard deviation for the train and validation RMSE over the five seeds.

Removing the ```-multi_seed``` argument, it simply trains on the full dataset, and makes a submission.

To see the different available cli arguments simply call the script with ```-h```.

### SVD++ extended
Running the ```models/svd++_with_wishlist.py``` script will run the model using seed ```42```.
To use other seeds, the following cli argument can be used: ```--seed SEED```.

To see the different available cli arguments simply call the script with ```-h```.

### NCF
Running the ```models/ncf.py``` scripts runs the NCF model over the five seeds, printing 
the train loss and validation RMSE for each epoch, and the final train and validation RMSE 
for each seed.

After running the five seeds, it prints the mean and standard deviation for the train and validation RMSE.

### Bagged NCF
Running the ```models/ncf_bagged.py``` scripts runs the Bagged NCF model over the five seeds, printing 
the train loss and validation RMSE for each epoch, and the final train and validation RMSE 
for each seed.

After running the five seeds, it prints the mean and standard deviation for the train and validation RMSE.
