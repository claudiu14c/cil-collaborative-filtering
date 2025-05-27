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
Note that the notebooks were mainly used for manual exploration, so to recreate the results 
the python scripts should be used.

After initializing the environment, the models can be run with ```python models/modelName.py```.
Below we describe how to run each model individually.

Note that there are some other models in the ```models``` directory that were not tested.
In particular, there are three different SVD++ scripts, including the SVD++-hybrid script,
which we described in the report, but did not include in testing due to its high 
computational requirements and lack of improvement over the SVD++ method.
We left them in the repository for completeness.

### ALS
Running the ```models/ALS.py``` script will perform, over five seeds, 30 iterations, with 15 latent factors, and regularization parameter of 20.
It will print the train and validation score for each iteration, and, at the end, the mean and standard deviation for the train and validation RMSE.
Afterward, in lines 141 to 156 it outputs a submission. These lines are commented.

At the end of the file, there is also the code for the hyperparameter optimization. To run it, uncomment the last five lines, and comment
the lines 110 to 138 running over the five seeds.

The notebook can be found in ```notebooks/ALS.ipynb```.

### SVD++
Running the ```models/svdpp_results.py``` script with the following CLI arguments: ```--factors 50 --lr 0.005 --reg 0.05 --epochs 45 --multi_seed```
will run the method over the five seeds, and print the mean and standard deviation for the train and validation RMSE over the five seeds.

Removing the ```-multi_seed``` argument, it simply trains on the full dataset, and makes a submission.

To see the different available CLI arguments simply call the script with ```-h```.

### SVD++ extended
Running the ```models/svd++_with_wishlist.py``` script will run the model using seed ```42```.
To use other seeds, the following CLI argument can be used: ```--seed SEED```.

To see the different available CLI arguments simply call the script with ```-h```.

### NCF
Running the ```models/ncf.py``` scripts runs the NCF model over the five seeds, printing 
the train loss and validation RMSE for each epoch, and the final train and validation RMSE 
for each seed.

After running the five seeds, it prints the mean and standard deviation for the train and validation RMSE.

### Bagged NCF
Running the ```models/bagged_ncf.py``` scripts runs the Bagged NCF model over the five seeds, printing 
the train loss and validation RMSE for each ensemble for each epoch, 
and the final train and validation RMSE for each seed (over all the ensembles).

After running the five seeds, it prints the mean and standard deviation for the train and validation RMSE.
