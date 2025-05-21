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
This section briefly explains how to run the different algorithms, including:
 - ALS
 - SVD++
 - ...

Each algorithm has a python file (```.py```), and a notebook (```.ipynb```).
The python scripts can be found in the ```./models``` directory, and the notebooks 
can be found in the ```./notebooks``` directory.

After initializing the environment, the models can be run with ```python models/modelName.py```.

### ALS
Running the ```models/ALS.py``` script will perform, over five seeds, 30 iterations, with 15 latent factors, and regularization parameter of 20.
It will print the train and validation score for each iteration, and, at the end, the mean and standard deviation for the train and validation rmse.
Afterward, in lines 141 to 156 it outputs a submission. These lines are commented.

At the end of the file, there is also the code for the hyperparameter optimization. To run it, uncomment the last five lines, and comment
the lines 110 to 138 running over the five seeds.

The notebook can be found in ```notebooks/ALS.ipynb```.
