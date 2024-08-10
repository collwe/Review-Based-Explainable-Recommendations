# Review-Based Explainable Recommendations: A Transparency Perspective

## Installation
Install PyTorch following the instructions on the (https://pytorch.org/). The code has been tested over PyTorch 1.8.1 versions.

Then install the other dependencies.
```
conda env create -f environment.yml
```

## Dataset
The dataset preprocessing code is in ``/data/Sentires-preprocessing.py`` with randomized train|validation|test split in 80|20|20 ratios. 

## Train
The code can be executed with:
``python /src/run.py``
Default parameters are not the best performing-hyper-parameters. Hyper-parameter selection has been specified in our paper and needs to be done through the command line arguments.
