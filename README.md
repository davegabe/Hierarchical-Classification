# Hierarchical-Classification

## Setup
Install the conda environment using the environment.yml file:
```
conda env create -f environment.yml
```

## Data
Download the data from [here](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) and extract it to the `dataset/` folder.
You should have the following folder structure:
```
dataset/
    train/
        n01440764/ # e.g. synset
            n01440764_10026.JPEG # e.g. image
            ...
    val/
        n01440765/ # e.g. synset
            n01440765_10026.JPEG # e.g. image
            ...
    test/
        n01440766/ # e.g. synset
            n01440766_10026.JPEG # e.g. image
            ...
    LOC_synset_mapping.txt # e.g. mapping from synset to human readable label
```