# Hierarchical-Classification

## Setup
Install the conda environment using the environment.yml file:
```
conda env create -f environment.yml
```

## Data
Download the data from [here](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) and extract it to the `dataset/` folder.

Now you can run the `xml_extractor.py` to correctly organize the validation data:
```
python -m moduels.xml_extractor
```

Now you should have the following folder structure:
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
    hierarchy.csv
    val_mapping.csv
```

## Training
To train the model, run the `run_l.py` file:
```
python run_l.py
```