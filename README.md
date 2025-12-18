# tda4halluAppl

This repository is an application of a TOpology-based HAllucination detector (TOHA; [paper](https://arxiv.org/abs/2504.10063v1), [repository](anonymous.4open.science/r/tda4hallu-BED5)) on LUSTER, an Emotionally Intelligent Task-oriented Dialogue System ([paper](https://arxiv.org/abs/2507.01594)).

## What we changed on the TOHA repository
We took the code from the TOHA repository and removed the usage of the Comet platform, since we ran our code on a HPC with no internet access. We also removed code parts that we were not able to or didn't intend to use, like anything related to the data set RAGTruth.
Since our goal was to apply TOHA on LUSTER, we added several files for preprocessing and labeling LUSTER data. The main ones (that led to the successfull results) are:

**src/preprocess/preprocessLuster.py**: Loads the turns.pkl file provided and the manual hallucinations labels and processes everything to the dataframe form required.

**src/preprocess/PreprocessLusterData.py**: Inherits from the class above and sets the class

**src/preprocess/get_manual_labels.py**: Contains the hallucination labels, that I added manually.

## Data set
The data set consists of dialogue simulations based on [MultiWOZ 2.1](https://arxiv.org/abs/1907.01669). It is located in ${LUSTER_REPOSITORY_BASE_PATH')/data/training_logs/experiences_succ/analysis_outputs/turns.pkl. The corresponding manual hallucination labels are in src/preprocess/get_manual_labels.py.

## How to use this repository
Pull this repository. You need to install LUSTER. There must me an environment variable LUSTER_REPOSITORY_BASE_PATH equal to the repository path of your LUSTER installation.
Run main.py by running "uv run python3 main.py" in the shell. On a NVIDIA RTX6000 GPU it should take ~1h to finish.
