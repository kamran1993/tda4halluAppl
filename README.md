# tda4halluAppl

This repository is an application of a TOpology-based HAllucination detector (TOHA; [paper](https://arxiv.org/abs/2504.10063v1), [repository](anonymous.4open.science/r/tda4hallu-BED5)) on LUSTER, an Emotionally Intelligent Task-oriented Dialogue System ([paper](https://arxiv.org/abs/2507.01594)).

## What we changed on the TOHA repository
We took the code from the TOHA repository and removed the usage of the Comet platform, since we ran our code on a HPC with no internet access. We also removed code parts that we were not able to or didn't intend to use, like anything related to the data set RAGTruth.
Since our goal was to apply TOHA on LUSTER, we added several files for preprocessing and labeling LUSTER data. The main ones (that led to the successfull results) are:

**src/preprocess/preprocessLuster.py**: Loads the turns.pkl file provided and the manual hallucinations labels and processes everything to the dataframe form required.

**src/preprocess/PreprocessLusterData.py**: Inherits from the class above and sets the class

**src/preprocess/get_manual_labels.py**: Contains the hallucination labels, that I added manually.

**src/preprocess/labelLusterData.py**: Labeling script for LUSTER data, based on ConvLab3/convlab/nlg/evaluate_unified_datasets_v3.py. Wasn't very successfull, therefore we didn't use it for our main results.

**src/preprocess/labelLusterData2.py**: Labeling script for LUSTER data, based on ConvLab3/convlab/nlg/analyze_systematic.py. Wasn't very successfull, therefore we didn't use it for our main results. However, when running tda4halluApppl on LusterDataNew or LusterDataNewSent by doing the setup in config/master.yaml, it will use the logical AND of labels from labelLusterData.py and labelLusterData2.py.

**src/preprocess/config_baseline.py**: Configuration file used by labelLusterData2.py. Retrieved from ConvLab3/convlab/nlg/config_baseline.py.

## Data set
The data set consists of dialogue simulations based on [MultiWOZ 2.1](https://arxiv.org/abs/1907.01669). It is located in ${LUSTER_REPOSITORY_BASE_PATH')/data/training_logs/experiences_succ/analysis_outputs/turns.pkl. The corresponding manual hallucination labels are in src/preprocess/get_manual_labels.py.

## How to use this repository
You need to install LUSTER. There must me an environment variable LUSTER_REPOSITORY_BASE_PATH equal to the repository path of your LUSTER installation. Install the requirements for this repository by running "pip3 install -r container_setups/requirements.txt". Run the porject by running "python3 main.py". On a NVIDIA RTX6000 GPU it should take ~1h to finish.

If you intend to use the less successfull labeling scripts src/preprocess/labelLusterData.py and src/preprocess/labelLusterData2.py, you need to install [ConvLab3](https://github.com/ConvLab/ConvLab-3) and add an environment variable CONVLAB3_REPOSITORY_BASE_PATH equal to the repository path of of ConvLab3 installation. In config/master.yaml change preprocess and transfer to lusterdatanew or lusterdatasent, before you run main.py.
