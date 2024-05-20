
# Filips Framework

This framework constitutes my MSc project via KTH in the HWW Analysis team.

This is a end-to-end framework where the user arrives with a HWWAnalysisCode cutted MC sample .root file and leaves with prefit and postfit histograms in atlasific style with N probability distributions that can be used as final discrimintant between your signal and background and. 

The framework is essentially a wrapper around scikit-learn using good ol' machine learning model instead of DNN paradigm, even though plain vanilla neural nets are supported. All models in scikit-learn are supported includ sciki-learn wrappers like XGBoost and lightGBM.

This is an alternative ML framework to https://gitlab.cern.ch/bejaeger/sfusmlkit and https://gitlab.cern.ch/ahmarkho/ggffml which uses https://gitlab.cern.ch/fsauerbu/freeforestml which is a Keras DNN framework implementation. 

Analysis I performed with the framework: Same flavour VBF HWW N_jet >= 2 DNN.

## Contact

If I am still not around email me at filiplbfrisk(at)gmail.com or contact via https://www.linkedin.com/in/filipfrisk/ and I will gladly help you out. 

## Installation

The framework is not packaged with setuptools and configurationsfiles, just good ol' main.py python for transparency and ease. Much effort were made to make this code approachable for MSc students with a pythonic background and basic scikit-learn ML knowledge and method. No ROOT tools or C/C++ code are used, all algorithms are built by hand using python libraries.

Run the following commands in your terminal:
```console
python3 -m venv venv-filipsframework
source venv-filipsframework/bin/activate
pip3 install pandas numpy atlasify matplotlib uproot scikit-learn tensorflow imblearn xgboost
mkdir data models plots
```
This repository assumes that you arrive with an already cutted MC sample .root nTuple file generated with the structure HWWAnalysisCode https://gitlab.cern.ch/atlas-physics/higgs/hww/HWWAnalysisCode (as of spring 2024), add it to data/ folder just created.

FYI, Python 3.12.2 was used, use this version if you get version conflicts. Other versions are not supported.

## Howto 

The project is based on UPPERCASE variables in main.py, tools needed are called sequentially. Use docstring to comment out the tools not needed. For example create_dataframe is only needed once per root file. 

Tools included are (sequential):
1. create_dataframe.py
2. create_pretty_histograms.py
3. pre_process_data.py
4. fit_models.py 
5. evaluate_models.py 
6. metrics.py

---

# Detailed documentation per file

## 1. create_dataframe

First it loops through all trees in your rootfile, then it trimmed the trees by channel selection and trimmed the leaves by varaiable selection. Eventually the trimmed root file is save in your data folder. Take a look at line 38 there I applied specific label trimming relevant for my naming convention for my rootfile, you probably need to change this.

### Input
- ROOT file with cuts applied in HWWAnalysisCode

### Parameters 
- `DATA_RELATIVE_FOLDER_PATH: string` ex. 'data/' # create if not available
- `DATA_FILENAME_WITHOUT_FILETYPE: string` ex. 'nTupleVBF2jSF'
- `SIGNAL_CHANNEL: List[strings]` ex. ['VBF']
- `BACKGROUND_CHANNEL: List[strings]` ex. ['WW', 'Zjets', 'ttbar']
- `SELECTED_OTHER_VARIABLES: List[strings]` ex ['eventType','label','eventNumber','weight']
- `SELECTED_PHYSICAL_VARIABLES: List[strings]`ex ['DPhijj', 'mll', 'mT']

### output:
- Selected dataframe with NAME.pkl in folder data/ 

---

## 2. create_pretty_histograms

This function generates pretty histograms for signal and background events, including overflow and underflow handling, and normalizes the weights if specified. The histograms are saved in the specified relative folder path. This uses https://pypi.org/project/atlasify/ and matplotlib in a pythonic way.

### Input:
- `df`: Pandas Dataframe . ex. 'df'
- `plot_variable`: string ex. 'mjj'
- `UNIT`: List[strings] ex. ['GeV', 'rad', 'Unitless']
- `SIGNAL`:List[strings] ex. ['VBF']
- `BACKGROUND`: :List[strings] ex. ['WW', 'Zjets', 'ttbar']
- `CUT`: string ex. 'nTupleVBF2jSF' # just for reference 
- `DATA_FILENAME_WITHOUT_FILETYPE: string` ex. 'nTupleVBF2jSF'
- `OVERFLOW_UNDERFLOW_PERCENTILE`: Dict{string: float} ex. {'lower_bound': 10, 'upper_bound': 90}
- `BINS`: int ex. 19
- `PLOT_RELATIVE_FOLDER_PATH`: string` ex. 'data/' # create if not available
- `PLOT_TYPE`: Type of the plot ('prefit' or 'postfit').
- `SIGNAL_ENVELOPE_SCALE`: The scale for the signal envelope.
- `NORMALIZE_WEIGHTS`: Boolean ex. True 
- `K_FOLD`: int ex. 3
- `EXPERIMENT_ID`: string '240520_I' # Use a syntax like DATE + ID: YYMMDD + rome numericals: I, II, III, IV, V, VI, VII, VIII, IX, X
- `CLASSIFICATION_TYPE`: string ex. binary # only binary and multi_class supported

### Output
- Selected histograms in folder plots/ 

---

## 3. pre_process_data

This function preprocesses the data by splitting it into training and testing datasets based on the specified class weight balancing method. It handles different class weight strategies, checks for duplicates and NaN values, and calculates statistics for the dataframes. It also saves the processed dataframes and their statistics.

### input:
- `df`: Pandas Dataframe . ex. 'df'
- `TRAIN_DATA_SIZE`: float ex. 0.8
- `RANDOM_SEED`: int ex. 42
- `EXPERIMENT_ID`: string ex. '240520_I'
- `DATA_RELATIVE_FOLDER_PATH`: string ex. 'data/'
- `DATA_FILENAME_WITHOUT_FILETYPE`: string  ex. 'nTupleVBF2jSF'
- `K_FOLD`: int . ex. 3
- `CLASS_WEIGHT`: string #Must be one of 'raw', 'MC_EACH_bkg_as_sgn',MC_TOTAL_bkg_as_sgn', 'CW_EACH_bkg_as_sgn', 'CW_TOTAL_bkg_as_sgn'.
- `SIGNAL_CHANNEL`: List[strings], ex. ['VBF']
- `BACKGROUND_CHANNEL`: List[strings],  ex. ['WW', 'Zjets', 'ttbar']

### output:
- As many datasets as in k_fold and saved in folder data/ 

---

## 4. fit_models
This function trains multiple machine learning models using k-fold cross-validation, saves the trained models, and prints the time taken for training each model. The models are saved in the specified relative folder path.


### input:
- `DATA_RELATIVE_FOLDER_PATH`: string ex. 'data/'
- `EXPERIMENT_ID`: string ex. '240520_I'
- `DATA_FILENAME_WITHOUT_FILETYPE`: string  ex. 'nTupleVBF2jSF'
- `K_FOLD`: int . ex. 3
- `CLASS_WEIGHT`: string #Must be one of 'raw', 'MC_EACH_bkg_as_sgn',MC_TOTAL_bkg_as_sgn', 'CW_EACH_bkg_as_sgn', 'CW_TOTAL_bkg_as_sgn'.
- `MODELS`: List[Object[sklearn model]] ex. [NamedClassifier(MLPClassifier(),name = "MLP"),NamedClassifier(XGBClassifier(),name = "XGB")]
- `SELECTED_PHYSICAL_VARIABLES: List[strings]`ex ['DPhijj', 'mll']
- ``MODELS_RELATIVE_FOLDER_PATH:` stringex. 'models/'
- `CLASSIFICATION_TYPE`: string ex. binary # only binary and multi_class supported


### output:
- As many trained models as in MODELS  and saved in folder models/ 


---

## 4. evaluate_models

This function evaluates machine learning models on a test dataset, generates various plots (using create_pretty_histograms.py) including histograms and ROC curves, and calculates performance metrics for each model. It handles different classification types, ensembles model results, and saves the plots and metrics in the specified folder.


### input:
- PLOT_RELATIVE_FOLDER_PATH: String, the relative path to the folder where plots will be saved. ex. 'plots/'
- MODELS_RELATIVE_FOLDER_PATH: String, the relative path to the folder where models are saved. ex. 'models/'
- `EXPERIMENT_ID`: string ex. '240520_I'
- `DATA_RELATIVE_FOLDER_PATH`: string ex. 'data/'
- `DATA_FILENAME_WITHOUT_FILETYPE`: string  ex. 'nTupleVBF2jSF'
- `K_FOLD`: int . ex. 3
- `CLASS_WEIGHT`: string #Must be one of 'raw', 'MC_EACH_bkg_as_sgn',MC_TOTAL_bkg_as_sgn', 'CW_EACH_bkg_as_sgn', 'CW_TOTAL_bkg_as_sgn'.
- `MODELS`: List[Object[sklearn model]] ex. [NamedClassifier(MLPClassifier(),name = "MLP"),NamedClassifier(XGBClassifier(),name = "XGB")]
- `CLASSIFICATION_TYPE`: string ex. binary # only binary and multi_class supported
- `SIGNAL_CHANNEL`: List[strings] ex. ['VBF']
- `BACKGROUND`: :List[strings] ex. ['WW', 'Zjets', 'ttbar']
- `CUT`: string ex. 'nTupleVBF2jSF' # just for reference 
- `SELECTED_PHYSICAL_VARIABLES: List[strings]`ex ['DPhijj', 'mll', 'mT']
### output:
- The plots and metrics in the specified folder plots/

---

## 5. metrics

This module contains functions to calculate various machine learning metrics, including confusion matrix, precision, recall, F1 score, accuracy, false alarm rate, and specificity. It also includes functions for generating ROC curves and handling weighted events. 


### input:
- N/A

### output:
- Auxiliary function providing various metrics and ROC curve to the framework. 
---
