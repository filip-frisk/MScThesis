# THIS REPO IS UNDER CONSTRUCTION!
# Filips Framework

This framework constitutes my MSc project via KTH in the HWW Analysis team.

This is a end-to-end framework where the user arrives with a HWWAnalysisCode cutted MC sample .root file and leaves with prefit and postfit histograms in atlasific style with N probability distributions that can be used as final discrimintant between your signal and background and. 

The framework is essentially a wrapper around scikit-learn using good ol' machine learning model instead of DNN paradigm, even though plain vanilla neural nets are supported.

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
This repository assumes that you arrive with an already cutted MC sample .root file with HWWAnalysisCode, add it to data/ folder just created.

FYI, Python 3.12.2 was used, use this version if you get version conflicts. 

## Howto 

The project is based on UPPERCASE variables in main.py, tools needed are called sequentially. Use docstring to comment out the tools not needed. For example create_dataframe is only needed once per root file. 

Tools included are (sequential):
1. create_dataframe.py
2. create_pretty_histograms.py
3. pre_process_data.py
4. evaluate_models.py (also uses 2. and 5.)
5. metrics.py

---

# Detailed documentation per file

## 1. create_dataframe

First it loops through all trees in your rootfile, then it trimmed the trees by channel selection and trimmed the leaves by varaiable selection. Eventually the trimmed root file is save in your data folder. Take a look at line 38 there I applied specific label trimming relevant for my naming convention for my rootfile, you probably need to change this.

### Input
- ROOT file with cuts applied in HWWAnalysisCode

### Parameters 
- `DATA_RELATIVE_FOLDER_PATH: string` ex. 'data/'
- `DATA_FILENAME_WITHOUT_FILETYPE: string` ex. 'ntuples-ggFVBF2jet-SF-28Jan24'
- `SIGNAL_CHANNEL: List[strings]` ex. ['VBF']
- `BACKGROUND_CHANNEL: List[strings]` ex. ['WW', 'Zjets', 'ttbar']
- `SELECTED_OTHER_VARIABLES: List[strings]` ex ['eventType','label','eventNumber','weight']
- `SELECTED_PHYSICAL_VARIABLES: List[strings]`ex ['DPhijj', 'mll', 'mT', 'DYjj', 'mjj', 'ptTot', 'mL1J1', 'mL1J2', 'mL2J1', 'mL2J2','ptJ1','ptJ2','ptJ3','METSig']

### output:
- Selected dataframe with NAME.pkl in folder data/ 

---

## 2. create_pretty_histograms

This uses https://pypi.org/project/atlasify/ and matplotlib in a pythonic way.

### input:
### parameter:
### output:

---

## 3. pre_process_data

### input:
### parameter:
### output:

---

## 4. evaluate_models

### input:
### parameter:
### output:

---

## 5. metrics

### input:
### parameter:
### output:

---
