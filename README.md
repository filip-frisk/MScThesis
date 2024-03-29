# THIS REPO IS UNDER CONSTRUCTION!
# Filips Framework

My MSc project via KTH.

Not packaged with setuptools and configurationsfiles, just good ol' main.py python for transparency and ease. 

Much effort were made to make this code approachable for MSc students with a pythonic background and basic scikit-learn ML knowledge.

This repository assumes that you arrive with an already cutted MC sample .root file with HWWAnalysisCode.

## Howto

Change UPPERCASE variables in main.py and comment out tools needed.

If I am still not around email me at filiplbfrisk(at)gmail.com and I will gladly help you out.

## List of tools currently built in tools/:
- create_dataframe.py: Using pickle and with channel and variables selections
- create_pretty_histograms.py: Using matplotlib and Atlasify plotting 5 to 95th percentile of the data   
-  metrics.py: Manually implemented ML metrics such as confusion matrix, accuracy, recall, precision e t c not dependent on scikit-learn.

## create_dataframe (DONE)

First it loops through all trees in your rootfile, then it trimmed the trees by channel selection and trimmed the leaves by varaiable selection. Eventually the trimmed root file is save in your data folder (create if you do not have it already). Take a look at line 29 there I applied specific label trimming relevant for my naming convention for my rootfile, you probably need to change this.

### Input
- ROOT file with cuts applied in HWWAnalysisCode

### Parameters 
- `DATA_RELATIVE_FOLDER_PATH` ex. 'data/'
- `DATA_FILENAME_WITHOUT_FILETYPE` ex. 'ntuples-ggFVBF2jet-SF-28Jan24'
- `SIGNAL_CHANNEL` ex. ['VBF']
- `BACKGROUND_CHANNEL` ex. ['WW', 'Zjets', 'ttbar']
- `SELECTED_OTHER_VARIABLES` ex ['eventType','label','eventNumber','weight']
- `SELECTED_PHYSICAL_VARIABLES`ex ['DPhijj', 'mll', 'mT', 'DYjj', 'mjj', 'ptTot', 'mL1J1', 'mL1J2', 'mL2J1', 'mL2J2','ptJ1','ptJ2','ptJ3','METSig']

### output:
    (1) Saved pandas dataframe (with pickle .pkl and in folder)
    
## create_pretty_histograms (NEEDS REBUILD for dynamic scaling and overflow/underflow)

This uses https://pypi.org/project/atlasify/ and matplotlib in a pythonic way..

### input:
    - Saved pandas dataframe

### parameter:
    document as above 
    
### output:
    - Kinematic Histograms of selected variables in atlasStyle all-in-one and separate (saved as png and in folder)

## pre_process_data (DONE)
    - 
### input:
    (1) Saved pandas dataframe (with pickle .pkl)

### parameters:
    (1) specify if class or label classification (if final probability distribution is normalized to 1 or not)
    (2) specify if binary (signal vs bkg) or multiclass (signal vs bkg1 vs bkg2 vs bkg3 ....) 
    (3) test splitt

## output:
    (1) df_train and df_test (including all eventType, label, eventNumber, weights for ) - needed later 
    (2) X_train, y_train_true_labels  : X_test, y_test_true_labels
    (2) 

## model_fitting
    (2) list of sklearn-models used in analysis 
    (3) Test to train split
    (4) Number of created models per model type (N of iterations to boxplot) (have in main instead?)
    
### output:
    (1) Models saved as pickle-files (in folder with pickle) , Name: ROOTFILENAME+DATE+MODEL_TYPE+#.pkl
    (2) Class distributions per model y_pred in pkl
    (3) X_test and y_test in pkl (So we can can check channel e t c)

#TODO add https://gitlab.cern.ch/ahmarkho/ggffml and https://gitlab.cern.ch/bejaeger/sfusmlkit

## model_predicting


### preliminaty calculations:
    (1) class prediction for each model on same test set as txt-files with np.argmax on class probability 

### ML-prf metrics metrics:
    (0) Set predict to np.argmax()
        (1) accuracy = (TP + TN) / (TP + TN + FP + FN)
        (2) precision = quality = TP / (TP + FP) 
        (3) recall = hit rate = TP / (TP + FN) = TPR
        (4) F1 = 2 * (precision * recall) / (precision + recall)
        (5) false alarm = FP / (FP + TN) = FPR
    (00) Set threashold in calc 
        (6) Calculate ROC curve and AUC score : based on hit rate and false alarm with different thresholds

### output:
    ()
    
### plots:
    (2) Boxplot of accuracy of models compared with all ensamble model 
    (2) Boxplot of accuracy of models compared with filipsFramework vs ahmedsFramework vs benjaminFramework
    (3) ROC curve of models compared with filipsFramework vs ahmedsFramework vs benjaminFramework

"""

# FREEZE PIP ENVIROMENT AND ADD ALL IN A REQUIREMENTS FILE WHEN DONE LATER 