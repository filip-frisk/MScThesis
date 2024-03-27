# Filips Framework

Not packaged with setuptools and configurationsfiles, just good ol' python for transparency 

## List of Modules:
    (1) data_viewer eqv. manipulate.py
    (2) model_fitting eqv. fit.py (multiprocessing)
    (3) model_predicting eqv. visualize.py

## List of tools:
    (1) ML_metrics
    (2) 
    (3) 

## data_viewer

### input: 
    (1) ROOT file with cuts applied with HWWAnalysisCode

### parameter:
    (1) list of kinetic variables 
    (2) specify signal and background labels
    (3) PATH to save data in data/ and Histogram in

### output:
    (1) Saved pandas dataframe (with pickle .pkl and in folder), Name: ROOTFILENAME+DATE
    (2) Kinematic Histograms of selected variables in atlasStyle all-in-one and separate (saved as png and in folder)

## model_fitting

### input:
    (1) Saved pandas dataframe (with pickle .pkl)

### parameters:
    (1) specify if binary or multiclass classification
    (2) list of sklearn-models used in analysis #TODO add https://gitlab.cern.ch/ahmarkho/ggffml and https://gitlab.cern.ch/bejaeger/sfusmlkit
    (3) Test to train split
    (4) Number of created models per model type (N of iterations to boxplot) (have in main instead?)
    
### output:
    (1) Models saved as pickle-files (in folder with pickle) , Name: ROOTFILENAME+DATE+MODEL_TYPE+#.pkl
    (2) Class distributions per model y_pred in pkl
    (3) X_test and y_test in pkl (So we can can check channel e t c)


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

# 