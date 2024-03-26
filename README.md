# Filips Framework

## data-viewer-module

### input: 
    (1) ROOT file with cuts applied with HWWAnalysisCode

### parameter:
    (1) list of variables in analysis
    (2) specify signal and background labels
    (3) PATH to save data and Histogram

###  output:
    (1) Kinematic Histograms of selected variables in atlasStyle all-in-one and separate (saved as png in folder)

##
    
parameters:
    (3) specify if binary or multiclass classification
    (4) list of sklearn-models used in analysis
    (5) test to train split
    (6) iterations of experiment / k-fold cross validation if you like


output:
    (1) Models saved as pickle-files
    
    (1) class probability distributions for each model on same test set (as txt-files)
    
preliminaty calculations:
    (1) class prediction for each model on same test set as txt-files with np.argmax on class probability
    (1) from class prediction we can calculate TP, FP, TN, FN (confusion matrix)

ML-prf metrics metrics:
    (1) accuracy = (TP + TN) / (TP + TN + FP + FN)
    (2) precision = TP / (TP + FP)
    (3) recall = TP / (TP + FN)
    (4) F1 = 2 * (precision * recall) / (precision + recall)
    (5) ROC curve and AUC score 

plots:
    (2) Boxplot of accuracy of models compared with all ensamble model 
    (2) Boxplot of accuracy of models compared with filipsFramework vs ahmedsFramework vs benjaminFramework
    (3) ROC curve of models compared with filipsFramework vs ahmedsFramework vs benjaminFramework

"""

# 