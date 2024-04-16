import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None # Weird irrelevant warning

############### NEEDED FOR ALL BINARY METRICS ###############

def confusion_matrix(df, y_true_col, y_pred_col): 
    TP = sum((df[y_true_col] == 'Signal') & (df[y_pred_col] == 'Signal'))
    TN = sum((df[y_true_col] == 'Background') & (df[y_pred_col] == 'Background'))
    FP = sum((df[y_true_col] == 'Background') & (df[y_pred_col] == 'Signal'))
    FN = sum((df[y_true_col] == 'Signal') & (df[y_pred_col] == 'Background'))
    return TP, TN, FP, FN

def predict_threshold(df,df_column, THRESHOLD):
    return df[df_column].apply(lambda x: 'Signal' if x >= THRESHOLD else 'Background')

############### ML SCORECARD ###############

#1e-10 is added to avoid division by zero and robustness

def precision(TP, FP): 
    return TP / (TP + FP + 1e-10)

def recall(TP, FN): # hit rate or TPR or sensitivity
    return TP / (TP + FN)

def f1_score(TP, FP, FN): 
    return 2 * precision(TP, FP) * recall(TP, FN) / ((precision(TP, FP) + recall(TP, FN) + 1e-10))

def accuracy(TP, TN, FP, FN): 
    return (TP + TN) / (TP + TN + FP + FN + 1e-10)

def false_alarm_rate(FP, TN): # Naming:FPR  or 1 - specificity
    return FP / (FP + TN + 1e-10)

def specificity(FP, TN): 
    return TN / (TN + FP + 1e-10)

############### ROC & AUC  ###############

def roc_curve(df, y_true_col, y_pred_col, THRESHOLDS):    
    tpr = np.zeros(len(THRESHOLDS))
    fpr = np.zeros(len(THRESHOLDS))
    for i, threshold in enumerate(THRESHOLDS):
        df['tmp_prediction_column'] = predict_threshold(df, y_pred_col, threshold)
        TP, TN, FP, FN = confusion_matrix(df, y_true_col, 'tmp_prediction_column')
        print(TP, TN, FP, FN)
        tpr[i] = recall(TP, FN)
        fpr[i] = false_alarm_rate(FP, TN)

    return tpr, fpr

################# Weighted metrics #################
# use if you use weighted samples when training, I do not use 


if __name__ == '__main__':
  pass 