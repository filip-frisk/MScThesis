import numpy as np

def accuracy_weighted(y_true, y_pred, sample_weight = None): # without confusion matrix
  assert len(y_true) == len(y_pred)
  if sample_weight is None:
    return np.sum([1 if truth == pred else 0 for truth,pred in zip(y_true,y_pred)])/len(y_true)
  else:
    assert len(y_true) == len(y_pred) == len(sample_weight)
    return np.sum([1*weight if truth == pred else 0 for truth,pred,weight in zip(y_true,y_pred,sample_weight)])/np.sum(sample_weight)

def confusion_matrix(y_true, y_pred): # sgn = 1 and bkg = 0
    TP = sum((y_true == 1) & (y_pred == 1))
    TN = sum((y_true == 0) & (y_pred == 0))
    FP = sum((y_true == 0) & (y_pred == 1))
    FN = sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN

# TODO
def confusion_matrix_multiclass():
    pass

# TODO
def confusion_matrix_weighted():
    pass

def accuracy(TP, TN, FP, FN): 
    return (TP + TN) / (TP + TN + FP + FN)

def precision(TP, FP): 
    return TP / (TP + FP)

def recall(TP, FN): # hit rate or TPR or sensitivity
    return TP / (TP + FN)

def false_alarm_rate(FP, TN): # Naming:FPR  or 1 - specificity
    #return FP / (FP + TN)
    return 1 - specificity(TN, FP)

def specificity(TN, FP): 
    return TN / (TN + FP)

def f1_score(TP, TN, FP, FN): 
    return 2 * precision(TP, FP) * recall(TP, FN) / ((precision(TP, FP) + recall(TP, FN)))

def predict(y_prob):
    return np.argmax(y_prob, axis=1)

def predict_threshold(y_prob, threshold):
    return np.array([1 if y_prob[i, 1] > threshold else 0 for i in range(len(y_prob))])

def roc_curve(y_true, y_prob):
    thresholds = np.linspace(0, 1, 10)
    tpr = np.zeros(len(thresholds))
    fpr = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        y_pred = predict_threshold(y_prob, threshold)
        TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
        tpr[i] = recall(TP, FN)
        fpr[i] = false_alarm_rate(FP, TN)
    return tpr, fpr

def auc(tpr, fpr): #not sure why it is negative
    return -1 * np.trapz(tpr, fpr)

# TODO
def roc_curve_weighted():
    pass

# TODO
def roc_curve_multiclass():
    pass
   
if __name__ == '__main__':
    print("Uncomment the tests to run them")

    """Test the ROC curve
    # read pickle file data/y_test.pkl and convert to numpy arrays 
    import pickle
    
    y_true_full = np.array(pickle.load(open('data/y_test.pkl', 'rb')))
    y_prob_rf_full = np.array(pickle.load(open('data/y_prob_rf.pkl', 'rb')))
    y_prob_gbdt_full = np.array(pickle.load(open('data/y_prob_gbdt.pkl', 'rb')))

    # Generate random indices for sampling
    indices = np.random.choice(np.arange(len(y_true_full)), size=10000, replace=False)

    # Sample the arrays using the same indices
    y_true = y_true_full[indices]
    y_prob_rf = y_prob_rf_full[indices]
    y_prob_gbdt = y_prob_gbdt_full[indices]

    tpr_rf, fpr_rf = roc_curve(y_true_full, y_prob_rf_full)
    tpr_gbdt, fpr_gbdt = roc_curve(y_true_full, y_prob_gbdt_full)

    print('AUC RF: {:.3f}'.format(auc(tpr_rf, fpr_rf)))
    print('AUC GBDT: {:.3f}'.format(auc(tpr_gbdt, fpr_gbdt)))
    
    
    import matplotlib.pyplot as plt

    #plt.xlim(0, 1)
    #plt.ylim(0, 1)
    plt.plot(fpr_rf, tpr_rf, label='random forest')
    plt.plot(fpr_gbdt, tpr_gbdt, label='gradient boosting')
    plt.plot([0, 1], [0, 1], 'k:', label='50/50 choice')
    plt.plot([0, 0, 1], [0, 1, 1], 'r:', label='Perfect classifier')
    plt.xlabel('False Signal Rate (FSR)')
    plt.ylabel('True Signal Rate (TSR)')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    """
    
    """ Test on accuracy_weighted
    y_true = [0,0,1,1,1,1]
    y_pred = [0,1,1,1,0,1]
    sample_weight = [1,2,1,0.5,1,2]

    print(accuracy_weighted(y_true,y_pred))
    print(accuracy_weighted(y_true,y_pred,sample_weight)) 

    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_true, y_pred))
    print(accuracy_score(y_true, y_pred, sample_weight=sample_weight))
    """

    """ Test on actual data 
    import pickle
    y_true = np.array(pickle.load(open('data/y_test.pkl', 'rb')))
    y_prob = np.array(pickle.load(open('data/y_prob_rf.pkl', 'rb')))

    
    y_pred = predict(y_prob)

    y_pred_threshold = predict_threshold(y_prob, 0.8)

    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)


    binary_confusion_matrix = f" ""Confusion Matrix:
                Predicted:sgn  Predicted:bkg
        Actual:sgn    {TP}          {FN}
        Actual:bkg    {FP}          {TN}
    

    print(binary_confusion_matrix) 

    print('accuracy: {:.1f}'.format(accuracy(TP, TN, FP, FN)))
    print('precision: {:.1f}'.format(precision(TP, FP)))
    print('recall: {:.1f}'.format(recall(TP, FN)))
    print('f1_score: {:.1f}'.format(f1_score(TP, TN, FP, FN)))
    print('false_alarm_rate: {:.1f}'.format(false_alarm_rate(FP, TN)))

    """

    """Test of predict and predict_threshold binary classification
    y_true = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0])
    
    # generate a 10x2 matrix where each row sum to 1
    np.random.seed(3)
    y_prob_binary = np.random.dirichlet(np.ones(2),size=10)
    
    print(np.round(y_prob_binary, 2))

    print(predict(y_prob_binary))
    print(predict_threshold(y_prob_binary, 0.3))

    y_prob_multiclass = np.random.dirichlet(np.ones(4),size=10)
           
    """
    
    """Tests of metrics binary classification
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    
    print(TP, TN, FP, FN)
    print('accuracy: {:.1f}'.format(accuracy(TP, TN, FP, FN)))
    print('precision: {:.1f}'.format(precision(TP, FP)))
    print('recall: {:.1f}'.format(recall(TP, FN)))
    print('f1_score: {:.1f}'.format(f1_score(TP, TN, FP, FN)))
    print('false_alarm_rate: {:.1f}'.format(false_alarm_rate(FP, TN)))
    """
    