import pickle
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .create_pretty_histograms import create_pretty_histograms

#Sklearn has a lot of irrelevant warnings that we want to ignore
import warnings
warnings.filterwarnings("ignore")

from typing import List

def evaluate_models(PLOT_RELATIVE_FOLDER_PATH: str,
                    MODELS_RELATIVE_FOLDER_PATH: str,
                    EXPERIMENT_ID: str,
                    DATA_RELATIVE_FOLDER_PATH: str,
                    DATA_FILENAME_WITHOUT_FILETYPE: str,
                    K_FOLD: int,
                    CLASS_WEIGHT: str,
                    MODELS: List[object],
                    CLASSIFICATION_TYPE: str,
                    SIGNAL_CHANNEL: str,
                    BACKGROUND_CHANNEL: List[str],
                    CUT: str,
                    SELECTED_PHYSICAL_VARIABLES: List[str]
                    ) -> None:
    
    
    print(f"Starting evaluation for K-Fold number {K_FOLD}\n")
    
    #### Load test data ####
    os.chdir(DATA_RELATIVE_FOLDER_PATH+EXPERIMENT_ID)
    
    TEST_DATA_FILE_PATH = f"{DATA_FILENAME_WITHOUT_FILETYPE}_fold{K_FOLD}_{CLASS_WEIGHT}_test.pkl"
    
    with open(TEST_DATA_FILE_PATH, 'rb') as f: #unique for each fold
        df_test = pickle.load(f)

    os.chdir('../..')

    ### Load model for K-fold ###
    os.chdir(MODELS_RELATIVE_FOLDER_PATH+EXPERIMENT_ID)

    model_names = []
    
    for model in MODELS:
        print(f"Starting evaluation for model: {model.name} for K-Fold:{K_FOLD}\n")
        
        MODEL_FILE_PATH = f'{DATA_FILENAME_WITHOUT_FILETYPE}_fold{K_FOLD}_{CLASS_WEIGHT}_train_{CLASSIFICATION_TYPE}_{model.name}.pkl'
        with open(MODEL_FILE_PATH, 'rb') as f:
            model = pickle.load(f)
        
        if CLASSIFICATION_TYPE == 'binary':
            
            if model.name == 'XGB':
                df_test['eventType'] = df_test['eventType'].map({'Signal': 0,'Background': 1})
                    
            df_distribution = pd.DataFrame(model.predict_proba(df_test[SELECTED_PHYSICAL_VARIABLES]), columns=model.classes_)
            
            if model.name == 'XGB':
                df_test['eventType'] = df_test['eventType'].map({0: 'Signal',1: 'Background'})
                df_distribution.columns = ['Signal','Background']
            
            print(f"Model classes example: {model.name} \n {df_distribution.head()}\n")
            model_name = f'MVAOutput_fold_{K_FOLD}_{CLASSIFICATION_TYPE}_{model.name}' # VBF-like
        
        elif CLASSIFICATION_TYPE == 'multi_class':

            if model.name == 'XGB':
                df_test['label'] = df_test['label'].map({'VBF': 0,'WW': 1,'Zjets': 2,'ttbar': 3})  

            df_distribution = pd.DataFrame(model.predict_proba(df_test[SELECTED_PHYSICAL_VARIABLES]), columns=model.classes_)
            
            if model.name == 'XGB':
                df_test['label'] = df_test['label'].map({0: 'VBF',1: 'WW',2: 'Zjets',3: 'ttbar'})
                df_distribution.columns = ['VBF','WW','Zjets','ttbar']
            
            print(f"Model classes example: {model.name} \n {df_distribution.head()}\n")
            model_name = f'MVAOutput_fold_{K_FOLD}_{CLASSIFICATION_TYPE}_{model.name}' # VBF-like

        else:
            raise ValueError(f"Classification type: {CLASSIFICATION_TYPE} not supported")
        
        model_names.append(model_name)
        df_test = df_test.reset_index(drop=True)

        if CLASSIFICATION_TYPE == 'binary':
            df_test[model_name] = df_distribution['Signal']  
                
        elif CLASSIFICATION_TYPE == 'multi_class':
            df_test[model_name] = df_distribution['VBF']
        else:
            raise ValueError(f"Classification type: {CLASSIFICATION_TYPE} not supported. Choose 'binary' or'multi-class' ")
        
        print(f"Finished evaluation for model: {model.name} for K-Fold:{K_FOLD}\n")
    
    ### Ensemble Averaging results in fold ###

    # Do not ensamble with compare with MLPClassifier
    models_ensamble = [model_name for model_name in model_names if not model_name.endswith('BENCHMARK')]        

    # mean
    df_test[f'MVAOutput_fold_{K_FOLD}_{CLASSIFICATION_TYPE}_Mean_Ensamble'] = df_test[models_ensamble].mean(axis=1)

    # Sample Weighted Mean
    #df_test[f'MVAOutput_fold_{K_FOLD}_{CLASSIFICATION_TYPE}_SampleWeightedMean_Ensamble'] = df_test[models_ensamble].apply(lambda x: np.average(x, weights=df_test['weight']), axis=1)

    # Class Weighted Mean??
    
    # Median
    df_test[f'MVAOutput_fold_{K_FOLD}_{CLASSIFICATION_TYPE}_Median_Ensamble'] = df_test[models_ensamble].median(axis=1)

    # Maximum
    df_test[f'MVAOutput_fold_{K_FOLD}_{CLASSIFICATION_TYPE}_Max_Ensamble'] = df_test[models_ensamble].max(axis=1)

    

    model_names.append(f'MVAOutput_fold_{K_FOLD}_{CLASSIFICATION_TYPE}_Mean_Ensamble')
    model_names.append(f'MVAOutput_fold_{K_FOLD}_{CLASSIFICATION_TYPE}_Median_Ensamble')
    model_names.append(f'MVAOutput_fold_{K_FOLD}_{CLASSIFICATION_TYPE}_Max_Ensamble')

    # change back to the main directory from the model directory
    os.chdir('../..')

    ######################################### SIGNAL LIKE PLOT #########################################

    ### PLOT SETTINGS ###
    PLOT_TYPE = 'postfit' # 'prefit', 'postfit'
    UNIT = ''
    OVERFLOW_UNDERFLOW_PERCENTILE = {'lower_bound': 5, 'upper_bound': 95}
    BINS = 10 # 9*1/10 = [0.90,1.0] last bin with 90% confidence  
    # plot the results
    
    SIGNAL_ENVELOPE_SCALE = 5000 # easier to guess than to scale dynamically
    NORMALIZE_WEIGHTS = False
    
    # plot all models in all folds
    for model in model_names:     
        create_pretty_histograms(df_test, model, UNIT, SIGNAL_CHANNEL , BACKGROUND_CHANNEL, CUT,DATA_FILENAME_WITHOUT_FILETYPE,OVERFLOW_UNDERFLOW_PERCENTILE,BINS,PLOT_RELATIVE_FOLDER_PATH, PLOT_TYPE,SIGNAL_ENVELOPE_SCALE,NORMALIZE_WEIGHTS,K_FOLD,EXPERIMENT_ID,CLASSIFICATION_TYPE)
        
    ######################################### ML metric PLOTS #########################################

    # create new compact dataframe for ROC curve
    df_test_ML_Metrics = df_test[['eventType','label','eventNumber','weight'] + model_names]

    ############### ML SCORECARD ###############

    from tools.metrics import predict_threshold,confusion_matrix,precision,recall,f1_score,accuracy,false_alarm_rate,specificity,confusion_matrix_event_weighted
        
    THRESHOLD = (BINS-1)* 1/BINS # last bin with 90% confidence
    
    for model_name in model_names:
        MVAOutput = model_name
        New_MVAOutput_prediction_column = MVAOutput + "_prediction"
        
        # get predictions
        df_test_ML_Metrics[New_MVAOutput_prediction_column] = predict_threshold(df_test_ML_Metrics, MVAOutput,THRESHOLD)
        
        y_true_col = 'eventType'
        y_pred_col = New_MVAOutput_prediction_column
        
        # get confusion matrix 
        TP, TN, FP, FN = confusion_matrix(df_test_ML_Metrics, y_true_col, y_pred_col)
        #TP, TN, FP, FN = confusion_matrix_event_weighted(df_test_ML_Metrics, y_true_col, y_pred_col, 'weight')

        # Plotting a minimalistic confusion matrix
        confusion_matrix_values = np.array([[TP, FP], [FN, TN]])
        labels = [['TP', 'FP'], ['FN', 'TN']]  # Labels for each cell
        
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8,6), height_ratios = [5, 1])
        cax = ax_top.matshow(confusion_matrix_values, cmap='cool')

        ax_top.title.set_text(f'{df_test_ML_Metrics.shape[0]:.0f} MC samples \n Threshold: {THRESHOLD*100:.0f}%') # not weighted resutslts 

        fig.colorbar(cax)
    
        ax_top.set_xticklabels(['', 'Signal', 'Background'])
        ax_top.set_yticklabels(['', 'Signal', 'Background'])
        
        # Add xlabel on top
        ax_top.xaxis.set_label_position('top')
        ax_top.set_xlabel('Actual',weight = 'bold')

        # plot_variable looks like: 
        # "MVAOutput_fold_{K_FOLD}_{CLASSIFICATION_TYPE}_Mean_Ensamble" OR
        #     0        1     2           3                4     5
        # MVAOutput_fold_{K_FOLD}_{CLASSIFICATION_TYPE}_model
        #    0       1     2           3                 4     
        parts = model_name.split("_")
        try:
            clean_label = f"{parts[0]}_{parts[3]}_{parts[4]}_{parts[5]}"
        except IndexError:
            clean_label = f"{parts[0]}_{parts[3]}_{parts[4]}"

        ax_top.set_ylabel(f'Predicted by {clean_label}',weight = 'bold')
            
        for (i, j), val in np.ndenumerate(confusion_matrix_values):
            label = labels[i][j]
            ax_top.text(j, i, f'{label} = {val}', ha='center', va='center', color='black')

        ### Calculate metrics ###
        MVAOutput_precision = precision(TP, FP)
        MVAOutput_recall = recall(TP, FN)
        MVAOutput_f1_score = f1_score(TP, FP, FN)
        
        MVAOutput_accuracy = accuracy(TP, TN, FP, FN)
        MVAOutput_false_alarm_rate = false_alarm_rate(FP, TN)
        MVAOutput_specificity = specificity(FP, TN)

        # Add metrics to the bottom plot
        ax_bottom.axis('off')
        ax_bottom.text(0.5, 0.75, f"Metrics:\n Precision: TP/(TP+FP) = {round(MVAOutput_precision,2)} \n True Positive Rate/Signal efficiency: TP/(TP+FN) = {round(MVAOutput_recall,2)}\n F1 Score: Harmonic_mean(Precision,Recall) = {round(MVAOutput_f1_score,2)} \n Accuracy: (TP+TN)/(TP+FP+FN+TN)  = {round(MVAOutput_accuracy,2)}\n Specificity: TN/(TN+FP) = {round(MVAOutput_specificity,2)}\n False Positive Rate/Background Efficiency: FP/(FP+TN) = {round(MVAOutput_false_alarm_rate,2)} ", ha='center', va='center', color='black')
        
        # tight layout    
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
        
        os.chdir(PLOT_RELATIVE_FOLDER_PATH+EXPERIMENT_ID)

        plt.savefig(f'{PLOT_TYPE}_metricsscorecard_{DATA_FILENAME_WITHOUT_FILETYPE}_{model_name}.png',dpi = 600) #

        print(f"Saved {PLOT_TYPE}_metricsscorecard_{DATA_FILENAME_WITHOUT_FILETYPE}_{model_name}.png in plots/ .")

        os.chdir('../..')

        plt.close()
            
    ############### ROC & AUC ###############
    
    from tools.metrics import roc_curve

    POINTS_IN_ROC_CURVE = 10
    THRESHOLDS = np.linspace(0, 1, POINTS_IN_ROC_CURVE) # 
    
    y_true_col = 'eventType'
    
    for model_name in model_names:
        y_pred_col = model_name
        tpr, fpr = roc_curve(df_test_ML_Metrics, y_true_col, y_pred_col, THRESHOLDS)
        
        # calculate area under the curve or AUC

        sorted_indices = np.argsort(fpr) # X axis fpr in ascending order to not get negative values

        AUC = np.trapz(tpr[sorted_indices], fpr[sorted_indices])

        # Fix labels in the plot
        if model_name.endswith('Ensamble'):
            model_name = model_name.split('_')[-2] + " " + model_name.split('_')[-1]
        else:
            model_name = model_name.split('_')[-1]



        plt.plot(fpr, tpr, label=f'{model_name}, AUC = {AUC:.2f}')
        
    # add random classifier as comparison
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess, AUC = 0.5')
    
    # add perfect classifier as comparison
    plt.plot([0, 0, 1], [0, 1, 1], linestyle='--', lw=2, color='b', label='Perfect classifier, AUC = 1.0')
        
    plt.xlabel('False Positive Rate/Background Efficiency')
    plt.ylabel('True Positive Rate/Signal Efficiency')
        
    # add legend above the plot window

    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
    #plt.tight_layout()
    plt.legend()
    os.chdir(PLOT_RELATIVE_FOLDER_PATH + EXPERIMENT_ID)

    plt.savefig(f'{PLOT_TYPE}_ROC_AUC_{DATA_FILENAME_WITHOUT_FILETYPE}_fold{K_FOLD}_{CLASSIFICATION_TYPE}.png',dpi = 600) #

    print(f"Saved {PLOT_TYPE}_ROC_AUC_{DATA_FILENAME_WITHOUT_FILETYPE}_fold{K_FOLD}_{CLASSIFICATION_TYPE}.png in plots/ .")

    os.chdir('../..')

    plt.close()