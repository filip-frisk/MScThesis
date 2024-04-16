import pickle
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .create_pretty_histograms import plot_one_physical_variable


def evaluate_models(PLOT_RELATIVE_FOLDER_PATH,MODELS_RELATIVE_FOLDER_PATH,EXPERIMENT_ID,DATA_RELATIVE_FOLDER_PATH,DATA_FILENAME_WITHOUT_FILETYPE,K_FOLD,CLASS_WEIGHT,MODELS,CLASSIFICATION_TYPE,SIGNAL_CHANNEL,BACKGROUND_CHANNEL,CUT,SELECTED_PHYSICAL_VARIABLES):
    
    plot_variables = []
    for K_FOLD in range(1,K_FOLD+1):
        print(f"Starting evaluation for K-Fold: {K_FOLD}\n")
        
        # Load test data
        os.chdir(DATA_RELATIVE_FOLDER_PATH+EXPERIMENT_ID)
        
        TEST_DATA_FILE_PATH = f"{DATA_FILENAME_WITHOUT_FILETYPE}_fold{K_FOLD}_{CLASS_WEIGHT}_test.pkl"
        
        with open(TEST_DATA_FILE_PATH, 'rb') as f:
            df_test = pickle.load(f)

        print(f'Loaded test data for K-Fold: {K_FOLD}\n')

        # change back to the main directory
        os.chdir('../../..')

        # Load model for K-fold
        os.chdir(MODELS_RELATIVE_FOLDER_PATH+EXPERIMENT_ID)

        output_variables = []
        
        for model in MODELS:
            print(f"Starting evaluation for model: {model.__class__.__name__} for K-Fold:{K_FOLD}\n")
    
            
            MODEL_FILE_PATH = f'{DATA_FILENAME_WITHOUT_FILETYPE}_fold{K_FOLD}_{CLASS_WEIGHT}_train_{CLASSIFICATION_TYPE}_{model.__class__.__name__}.pkl'
            with open(MODEL_FILE_PATH, 'rb') as f:
                model = pickle.load(f)
            
            # evaluate the model
            if CLASSIFICATION_TYPE == 'binary':
                df_distribution = pd.DataFrame(model.predict_proba(df_test[SELECTED_PHYSICAL_VARIABLES]), columns=model.classes_)
                
                output_variable = f'MVAOutput_fold{K_FOLD}_{model.__class__.__name__}' # VBF-like
            
            elif CLASSIFICATION_TYPE == 'multi-class':
                pass

            else:
                raise ValueError(f"Classification type: {CLASSIFICATION_TYPE} not supported")
            
            output_variables.append(output_variable)
            df_test = df_test.reset_index(drop=True)
            df_test[output_variable] = df_distribution['Signal']    
            
            print(f"Finished evaluation for model: {model.__class__.__name__}\n")
        
        ### Ensamble results ###
    
        # mean
        
        df_test[f'MVAOutput_fold{K_FOLD}_Mean_Ensamble'] = df_test[output_variables].mean(axis=1)

        # median
        df_test[f'MVAOutput_fold{K_FOLD}_Median_Ensamble'] = df_test[output_variables].median(axis=1)

        
        output_variables.append(f'MVAOutput_fold{K_FOLD}_Mean_Ensamble')
        output_variables.append(f'MVAOutput_fold{K_FOLD}_Median_Ensamble')

        # Save plot variables 
        plot_variables.append(output_variables)

        # change back to the main directory
        os.chdir('../..')
            
    ######################################### SIGNAL LIKE PLOT #########################################
    ### PLOT SETTINGS ###
    PLOT_TYPE = 'postfit' # 'prefit', 'postfit'
    UNIT = ''
    OVERFLOW_UNDERFLOW_PERCENTILE = {'lower_bound': 5, 'upper_bound': 95}
    BINS = 7
    # plot the results
    
    SIGNAL_ENVELOPE_SCALE = 500 # easier to guess than to scale dynamically
    NORMALIZE_WEIGHTS = False
    
    # plot all folds
    for fold in plot_variables: 
        plot_variables = fold 
        
        # plot all variables
        for variables in plot_variables:
            plot_one_physical_variable(df_test, variables, UNIT, SIGNAL_CHANNEL , BACKGROUND_CHANNEL, CUT,DATA_FILENAME_WITHOUT_FILETYPE,OVERFLOW_UNDERFLOW_PERCENTILE,BINS,PLOT_RELATIVE_FOLDER_PATH, PLOT_TYPE,SIGNAL_ENVELOPE_SCALE,NORMALIZE_WEIGHTS)
            
    ######################################### ML metric PLOTS #########################################

    if CLASSIFICATION_TYPE == 'binary':
        
        # create new compact dataframe for ROC curve
        df_test_ML_Metrics = df_test[['eventType','label','eventNumber','weight'] + output_variables]

        ############### ML SCORECARD ###############

        from tools.binary_metrics import predict_threshold,confusion_matrix,precision,recall,f1_score,accuracy,false_alarm_rate,specificity
            
        
        THRESHOLD = 0.5
        
        for output_variable in output_variables:
            MVAOutput = output_variable
            New_MVAOutput_prediction_column = MVAOutput + "_prediction"
            
            # get predictions
            df_test_ML_Metrics[New_MVAOutput_prediction_column] = predict_threshold(df_test_ML_Metrics, MVAOutput,THRESHOLD)
            
            y_true_col = 'eventType'
            y_pred_col = New_MVAOutput_prediction_column
            
            # get confusion matrix 
            TP, TN, FP, FN = confusion_matrix(df_test_ML_Metrics, y_true_col, y_pred_col)

            # Plotting a minimalistic confusion matrix
            confusion_matrix_values = np.array([[TP, FP], [FN, TN]])
            labels = [['TP', 'FP'], ['FN', 'TN']]  # Labels for each cell
            
            fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8,6), height_ratios = [5, 1])
            cax = ax_top.matshow(confusion_matrix_values, cmap='cool')

            ax_top.title.set_text(f'ML Metrics Score card on {df_test_ML_Metrics.shape[0]} MC samples with {THRESHOLD*100:.0f}% as the signal threshold \n') # not weighted resutslts 
    
            fig.colorbar(cax)
            
        
            ax_top.set_xticklabels(['', 'Signal', 'Background'])
            ax_top.set_yticklabels(['', 'Signal', 'Background'])
            
            # Add xlabel on top
            ax_top.xaxis.set_label_position('top')
            ax_top.set_xlabel('Actual',weight = 'bold')
            ax_top.set_ylabel(f'Predicted by {output_variable}',weight = 'bold')
                
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
            ax_bottom.text(0.5, 0.75, f"Metrics:\n Precision: TP/(TP+FP) = {round(MVAOutput_precision,2)} \n Recall: TP/(TP+FN) = {round(MVAOutput_recall,2)}\n F1 Score: Harmonic_mean(Precision,Recall) = {round(MVAOutput_f1_score,2)} \n Accuracy: (TP+TN)/(TP+FP+FN+TN)  = {round(MVAOutput_accuracy,2)}\n Specificity: TN/(TN+FP) = {round(MVAOutput_specificity,2)}\n False Alarm Rate: FP/(FP+TN) = {round(MVAOutput_false_alarm_rate,2)} ", ha='center', va='center', color='black')
            
            # tight layout    
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.tight_layout()
            
            
            os.chdir(PLOT_RELATIVE_FOLDER_PATH)

            plt.savefig(f'{PLOT_TYPE}_metricsscorecard_{DATA_FILENAME_WITHOUT_FILETYPE}_{output_variable}.png',dpi = 600) #

            print(f"Saved {PLOT_TYPE}_metricsscorecard_{DATA_FILENAME_WITHOUT_FILETYPE}_{output_variable}.png in plots/ .")

            os.chdir('..')

            plt.close()
            
            
    
    else:
        print("Multi-class classification not supported yet for ML metric scorecard")
        pass
    
    
        ############### ROC & AUC ###############
    
    if CLASSIFICATION_TYPE == 'binary':
        from tools.binary_metrics import roc_curve

        POINTS_IN_ROC_CURVE = 10
        THRESHOLDS = np.linspace(0, 1, POINTS_IN_ROC_CURVE) # 
        
        y_true_col = 'eventType'
        
        for output_variable in output_variables:
            y_pred_col = output_variable
            tpr, fpr = roc_curve(df_test_ML_Metrics, y_true_col, y_pred_col, THRESHOLDS)
            

            # calculate area under the curve or AUC

            sorted_indices = np.argsort(fpr) # X axis 7 fpr in ascending order to not get negative values

            print(f"{output_variable} tpr: {tpr[sorted_indices]}\n")
            print(f"{output_variable} fpr: {fpr[sorted_indices]}\n")

            AUC = np.trapz(tpr[sorted_indices], fpr[sorted_indices])


            
            plt.plot(fpr, tpr, label=f'{output_variable}, AUC = {AUC:.2f}')
            
        # add random classifier as comparison
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random guess')
        
        # add perfect classifier as comparison
        plt.plot([0, 0, 1], [0, 1, 1], linestyle='--', lw=2, color='b', label='Perfect classifier')
            
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
            
        plt.legend()
        
        os.chdir(PLOT_RELATIVE_FOLDER_PATH)

        plt.savefig(f'{PLOT_TYPE}_ROC_AUC_{DATA_FILENAME_WITHOUT_FILETYPE}.png',dpi = 600) #

        print(f"Saved {PLOT_TYPE}_ROC_AUC_{DATA_FILENAME_WITHOUT_FILETYPE}.png in plots/ .")

        os.chdir('..')

        plt.close()
    else:
        print("Multi-class classification not supported yet for ROC & AUC")
        pass



    

    

    
    ######################################### ML METRIC PLOTS #########################################
    
    
    

            


