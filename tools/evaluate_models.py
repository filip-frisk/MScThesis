import pickle
import os 
import pandas as pd

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

            elif CLASSIFICATION_TYPE == 'multi-label':
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
    plot_variables = plot_variables[0] # TODO fix later when scaled up with multiple folds
    SIGNAL_ENVELOPE_SCALE = 500 # easier to guess than to scale dynamically

    NORMALIZE_WEIGHTS = False

    for variables in plot_variables:
        plot_one_physical_variable(df_test, variables, UNIT, SIGNAL_CHANNEL , BACKGROUND_CHANNEL, CUT,DATA_FILENAME_WITHOUT_FILETYPE,OVERFLOW_UNDERFLOW_PERCENTILE,BINS,PLOT_RELATIVE_FOLDER_PATH, PLOT_TYPE,SIGNAL_ENVELOPE_SCALE,NORMALIZE_WEIGHTS)
        break
    ######################################### ROC PLOTS #########################################

    ######################################### ML METRIC PLOTS #########################################
    
    
    

            


