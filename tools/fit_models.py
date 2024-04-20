import pickle
import os
import time

from typing import List

def fit_models(DATA_RELATIVE_FOLDER_PATH: str,
               EXPERIMENT_ID: str,
               DATA_FILENAME_WITHOUT_FILETYPE: str,
               K_FOLD: int,
               CLASS_WEIGHT: str,
               MODELS: List[object],
               SELECTED_PHYSICAL_VARIABLES: list,
               MODELS_RELATIVE_FOLDER_PATH: str,
               CLASSIFICATION_TYPE: str
            ) -> None:
    
    # create experiment ID folder in plots/
    os.chdir(MODELS_RELATIVE_FOLDER_PATH)
    os.makedirs(EXPERIMENT_ID, exist_ok=True)
    os.chdir('..')

    print("\n")
    print("Starting training with...")
    print(f"Number of models: {len(MODELS)} with K-Fold: {K_FOLD} so {len(MODELS)*K_FOLD} models will be trained\n")
    print(f"Models: {MODELS}")
    print(f"Class weight: {CLASS_WEIGHT}")
    print(f"Classificaiton type: {CLASSIFICATION_TYPE}")

    for K_FOLD in range(1,K_FOLD+1):
        print(f"Starting analysing K-Fold: {K_FOLD}\n")

        # change to the data folder
        os.chdir(DATA_RELATIVE_FOLDER_PATH+EXPERIMENT_ID)

        TRAIN_DATA_FILE_PATH = f"{DATA_FILENAME_WITHOUT_FILETYPE}_fold{K_FOLD}_{CLASS_WEIGHT}_train"

        # load the data
        with open(TRAIN_DATA_FILE_PATH+'.pkl', 'rb') as f:
            df_train = pickle.load(f)
        
        # change back to the main directory
        os.chdir('../..')

        os.chdir(MODELS_RELATIVE_FOLDER_PATH+EXPERIMENT_ID)

        # fit the models
        for model in MODELS:
            print(f"Training started for fitting model: {model.__class__.__name__}\n")
            start_time = time.time()
            if CLASSIFICATION_TYPE == 'binary':
            
                model.fit(df_train[SELECTED_PHYSICAL_VARIABLES], df_train['eventType']) # predict: 'Background' or 'Signal' (not 0 or 1)
                
            elif CLASSIFICATION_TYPE == 'multi_class':

                model.fit(df_train[SELECTED_PHYSICAL_VARIABLES], df_train['label']) # predict: label  
            
            else:
                raise ValueError(f"CLASSIFICATION_TYPE: {CLASSIFICATION_TYPE} not supported. Choose 'binary' or'multi-class' ")
            
            end_time = time.time()
            print(f"Training ended for fitting model: {model.__class__.__name__} it took {end_time-start_time:.2f} seconds \n")
                
            # save the model
            MODEL_FILE_PATH = f'{TRAIN_DATA_FILE_PATH}_{CLASSIFICATION_TYPE}_{model.__class__.__name__}'
            print(f"Saving model to {MODEL_FILE_PATH}.pkl\n")
            with open(MODEL_FILE_PATH+'.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        # change back to the main directory
    
        os.chdir('../..')
