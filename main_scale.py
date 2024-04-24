import pickle 

######################################################################################################################################
############################################################# PARAMETERS #############################################################
######################################################################################################################################

########################################################## PATHS & FILENAME ##########################################################

# All relative to main folder
DATA_RELATIVE_FOLDER_PATH = 'data/'
PLOT_RELATIVE_FOLDER_PATH = 'plots/'
MODELS_RELATIVE_FOLDER_PATH = 'models/'

DATA_FILENAME_WITHOUT_FILETYPE = 'ntuples-ggFVBF2jet-SF-28Jan24'
CUT = 'ggFVBF2jet-SF-28Jan24'
EXPERIMENT_ID = '240424_I' # DATE + ID: YYMMDD + rome numericals: I, II, III, IV, V, VI, VII, VIII, IX, X

########################################################## SIGNAL & VARIABLES  ##########################################################

SIGNAL_CHANNEL = ['VBF']
BACKGROUND_CHANNEL = ['Zjets', 'ttbar','WW'] # order in size event weight or MC samples

SELECTED_OTHER_VARIABLES = ['eventType','label','eventNumber','weight']
SELECTED_PHYSICAL_VARIABLES = ['DPhill', 'DYjj', 'mjj', 'mll', 'mT', 'ptTot','sumOfCentralitiesL','mL1J1', 'mL1J2', 'mL2J1', 'mL2J2','ptJ1','ptJ2','ptJ3','METSig'] # https://gitlab.cern.ch/bejaeger/sfusmlkit/-/blob/master/configs/HWW/train.cfg (row 18)
SELECTED_PHYSICAL_VARIABLES_UNITS = ['rad?','?eV','?eV','','?eV','?eV','?eV','?eV','?eV','?eV','?eV','?eV','?eV',''] # Is it really GeV? units? '' empty for unitless

CLASS_WEIGHT = 'raw' #alternatives are 'raw', 'MC_EACH_bkg_as_sgn', 'MC_TOTAL_bkg_as_sgn', 'CW_EACH_bkg_as_sgn', 'CW_TOTAL_bkg_as_sgn'

########################################################## CLASSIFICATION PROBLEM TYPE ##########################################################

CLASSIFICATION_TYPE = 'binary' #'multi_class', 'binary' (multi-label is not relevant since each event is a definite process and not a mix of processes)
K_FOLD = 1 # number of k-folds for cross-validation

#Name Wrapper for sklearn based models
from sklearn.base import BaseEstimator
class NamedClassifier(BaseEstimator):
    def __init__(self, classifier, name=None):
        self.classifier = classifier
        self.name = name
    
    def fit(self, X, y):
        return self.classifier.fit(X, y)
    
    def predict_proba(self, X):
        return self.classifier.predict_proba(X)
    
    def __getattr__(self, attr):
        # Delegate attribute access to the wrapped classifier if not defined in NamedClassifier
        return getattr(self.classifier, attr)

########################### SKLEARN MODEL IMPORTS ###########################

# Wrapper helper class used to name the classifier for sklearn 

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier # in TMVA BDT / class_weights need integers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier # in TMVA KNN / no class_weight

from sklearn.neural_network import MLPClassifier # in TMVA DNN

mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128, 64, 32, 24, 16, 8, 4, 2),
    activation='relu',       # ReLU activation function
    solver='adam',           # Adam optimizer
    alpha=0.0001,            # L2 regularization term
    batch_size='auto',       # 'auto' means min(200, n_samples)
    learning_rate='adaptive',# Adaptive learning rate
    max_iter=1000            # Maximum number of iterations
)

rf_classifier = RandomForestClassifier(
    n_estimators=500,      # More trees
    max_depth=None,        # Full depth
    max_features='sqrt',   # Maximum features to consider for a split
    bootstrap=True,        # Use bootstrapping
    n_jobs=20             # Use all available cores
)

lr_classifier = LogisticRegression(
    max_iter=1000,         # Increased iterations for convergence
    solver='saga',         # 'saga' is good for large datasets and supports multiple penalty types
    penalty='l1',          # L1 regularization to encourage sparsity
    C=0.1,                 # Regularization strength
    class_weight='balanced', # Adjust for imbalanced datasets
    n_jobs=20              # Use all CPU cores
)

hgbc_classifier = HistGradientBoostingClassifier(
    max_iter=1000,          # maximum number of trees
    learning_rate=0.1,     # influences the contribution of each tree in the ensemble
    max_depth=None,        # no maximum depth, trees can grow until the leaves are pure
    max_bins=255,          # The maximum number of bins that features are bucketed into for faster computation
    early_stopping=True,   # Enables stopping the training process if the validation score does not improve for a certain number of iterations
    n_iter_no_change=20,   # The number of iterations with no improvement to wait before stopping training if early_stopping is True
    validation_fraction=0.1 # The proportion of training data to set aside as validation set for early stopping
)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# Create a pipeline with caching
knn_pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=50, svd_solver='randomized'),
    KNeighborsClassifier(n_neighbors=30, weights='distance', algorithm='auto', n_jobs=20)
)

########################### SKLEARN WRAPPER MODELS ###########################

#XGBoost
# uses a sklearn wrapper class for XGBoost https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier
from xgboost import XGBClassifier  
# XGBoost is short for Extreme Gradient Boosting 
# and is an efficient implementation of the stochastic gradient boosting machine learning algorithm.

#BalancedRandomForest 
# From https://imbalanced-learn.org/stable/index.html
from imblearn.ensemble import BalancedRandomForestClassifier

xgb_classifier = XGBClassifier(
    n_estimators=1000,     # The number of trees to build, increase complexity and overfitting
    max_depth=10,          # The maximum depth of each tree, increase complexity and overfitting
    learning_rate=0.01,    # prevents overfitting. Smaller values make the boosting process more conservative.
    scale_pos_weight=1/0.0003609388,  # Balancing of positive and negative weights. Useful in unbalanced classes to scale the gradient for the minority class.
    n_jobs=20              # Number of parallel threads used to run XGBoost. Setting to -1 means using all available cores.
)

brf_classifier = BalancedRandomForestClassifier(
    n_estimators=1000,  # n_estimators: Specifies the number of trees in the forest. A higher number improves the model's performance and robustness but increases computational load.
    max_depth=10,       # max_depth: Sets the maximum depth of each tree. Limiting depth helps prevent overfitting but too shallow trees might underfit.
    min_samples_leaf=2, # min_samples_leaf: The minimum number of samples required to be at a leaf node. A smaller leaf makes the model more sensitive to noise in the dataset, whereas a larger value results in a smoother decision boundary.
    max_features='auto',# max_features: The number of features to consider when looking for the best split. Using 'auto' lets the model consider all features which can provide the best splits but might increase computation time.
    bootstrap=True,     # bootstrap: Whether bootstrap samples are used when building trees. If True, each tree is trained on a random subset of the original data, with samples being drawn with replacement.
    n_jobs=20           # n_jobs: The number of jobs to run in parallel for both fit and predict. Setting n_jobs=-1 uses all processors, speeding up training but consuming more system resources.
)

###########################  MODELS ###########################

MLP = NamedClassifier(mlp_classifier,name = "MLP")
RF = NamedClassifier(rf_classifier,name = "RF")
LR = NamedClassifier(lr_classifier,name = "LR") # 
HGBC = NamedClassifier(hgbc_classifier,name = "HGBC")

KNN = NamedClassifier(knn_pipeline,name = "KNN")

XGB = NamedClassifier(xgb_classifier,name = "XGB") 

BRF = NamedClassifier(brf_classifier,name = "BRF")


MODELS = [
    MLP,
    RF,
    LR,
    HGBC,
    KNN,
    XGB,
    BRF
]
# bench: 
# see fit keras model for specifics 
#BENCHMARK_MODEL = NamedClassifier(MLP,name = "BENCHMARK")   
#MODELS.append(BENCHMARK_MODEL)

######################################################################################################################################
############################################################### MODULES ##############################################################
######################################################################################################################################

########################################################## 1. DATA SELECTION ##########################################################

# New root file to dataframe
#"""
from tools.create_dataframe import create_dataframe

create_dataframe(DATA_RELATIVE_FOLDER_PATH, 
                 DATA_FILENAME_WITHOUT_FILETYPE, 
                 SIGNAL_CHANNEL, 
                 BACKGROUND_CHANNEL, 
                 SELECTED_OTHER_VARIABLES, 
                 SELECTED_PHYSICAL_VARIABLES)

#"""
# Old root file arleady in dataframe


#"""
with open(f'{DATA_RELATIVE_FOLDER_PATH+DATA_FILENAME_WITHOUT_FILETYPE}.pkl', 'rb') as f:
    df = pickle.load(f)
    
########################################################## 3. DATA PREPROCESSING ##########################################################

#"""
from tools.pre_process_data import pre_process_data

training_data_size = 0.8
random_seed = None # 42

for k_fold in range(1, K_FOLD+1):
    print(f"Creating K-Fold: {k_fold} .")
    pre_process_data(df,
                     training_data_size,
                     random_seed,
                     EXPERIMENT_ID,
                     DATA_RELATIVE_FOLDER_PATH,
                     DATA_FILENAME_WITHOUT_FILETYPE,
                     k_fold,
                     CLASS_WEIGHT,
                     SIGNAL_CHANNEL,
                     BACKGROUND_CHANNEL)
#"""
########################################################## 4. Fit/TRAINING ##########################################################

#"""
from tools.fit_models import fit_models

fit_models(DATA_RELATIVE_FOLDER_PATH,
        EXPERIMENT_ID,
        DATA_FILENAME_WITHOUT_FILETYPE,
        K_FOLD,
        CLASS_WEIGHT,
        MODELS,
        SELECTED_PHYSICAL_VARIABLES,
        MODELS_RELATIVE_FOLDER_PATH,
        CLASSIFICATION_TYPE)
#"""

########################################################## 5. EVALUATE MODELS ##########################################################

#"""
from tools.evaluate_models import evaluate_models

for k_fold in range(1, K_FOLD+1):
    evaluate_models(
        PLOT_RELATIVE_FOLDER_PATH,
        MODELS_RELATIVE_FOLDER_PATH,
        EXPERIMENT_ID,
        DATA_RELATIVE_FOLDER_PATH,
        DATA_FILENAME_WITHOUT_FILETYPE,
        k_fold,
        CLASS_WEIGHT,
        MODELS,
        CLASSIFICATION_TYPE,
        SIGNAL_CHANNEL,
        BACKGROUND_CHANNEL,
        CUT,
        SELECTED_PHYSICAL_VARIABLES
    )
#"""
