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
EXPERIMENT_ID = '240422_II' # DATE + ID: YYMMDD + rome numericals: I, II, III, IV, V, VI, VII, VIII, IX, X

########################################################## SIGNAL & VARIABLES  ##########################################################

SIGNAL_CHANNEL = ['VBF']
BACKGROUND_CHANNEL = ['Zjets', 'ttbar','WW'] # order in size event weight or MC samples

SELECTED_OTHER_VARIABLES = ['eventType','label','eventNumber','weight']
SELECTED_PHYSICAL_VARIABLES = ['DPhill', 'DYjj', 'mjj', 'mll', 'mT', 'ptTot','sumOfCentralitiesL','mL1J1', 'mL1J2', 'mL2J1', 'mL2J2','ptJ1','ptJ2','ptJ3','METSig'] # https://gitlab.cern.ch/bejaeger/sfusmlkit/-/blob/master/configs/HWW/train.cfg (row 18)
SELECTED_PHYSICAL_VARIABLES_UNITS = ['rad?','?eV','?eV','','?eV','?eV','?eV','?eV','?eV','?eV','?eV','?eV','?eV',''] # Is it really GeV? units? '' empty for unitless

CLASS_WEIGHT = 'raw' #alternatives are 'raw', 'MC_EACH_bkg_as_sgn', 'MC_TOTAL_bkg_as_sgn', 'CW_EACH_bkg_as_sgn', 'CW_TOTAL_bkg_as_sgn'

CHANNEL_EVENT_WEIGHT = {'Zjets': 209848.92, 'WW': 10025.92, 'ttbar': 25341.35, 'VBF': 88.54} # Root 28Jan24

CHANNEL_EVENT_WEIGHT_RATIO = {key: value/sum(CHANNEL_EVENT_WEIGHT.values()) for key, value in CHANNEL_EVENT_WEIGHT.items()}

########################################################## CLASSIFICATION PROBLEM TYPE ##########################################################

CLASSIFICATION_TYPE = 'binary' #'multi_class', 'binary' (multi-label is not relevant since each event is a definite process and not a mix of processes)

K_FOLD = 1 # number of k-folds for cross-validation

# Relevant classifiers in sklearn

from sklearn.base import BaseEstimator

# used to name the classifier
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier # in TMVA BDT
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC # in TMVA SVM
#from sklearn.neighbors import KNeighborsClassifier # in TMVA KNN / no class_weight

from sklearn.neural_network import MLPClassifier # in TMVA DNN

# uses a sklearn wrapper class for XGBoost https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier
from xgboost import XGBClassifier # used in https://gitlab.cern.ch/bejaeger/sfusmlkit/-/blob/master/train_bdt.py
# XGBoost is short for Extreme Gradient Boosting 
# and is an efficient implementation of the stochastic gradient boosting machine learning algorithm.
MLP = MLPClassifier(hidden_layer_sizes = (128,64,64,16,),
                                    activation='relu',
                                    batch_size=2048,
                                    solver='adam',
                                    learning_rate_init=0.001,
                                    beta_1=0.95,
                                    beta_2=0.9,
                                    max_iter=1000) 

# scale_pos_weight 
# https://stats.stackexchange.com/questions/243207/what-is-the-proper-usage-of-scale-pos-weight-in-xgboost-for-imbalanced-datasets 
# https://datascience.stackexchange.com/questions/16342/unbalanced-multiclass-data-with-xgboost

binary_scale_pos_weight = (sum(CHANNEL_EVENT_WEIGHT.values()) - CHANNEL_EVENT_WEIGHT['VBF']) / CHANNEL_EVENT_WEIGHT['VBF']
print(CHANNEL_EVENT_WEIGHT.values())
print(f"scale_pos_weight: {binary_scale_pos_weight}")
XGB = NamedClassifier(XGBClassifier(n_estimators=100,scale_pos_weight = binary_scale_pos_weight),name = "XGB") # Benjamin uses learning_rate and max_depth

#MLP1 = NamedClassifier(MLP,name = "MLP1")
#MLP2 = NamedClassifier(MLP,name = "MLP2")
#MLP3 = NamedClassifier(MLP,name = "MLP3")

RF = NamedClassifier(RandomForestClassifier(class_weight=CHANNEL_EVENT_WEIGHT_RATIO),name = "RF")
LR = NamedClassifier(LogisticRegression(class_weight=CHANNEL_EVENT_WEIGHT_RATIO),name = "LR")
HGBC = NamedClassifier(HistGradientBoostingClassifier(class_weight=CHANNEL_EVENT_WEIGHT_RATIO),name = "HGBC")
SVC = NamedClassifier(SVC(class_weight=CHANNEL_EVENT_WEIGHT_RATIO),name = "SVC")

# Minimized due to memory constraints
MODELS = [
    XGB,
    RF,
    LR,
    HGBC,
    SVC
    #MLP2,
    #MLP3
]
# bench: 
# https://gitlab.cern.ch/bejaeger/sfusmlkit/-/blob/master/configs/HWW/train.cfg 

BENCHMARK_MODEL = NamedClassifier(MLP,name = "BENCHMARK")
    
MODELS.append(BENCHMARK_MODEL)

######################################################################################################################################
############################################################### MODULES ##############################################################
######################################################################################################################################

########################################################## 1. DATA SELECTION ##########################################################

# New root file to dataframe
"""
from tools.create_dataframe import create_dataframe

create_dataframe(DATA_RELATIVE_FOLDER_PATH, 
                 DATA_FILENAME_WITHOUT_FILETYPE, 
                 SIGNAL_CHANNEL, 
                 BACKGROUND_CHANNEL, 
                 SELECTED_OTHER_VARIABLES, 
                 SELECTED_PHYSICAL_VARIABLES)

"""
# Old root file arleady in dataframe


"""
with open(f'{DATA_RELATIVE_FOLDER_PATH+DATA_FILENAME_WITHOUT_FILETYPE}.pkl', 'rb') as f:
    df = pickle.load(f)
    
"""
########################################################## 2. DATA VISUALIZATION ##########################################################

# multiple variables plots 
"""
from tools.create_pretty_histograms import create_pretty_histograms    
overflow_underflow_percentile = {'lower_bound': 5, 'upper_bound': 95} # ex 1% and 99% percentile, all data outside this range will be added to the last and first bin respectively
bins = 7
plot_type = 'prefit' # 'postfit
signal_envelope_scale = 5000 # easier to guess than to scale dynamically 

NORMALIZE_WEIGHTS = False

for variable, unit in zip(SELECTED_PHYSICAL_VARIABLES, SELECTED_PHYSICAL_VARIABLES_UNITS):
    create_pretty_histograms(df, 
                               variable, 
                               unit, 
                               SIGNAL_CHANNEL, 
                               BACKGROUND_CHANNEL, 
                               CUT,
                               DATA_FILENAME_WITHOUT_FILETYPE,
                               overflow_underflow_percentile,
                               bins,
                               PLOT_RELATIVE_FOLDER_PATH, 
                               plot_type,
                               signal_envelope_scale,
                               NORMALIZE_WEIGHTS,
                               K_FOLD,
                               EXPERIMENT_ID,
                               CLASSIFICATION_TYPE)  
    
"""

########################################################## 3. DATA PREPROCESSING ##########################################################

"""
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
"""
########################################################## 4. Fit/TRAINING ##########################################################

"""
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
"""

########################################################## 5. EVALUATE MODELS ##########################################################

"""
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
"""

###################################################### CURRENT DATASET: 28Jan24 ########################################################
""" In root file ggFVBF2jet-SF-28Jan24.root, you have the following:
Variables:

ALL_PHYSICAL_VARIABLES = [
    "ptJ1", "etaJ1", "phiJ1", "mJ1",
    "ptJ2", "etaJ2", "phiJ2", "mJ2", "ptL1", "etaL1", "phiL1", "mL1", "ptL2", "etaL2",
    "phiL2", "mL2", "avgMu", "mll", "mT", "nJ", "ptTot", "mjj", "DYjj", "DPhill",
    "DPhijj", "mL1J1", "mL1J2", "mL2J1", "mL2J2", "isEE", "OLV", "etaJ3", "ptJ3",
    "mJ3", "phiJ3", "sumOfCentralitiesL", "dRLeps", "dRJets", "dEtaLepsAbs", "dPhiJetsAbs",
    "sumEtaJetsAbs", "mtt", "centralityL1", "centralityL2", "centralJetVetoLeadpT",
    "MET", "METSig", "phiMET", "Ptll", "DRl0j0", "DRl0j1", "DRl1j0", "DRl1j1", "minDRl0ji",
    "minDRl1ji"
]

ALL_PHYSICAL_VARIABLES_UNITS = [
    "GeV", "Unitless", "rad", "GeV",
    "GeV", "Unitless", "rad", "GeV", "GeV", "Unitless", "rad", "GeV", "GeV", "Unitless",
    "rad", "GeV", "?", "GeV", "GeV", "Number of", "GeV", "GeV", "length?", "rad",
    "rad", "GeV", "GeV", "GeV", "GeV", "yes/no", "?", "Unitless", "GeV",
    "GeV", "rad", "length?", "Unitless", "Unitless", "Unitless", "rad",
    "rad", "GeV", "length?", "length?", "?",
    "Unitless", "Unitless", "Unitless", "GeV", "Unitless", "Unitless", "Unitless", "Unitless", "Unitless",
    "Unitless"
]

Trees/channels:
* Tree/channel: HWW_Data;1, with 281,950 MC simulations and 281,950.00 total event weight.
* Tree/channel: HWW_Vgamma;1, with 38,521 MC simulations and 10,570.94 total event weight.
* Tree/channel: HWW_Zjets;1, with 4,505,202 MC simulations and 209,848.92 total event weight. (***)
* Tree/channel: HWW_OtherVV;1, with 1,403,847 MC simulations and 5,196.99 total event weight.
* Tree/channel: HWW_WW;1, with 1,384,730 MC simulations and 10,025.92 total event weight. (***)
* Tree/channel: HWW_singletop;1, with 243,573 MC simulations and 3,190.52 total event weight.
* Tree/channel: HWW_ttbar;1, with 682,008 MC simulations and 25,341.35 total event weight. (***)
* Tree/channel: HWW_triboson;1, with 30,387 MC simulations and 140.99 total event weight.
* Tree/channel: HWW_ggF;1, with 35,129 MC simulations and 271.94 total event weight.
* Tree/channel: HWW_htt;1, with 49,051 MC simulations and 74.44 total event weight.
* Tree/channel: HWW_VBF;1, with 105,641 MC simulations and 88.54 total event weight. (***)
* Tree/channel: HWW_VH;1, with 120,704 MC simulations and 86.82 total event weight.
"""