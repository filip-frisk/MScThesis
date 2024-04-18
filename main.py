import pickle 

# Parameters are in capital letters

########################################################## PATHS & FILENAME ##########################################################

# All relative to main folder
DATA_RELATIVE_FOLDER_PATH = 'data/28Jan24/'
DATA_FILENAME_WITHOUT_FILETYPE = 'ntuples-ggFVBF2jet-SF-28Jan24'
CUT = 'ggFVBF2jet-SF-28Jan24'
EXPERIMENT_ID = '240418_I' # DATE + ID: YYMMDD + rome numericals: I, II, III, IV, V, VI, VII, VIII, IX, X
PLOT_RELATIVE_FOLDER_PATH = 'plots/'
MODELS_RELATIVE_FOLDER_PATH = 'models/'

########################################################## SIGNAL & VARIABLES  ##########################################################

SIGNAL_CHANNEL = ['VBF']
BACKGROUND_CHANNEL = ['WW', 'Zjets', 'ttbar'] # order in size event weight or MC samples

SELECTED_OTHER_VARIABLES = ['eventType','label','eventNumber','weight']
SELECTED_PHYSICAL_VARIABLES = ['DPhijj', 'mll', 'mT', 'DYjj', 'mjj', 'ptTot', 'mL1J1', 'mL1J2', 'mL2J1', 'mL2J2','ptJ1','ptJ2','ptJ3','METSig'] # eta_l_centrality missing?
SELECTED_PHYSICAL_VARIABLES_UNITS = ['rad?','?eV','?eV','','?eV','?eV','?eV','?eV','?eV','?eV','?eV','?eV','?eV',''] # Is it really GeV? units? '' empty for unitless

CLASS_WEIGHT = 'MC_EACH_bkg_as_sgn' #alternatives are 'raw', 'MC_EACH_bkg_as_sgn', 'MC_TOTAL_bkg_as_sgn', 'CW_EACH_bkg_as_sgn', 'CW_TOTAL_bkg_as_sgn'

########################################################## CLASSIFICATION PROBLEM TYPE ##########################################################

CLASSIFICATION_TYPE = 'binary' # Normalized distributions: 'multi-class', 'binary' Non-normalized: 'multi-label'

#from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neural_network import MLPClassifier
#from sklearn.svm import SVC
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.naive_bayes import GaussianNB

# See https://scikit-learn.org/stable/supervised_learning.html 
MODELS = [
#    GradientBoostingClassifier(),
    RandomForestClassifier(),
#    MLPClassifier(),
#    SVC(),  # SVM with probability estimates
#    KNeighborsClassifier(),
    LogisticRegression(),
#   DecisionTreeClassifier(),
#   GaussianNB()
]

########################################################## DATA SELECTION ##########################################################

# New dataframe
""" 
from tools.create_dataframe import create_dataframe

create_dataframe(DATA_RELATIVE_FOLDER_PATH, 
                 DATA_FILENAME_WITHOUT_FILETYPE, 
                 SIGNAL_CHANNEL, 
                 BACKGROUND_CHANNEL, 
                 SELECTED_OTHER_VARIABLES, 
                 SELECTED_PHYSICAL_VARIABLES)
"""

# Old dataframe

with open(f'{DATA_RELATIVE_FOLDER_PATH+DATA_FILENAME_WITHOUT_FILETYPE}.pkl', 'rb') as f:
    df = pickle.load(f)

########################################################## DATA VISUALIZATION ##########################################################

# multiple variables plots 
"""
from tools.create_pretty_histograms import plot_one_physical_variable    
overflow_underflow_percentile = {'lower_bound': 5, 'upper_bound': 95} # ex 1% and 99% percentile, all data outside this range will be added to the last and first bin respectively
bins = 20
plot_type = 'prefit' # 'postfit
signal_envelope_scale = 5000 # easier to guess than to scale dynamically 

NORMALIZE_WEIGHTS = False

for variable, unit in zip(SELECTED_PHYSICAL_VARIABLES, SELECTED_PHYSICAL_VARIABLES_UNITS):
    plot_one_physical_variable(df, 
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
                               NORMALIZE_WEIGHTS)  
    
"""

########################################################## DATA PREPROCESSING ##########################################################

# One dataframe

from tools.pre_process_data import pre_process_data

training_data_size = 0.8
random_seed = None # 42

k_fold = 0
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


# Multiple dataframes
"""
from tools.pre_process_data import pre_process_data

training_data_size = 0.8
random_seed = None # 42

for k_fold in range(1,6):
    print(f"Starting analysing K-Fold: {K_FOLD}")
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
########################################################## DATA TRAINING ##########################################################

"""
from tools.fit_models import fit_models
k_fold = 1

fit_models(DATA_RELATIVE_FOLDER_PATH,
           EXPERIMENT_ID,
           DATA_FILENAME_WITHOUT_FILETYPE,
           k_fold,
           CLASS_WEIGHT,
           MODELS,
           SELECTED_PHYSICAL_VARIABLES,
           MODELS_RELATIVE_FOLDER_PATH,
           CLASSIFICATION_TYPE)
"""
########################################################## EVALUATE MODELS ##########################################################

""" 
from tools.evaluate_models import evaluate_models

K_FOLD = 1

evaluate_models(
    PLOT_RELATIVE_FOLDER_PATH,
    MODELS_RELATIVE_FOLDER_PATH,
    EXPERIMENT_ID,
    DATA_RELATIVE_FOLDER_PATH,
    DATA_FILENAME_WITHOUT_FILETYPE,
    K_FOLD,
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
* Tree/channel: HWW_Zjets;1, with 4,505,202 MC simulations and 209,848.92 total event weight.
* Tree/channel: HWW_OtherVV;1, with 1,403,847 MC simulations and 5,196.99 total event weight.
* Tree/channel: HWW_WW;1, with 1,384,730 MC simulations and 10,025.92 total event weight.
* Tree/channel: HWW_singletop;1, with 243,573 MC simulations and 3,190.52 total event weight.
* Tree/channel: HWW_ttbar;1, with 682,008 MC simulations and 25,341.35 total event weight.
* Tree/channel: HWW_triboson;1, with 30,387 MC simulations and 140.99 total event weight.
* Tree/channel: HWW_ggF;1, with 35,129 MC simulations and 271.94 total event weight.
* Tree/channel: HWW_htt;1, with 49,051 MC simulations and 74.44 total event weight.
* Tree/channel: HWW_VBF;1, with 105,641 MC simulations and 88.54 total event weight.
* Tree/channel: HWW_VH;1, with 120,704 MC simulations and 86.82 total event weight.
"""