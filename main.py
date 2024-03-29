import pickle 

from tools.create_dataframe import create_dataframe

from tools.create_pretty_histograms import plot_one_physical_variable

from tools.pre_process_data import pre_process_data

### PATHS & FILENAME ###
DATA_RELATIVE_FOLDER_PATH = 'data/28Jan24/'
DATA_FILENAME_WITHOUT_FILETYPE = 'ntuples-ggFVBF2jet-SF-28Jan24'
CUT = 'ggFVBF2jet-SF-28Jan24'

### DATA SELECTION ###

SIGNAL_CHANNEL = ['VBF']
BACKGROUND_CHANNEL = ['WW', 'Zjets', 'ttbar'] # order in size event weight or MC samples

SELECTED_OTHER_VARIABLES = ['eventType','label','eventNumber','weight']
SELECTED_PHYSICAL_VARIABLES = ['DPhijj', 'mll', 'mT', 'DYjj', 'mjj', 'ptTot', 'mL1J1', 'mL1J2', 'mL2J1', 'mL2J2','ptJ1','ptJ2','ptJ3','METSig'] # eta_l_centrality missing?
SELECTED_PHYSICAL_VARIABLES_UNITS = ['rad?','?eV','?eV','','?eV','?eV','?eV','?eV','?eV','?eV','?eV','?eV','?eV',''] # Is it really GeV? units? '' empty for unitless
### DATA VISUALIZATION ###
OVERFLOW_UNDERFLOW_PERCENTILE = {'lower_bound': 5, 'upper_bound': 95} # ex 1% and 99% percentile, all data outside this range will be added to the last and first bin respectively
BINS = 20
### DATA PREPROCESSING ###
TRAIN_DATA_SIZE = 0.8
RANDOM_SEED = None # 42
EXPERIMENT_ID = '240328_I' # DATE + ID: YYMMDD + rome numericals: I, II, III, IV, V, VI, VII, VIII, IX, X
CLASS_WEIGHT = 'tot_bkg_as_sgn' # alt.'as_is' or 'bkg_as_VBF' or tot_bkg_as_VBF

# run if you want to create a new dataframe
#create_dataframe(DATA_RELATIVE_FOLDER_PATH, DATA_FILENAME_WITHOUT_FILETYPE, SIGNAL_CHANNEL, BACKGROUND_CHANNEL, SELECTED_OTHER_VARIABLES, SELECTED_PHYSICAL_VARIABLES)

# run if you want to load an old dataframe
with open(f'{DATA_RELATIVE_FOLDER_PATH+DATA_FILENAME_WITHOUT_FILETYPE}.pkl', 'rb') as f:
    df = pickle.load(f)

# create kinetic plots for entire dataframe
    
for variable, unit in zip(SELECTED_PHYSICAL_VARIABLES, SELECTED_PHYSICAL_VARIABLES_UNITS):
    plot_one_physical_variable(df, variable, unit, SIGNAL_CHANNEL , BACKGROUND_CHANNEL, CUT,DATA_FILENAME_WITHOUT_FILETYPE,OVERFLOW_UNDERFLOW_PERCENTILE,BINS)     


# run if you want to pre-process the data
#for K_FOLD in range(1,6):
#    print(f"Starting analysing K-Fold: {K_FOLD}")
#    pre_process_data(df,TRAIN_DATA_SIZE,RANDOM_SEED,EXPERIMENT_ID,DATA_RELATIVE_FOLDER_PATH,DATA_FILENAME_WITHOUT_FILETYPE,K_FOLD,CLASS_WEIGHT,SIGNAL_CHANNEL,BACKGROUND_CHANNEL)


    
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