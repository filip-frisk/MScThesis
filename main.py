#from tools.metrics import *

from tools.create_dataframe import create_dataframe

DATA_RELATIVE_FOLDER_PATH = 'data/'
DATA_FILENAME_WITHOUT_FILETYPE = 'ntuples-ggFVBF2jet-SF-28Jan24'

SIGNAL_CHANNEL = ['VBF']
BACKGROUND_CHANNEL = ['WW', 'Zjets', 'ttbar']

SELECTED_OTHER_VARIABLES = ['eventType','label','eventNumber','weight']
SELECTED_PHYSICAL_VARIABLES = ['DPhijj', 'mll', 'mT', 'DYjj', 'mjj', 'ptTot', 'mL1J1', 'mL1J2', 'mL2J1', 'mL2J2','ptJ1','ptJ2','ptJ3','METSig']

create_dataframe(DATA_RELATIVE_FOLDER_PATH, 
                 DATA_FILENAME_WITHOUT_FILETYPE, 
                 SIGNAL_CHANNEL, BACKGROUND_CHANNEL, 
                 SELECTED_OTHER_VARIABLES, 
                 SELECTED_PHYSICAL_VARIABLES)

