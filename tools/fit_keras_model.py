import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau


##################################################################################################################################
########################################################## ARCHITECTURE ##########################################################
##################################################################################################################################

CLASSIFICATION_TYPE = 'multi_class' # 'binary' or 'multi_class
SIGNAL_CHANNEL = ['VBF']
BACKGROUND_CHANNEL = ['Zjets', 'ttbar','WW'] # order in size event weight or MC samples
SELECTED_PHYSICAL_VARIABLES = ['DPhill', 'DYjj', 'mjj', 'mll', 'mT', 'ptTot','sumOfCentralitiesL','mL1J1', 'mL1J2', 'mL2J1', 'mL2J2','ptJ1','ptJ2','ptJ3','METSig'] # https://gitlab.cern.ch/bejaeger/sfusmlkit/-/blob/master/configs/HWW/train.cfg (row 18)

# Reimplemented 
# Without custom callback Z0Callback
# Binary 2jSF: https://gitlab.cern.ch/bejaeger/sfusmlkit/-/blob/master/configs/HWW/train.cfg
# Multi-class 2jSF: https://gitlab.cern.ch/bejaeger/sfusmlkit/-/blob/master/configs/HWW/trainMultiClass.cfg
# For 2jDF see images from Ahmed 


if CLASSIFICATION_TYPE == 'binary':
    ### Architecture for binary classification ###

    # For binary you have to alternatives in practise 
    # 1) One Neuron with Sigmoid Activation (most common)
    last_layer = {'units': 1, 'activation': 'sigmoid'}
    # 2) Two Neurons with Softmax Activation
    #last_layer = {'units': 2, 'activation': 'softmax'}

    dropout_rate = 0.4
    second_dropout = {'rate': dropout_rate}

    ### Training parameters ###
    nepochs = 20
    batchsize = 512
    metric = 'binary_accuracy'

if CLASSIFICATION_TYPE == 'multi_class':
    
    ### Architecture for binary classification ###
    last_layer = {'units': len(BACKGROUND_CHANNEL+SIGNAL_CHANNEL), 'activation': 'softmax'}

    dropout_rate = 0.2
    second_dropout = ''
    
    ### Training parameters / Optimizer / Metrics ###
    nepochs = 40
    batchsize = 256
    metric = 'categorical_accuracy'

# Callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]

# Optimizer configuration
optimizer = tf.keras.optimizers.Adagrad(learning_rate=30.0)

# Model specification
# input layer is not defined here 
model_hidden_layer_spec = {
    'Dense_1': {'units': 256, 'activation': 'relu', 'input_shape': (len(SELECTED_PHYSICAL_VARIABLES),)},
    'Dropout_1': {'rate': dropout_rate},
    'Dense_2': {'units': 128, 'activation': 'relu'},
    'Dropout_2': second_dropout,
    'Dense_3': {'units': 64, 'activation': 'relu'},
    'Dropout_3': {'rate': dropout_rate},
    'Dense_4': {'units': 32, 'activation': 'relu'},
    'Dense_5': {'units': 24, 'activation': 'relu'},
    'Dropout_4': {'rate': dropout_rate},
    'Dense_6': {'units': 16, 'activation': 'relu'},
    'Dense_7': last_layer
}

# Create the Sequential model
model = Sequential()

# Add layers from the specification
for key, params in model_hidden_layer_spec.items():
    layer_type = key.split('_')[0]
    if params == '':
        print(params)
        continue
    else:
        model.add(getattr(tf.keras.layers, layer_type)(**params))


# Compile the model
model.compile(optimizer=optimizer, loss='log_cosh', metrics=[metric])

# Print the model summary
model.summary()

##################################################################################################################################
########################################################## Train #################################################################
##################################################################################################################################


# Save the model
# As of now, Keras models are pickle-able. But we still recommend using model.save() to save model to disk.
# Models saved with model.save() will be compatible with future versions of Keras and can also be exported to other platforms and implementations

# Can't pickle weakref comes because Deep Learning models are too large and pickle only used for storing small models
# Use this : HDF5 used for storing large data

