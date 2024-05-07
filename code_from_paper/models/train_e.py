import os

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.append('/Users/rahat/Documents/GitHub/ics691d_bci/code_from_paper')
from model_set.models import HopefullNet
import numpy as np

import tensorflow as tf
from data_processing.general_processor import Utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle
from sklearn.preprocessing import minmax_scale
tf.autograph.set_verbosity(0)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
#config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


#Params
source_path = "/Users/rahat/Downloads/eegdatabci/processed_files_run2/" # Path to processed data
save_path = os.path.join("/Users/rahat/Documents/GitHub/ics691d_bci/code_from_paper/models/saved_models/", "roi_e")
try:
    os.mkdir(save_path)
except FileExistsError:
    # directory already exists
    pass


# Load data
channels = Utils.combinations["e"] #[["C5", "C6"], ["C3", "C4"], ["C1", "C2"]]

exclude =  [38, 88, 89, 92, 100, 104]
subjects = [n for n in np.arange(1,110) if n not in exclude]
#Load data
x, y = Utils.load(channels, subjects, base_path=source_path)

#Transform y to one-hot-encoding
y_one_hot  = Utils.to_one_hot(y, by_sub=False)
#Reshape for scaling
reshaped_x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
#Grab a test set before SMOTE
x_train_raw, x_valid_test_raw, y_train_raw, y_valid_test_raw = train_test_split(reshaped_x,
                                                                            y_one_hot,
                                                                            stratify=y_one_hot,
                                                                            test_size=0.20,
                                                                            random_state=42)

#Scale indipendently train/test
x_train_scaled_raw = minmax_scale(x_train_raw, axis=1)
x_test_valid_scaled_raw = minmax_scale(x_valid_test_raw, axis=1)

#Create Validation/test
x_valid_raw, x_test_raw, y_valid, y_test = train_test_split(x_test_valid_scaled_raw,
                                                    y_valid_test_raw,
                                                    stratify=y_valid_test_raw,
                                                    test_size=0.50,
                                                    random_state=42)

# Reshaping to 2 channel structure (samples, channels, events) ??
x_train = x_train_scaled_raw.reshape(x_train_scaled_raw.shape[0], int(x_train_scaled_raw.shape[1]/2), 2).astype(np.float64)
x_valid = x_valid_raw.reshape(x_valid_raw.shape[0], int(x_valid_raw.shape[1]/2),2).astype(np.float64)
x_test = x_test_raw.reshape(x_test_raw.shape[0], int(x_test_raw.shape[1]/2),2).astype(np.float64)

y_train = y_train_raw

print(f"X_train: {x_train.shape}")
print(f"X_valid: {x_valid.shape}")
print(f"X_test: {x_test.shape}")
print(f"y_train: {y_train.shape}")


# #apply smote to train data
# print('classes count')
# print ('before oversampling = {}'.format(y_train_raw.sum(axis=0)))
# # smote
# from imblearn.over_sampling import SMOTE
# sm = SMOTE(random_state=42)
# x_train_smote_raw, y_train = sm.fit_resample(x_train_scaled_raw, y_train_raw)
# print('classes count')
# print ('before oversampling = {}'.format(y_train_raw.sum(axis=0)))
# print ('after oversampling = {}'.format(y_train.sum(axis=0)))

# x_train = x_train_smote_raw.reshape(x_train_smote_raw.shape[0], int(x_train_smote_raw.shape[1]/2), 2).astype(np.float64)

learning_rate = 1e-4

loss = tf.keras.losses.categorical_crossentropy
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model = HopefullNet()
modelPath = os.path.join(os.getcwd(), 'bestModel.weights.h5')

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

checkpoint = ModelCheckpoint( # set model saving checkpoints
    modelPath, # set path to save model weights
    monitor='val_loss', # set monitor metrics
    verbose=1, # set training verbosity
    save_best_only=True, # set if want to save only best weights
    save_weights_only=True, # set if you want to save only model weights (Must be true because of custom tf 1D-CNN)
    mode='auto', # set if save min or max in metrics
    save_freq='epoch' # interval between checkpoints
    )

earlystopping = EarlyStopping(
    monitor='val_loss', # set monitor metrics
    min_delta=0.001, # set minimum metrics delta
    patience=4, # number of epochs to stop training
    restore_best_weights=True, # set if use best weights or last weights
    )
callbacksList = [checkpoint, earlystopping] # build callbacks list
#%%

hist = model.fit(x_train, y_train, epochs=100, batch_size=10,
                validation_data=(x_valid, y_valid), callbacks=callbacksList) #32

with open(os.path.join(save_path, "hist.pkl"), "wb") as file:
    pickle.dump(hist.history, file)

model_filename = os.path.join(save_path, "model.keras")  # Using '.keras' as recommended for the native Keras format
model.save(model_filename)

"""
Test model
"""

del model # Delete the original model, just to be sure!
model = tf.keras.models.load_model(model_filename, custom_objects={"CustomModel": HopefullNet})

testLoss, testAcc = model.evaluate(x_test, y_test)
print('\nAccuracy:', testAcc)
print('\nLoss: ', testLoss)

from sklearn.metrics import classification_report, confusion_matrix
yPred = model.predict(x_test)

# convert from one hot encode in string
yTestClass = np.argmax(y_test, axis=1)
yPredClass = np.argmax(yPred,axis=1)

print('\n Classification report \n\n',
  classification_report(
      yTestClass,
      yPredClass,
       target_names=["B", "BI", "FI", "BR", "FR"]
      )
  )

print('\n Confusion matrix \n\n',
  confusion_matrix(
      yTestClass,
      yPredClass,
      )
  )