import scipy.io as scio
import numpy as np
import pandas as pd
import datetime
from utils import auc, f1
from keras import *
import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 5 == 0:  # or save after some epoch, each k-th epoch etc.
            self.model.save("./result/eachEpochInterval/model_{}.hd5".format(epoch + 1))


training_set1 = scio.loadmat("./preprocessed_data/normtraining_set1.mat")
train1 = np.array(training_set1['y'])
training_set2 = scio.loadmat("./preprocessed_data/normtraining_set2.mat")
train2 = np.array(training_set2['y'])
training_set3 = scio.loadmat("./preprocessed_data/normtraining_set3.mat")
train3 = np.array(training_set3['y'])
training_set4 = scio.loadmat("./preprocessed_data/normtraining_set4.mat")
train4 = np.array(training_set4['y'])

train_label = np.concatenate([train1,train2,train3,train4])
train_label = np.array(train_label)

test_set1 = scio.loadmat("./preprocessed_data/normvalidation_set1.mat")
val1 = np.array(test_set1['y'])
test_set2 = scio.loadmat("./preprocessed_data/normvalidation_set2.mat")
val2 = np.array(test_set2['y'])
test_set3 = scio.loadmat("./preprocessed_data/normvalidation_set3.mat")
val3 = np.array(test_set3['y'])
test_set4 = scio.loadmat("./preprocessed_data/normvalidation_set4.mat")
val4 = np.array(test_set4['y'])

val_label = np.concatenate([val1,val2,val3,val4])
val_label = np.array(val_label)

data = pd.read_csv("features_unnorm_train3.csv")
val = pd.read_csv("features_unnorm_val3.csv")

def build_model(data):
    model = Sequential()
    #model.add(LSTM(128, return_sequences=True , input_shape=(1, 5)))
    #model.add(LSTM(256))

    model.add(LSTM(64, return_sequences=True , input_shape=(1, 5)))
    model.add(LSTM(256))
    model.add(Dropout(0.5))


    model.add(Dense(2, activation='softmax', kernel_initializer='he_uniform'))
    return model

start = 0
end = 5

result = []
for i in range(19):
    temp =[]
    print(i)
    df1 = data[list(data.columns.values[start:end])]
    test1 = val[list(val.columns.values[start:end])]
    start = end
    end += 5
    data_feat = df1.values
    test_feat= test1.values

    data_feat = data_feat.reshape((data_feat.shape[0], 1, data_feat.shape[1]))
    val_feat = test_feat.reshape((test_feat.shape[0], 1, test_feat.shape[1]))
    val_feat = np.array(val_feat)
    print(df1.columns.values)
    print(df1.shape)
    data_feat = np.array(data_feat)

    model = build_model(data_feat)
    print(model.summary())

    date = datetime.date.today().strftime("%Y%m%d")
    filepath = "./lstm/best_model(channel LSTM)/" + str(i) + "/" + date + 'best_model(channel).{epoch:02d}-{val_auc:.4f}.h5'
    ckpt = keras.callbacks.ModelCheckpoint(filepath=filepath,
                                           monitor='val_auc',
                                           save_best_only=True,
                                           verbose=1,
                                           mode='max')

    saver = CustomSaver()

    # adam = keras.optimizers.Adam(lr=1e-05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.95)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', auc, f1])

    hist = model.fit(
        data_feat,
        train_label,
        batch_size=300,
        epochs=120,
        validation_data=(val_feat, val_label),
        callbacks=[ckpt, saver])
