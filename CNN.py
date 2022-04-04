import keras
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Flatten
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
import os
import keras.backend as K
import tensorflow as tf


def get_callbacks():
    model_type = "CNN"
    save_dir = os.path.join(os.getcwd(), 'CNN_co')
    model_name = "Methane_%s_model.{epoch:03d}.h5" % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    checkpoint = ModelCheckpoint(
        filepath=filepath,
        monitor='val_loss',
        verbose=1,
        save_best_only=True)


    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
        factor=np.sqrt(0.1),
        cooldown=0,
        patience=5,
        min_lr=0.5e-6)

    return [checkpoint,es,lr_reducer]

model = Sequential()
# model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
model.add(layers.Conv1D(120,1, activation='relu',input_shape=(1,319)))
model.add(Flatten())
model.add(layers.Dense(16, activation="sigmoid"))

model.summary()


x_train=np.load("train_x_methane.npy")
y_train=np.load("train_y_methane.npy")

x_val=np.load("val_x_methane.npy")
y_val=np.load("val_y_methane.npy")

x_test=np.load("test_x_methane.npy")
y_test=np.load("test_y_methane.npy")

print(x_test.shape)
x=x_train.reshape(20513*128,319)
x=np.expand_dims(x, axis=1)
y=y_train.reshape(20513*128,16)

xv=x_val.reshape(2014*128,319)
xv=np.expand_dims(xv, axis=1)
yv=y_val.reshape(2014*128,16)

xt=x_test.reshape(9631*128,319)
xt=np.expand_dims(xt, axis=1)
yt=y_test.reshape(9631*128,16)

def MASE(training_series,prediction_series):
    #

    d=K.mean(K.abs((training_series[-1]-training_series[1:])))

    errors = K.abs(training_series - prediction_series)
    return K.mean(errors / d)

callbacks = get_callbacks()
model.compile(optimizer=keras.optimizers.RMSprop(), loss="mae")
history =    model.fit(x, y,
              batch_size=128,
              epochs=1,
              verbose=2,
              validation_data=(xv, yv),
              callbacks=callbacks)
score = model.evaluate(xv, yv, verbose=0)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
print("MASE:")
predict=model.predict(x)
d=np.mean(np.abs(x[1:]-x[-1]))
e=np.mean(np.abs(y-predict))
print("training:",e/d)

predict1=model.predict(xv)
d=np.mean(np.abs(xv[1:]-xv[-1]))
e=np.mean(np.abs(yv-predict1))
print("validating:",e/d)

predict2=model.predict(xt)
d=np.mean(np.abs(xt[1:]-xt[-1]))
e=np.mean(np.abs(yt-predict2))
print("testing:",e/d)

print("MSE:")

print("training:", mean_squared_error(y,predict))
print("validating:", mean_squared_error(yv,predict1))
print("testing:", mean_squared_error(yt,predict2))

print("MAE:")

print("training:", mean_absolute_error(y,predict))
print("validating:", mean_absolute_error(yv,predict1))
print("testing:", mean_absolute_error(yt,predict2))
#
# print(score)
# import matplotlib.pyplot as plt
# loss = history.history['loss']
# # loss[:] = [x-0.1 for x in loss]
# val_loss = history.history['val_loss']
# # val_loss[:] = [x / 100 for x in val_loss]
# epochs = range(1, len(loss) + 1)
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss (MAE) Ethylene_CO')
# plt.legend()
# plt.show()


# 4178392
#
# 3789784
#
#
#
# 2785608
#
# 29256