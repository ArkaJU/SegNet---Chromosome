import matplotlib.pyplot as plt
import os
import numpy as np

from keras.callbacks import ModelCheckpoint, EarlyStopping
from SegNet import *
from load_data import *

X_train, y_train, X_test, y_test = load_data() 


num_classes = 4
Y_train = to_categorical(y_train, num_classes)
model = SegNet()
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
model.summary()



model_checkpoint = ModelCheckpoint('Weights.h5', monitor='val_loss', save_best_only=True)
early_stopping = EarlyStopping(patience=2, verbose=2)


print('Fitting model...')
history = model.fit(X_train, Y_train, batch_size=16, epochs=10, validation_split=0.05, 
                    shuffle=True, callbacks=[model_checkpoint, early_stopping])


model.load_weights('Weights.h5')
Y_test = model.predict(X_test, verbose=1)
Y_test = Y_test.reshape(20, 88, 88, 4)
Y = np.argmax(Y_test, axis=-1)


print('Saving predicted masks to files...')

pred_dir = 'Predictions'
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)
for i in range(20):
    image = Y[i, :, : ]
    plt.imsave(os.path.join(pred_dir, str(i + 1) + '_pred.png'), image, cmap = 'viridis')