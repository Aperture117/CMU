import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

train_x = pd.read_csv("Newton_vs_Machine/train_X.csv")
train_y = pd.read_csv("Newton_vs_Machine/train_Y.csv")

print(train_x.shape) # time, x1 initial
print(train_y.shape) # x1,x2 locations

# Build model
input_dim = train_x.shape[1]
output_dim = train_y.shape[1]

model = tf.keras.Sequential([
tf.keras.layers.InputLayer(input_shape=(input_dim, )),

tf.keras.layers.Dense(128, activation='relu'), # 1
tf.keras.layers.Dense(128, activation='relu'), # 2
tf.keras.layers.Dense(128, activation='relu'), # 3
tf.keras.layers.Dense(128, activation='relu'), # 4
tf.keras.layers.Dense(128, activation='relu'), # 5
tf.keras.layers.Dense(128, activation='relu'), # 6
tf.keras.layers.Dense(128, activation='relu'), # 7
tf.keras.layers.Dense(128, activation='relu'), # 8
tf.keras.layers.Dense(128, activation='relu'), # 9
tf.keras.layers.Dense(128, activation='relu'), # 10

tf.keras.layers.Dense(output_dim, activation='linear')
])

# Complie model with Adam optimizers and MAE loss function
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae', metrics=['mae'])
# Tensorboard connect
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='TB_logDir', histogram_freq=1)

# Model train
history = model.fit(train_x, train_y,
                    epochs=1000,          
                    batch_size=5000,
                    validation_split = 0.01,     
                    callbacks=[tensorboard_callback],
                    verbose=1)

# plot
plt.figure(figsize=(12, 6))

# train loss
plt.plot(history.history['loss'], label='Training Loss')
# test set loss
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('loss of train and test')
plt.legend()
plt.show()
plt.savefig("hw6_loss.png")
