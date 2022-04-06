import copy
import multiprocessing as mp
import timeit

import nn
from tensorflow import keras

print("Number of processors: ", mp.cpu_count())
model = "models\\model_7_149"
pc = nn.LiteModel.from_keras_model(keras.models.load_model(model))
with open('tflite\\model.tflite', 'wb') as f:
  f.write(pc)

def loader(pc):
    q = nn.LiteModel.from_file(pc)

popo = timeit.timeit(lambda: loader('tflite\\model.tflite'), number=100)

print(popo)

