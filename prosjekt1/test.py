import random

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from Critic import CriticNN


c = CriticNN()

states = np.random.uniform(size=(10,9)) < 0.3
q = np.asarray([4, 1, 3, 4])
print(q[None].shape)
print(tf.convert_to_tensor([q]))
print(c.model.predict(x=q[None]))
print(c.model.fit(x=q[None], y=tf.convert_to_tensor([2])))
print(c.model.predict(x=q[None]))

print(c.model.fit(x=q[None], y=tf.convert_to_tensor([56])))
print(c.model.predict(x=q[None]))
print([state[None] for state in states])
print(states)
#print( [c.model(state) for state in states])
print(c)
exit()
print(tf.__version__)

critic = tf.keras.Sequential([
    tf.keras.layers.Input((9,)),
    tf.keras.layers.Dense(16, activation='tanh'),
    tf.keras.layers.Dense(16, activation='tanh'),
    tf.keras.layers.Dense(1)
])
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

states = np.random.uniform(size=(10, 9)) < 0.3
[critic(state[None]) for state in states]


x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
print(x_test)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(4, 3, epochs=3)
print(model.summary())
exit()
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

model.save('num_reader.model')

new_model = tf.keras.models.load_model('num_reader.model')

predictions = new_model.predict([x_test])

print(predictions)
print(np.argmax(predictions[0]))
plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.show()


exit()
x_bins = ([-3.1, -1.5], [-1.5, -1], [-1, -0.5], [-0.5, 0], [0, 0.5], [0.5, 1], [1, 1.5], [1.5, 3.1])
theta_bins = ([-5.21, -0.15], [-0.15, -0.1], [-0.1, -0.05], [-0.05, 0], [0, 0.05], [0.05, 0.1], [0.10, 0.15], [0.15, 5.21])
x_d1_bins = ([-4.1, -1.5], [-1.5, -1], [-1, -0.5], [-0.5, 0], [0, 0.5], [0.5, 1], [1, 1.5], [1.5, 4.1])
theta_d1_bins = ([-5, -1.5], [-1.5, -1], [-1, -0.5], [-0.5, 0], [0, 0.5], [0.5, 1], [1, 1.5], [1.5, 5])

bin_dict = {'x_bins': x_bins, 'theta_bins': theta_bins, 'x_d1_bins': x_d1_bins,
                 'theta_d1_bins': theta_d1_bins}

t = [[]] * 3
t[0] = list(range(5, 0, -1))
t[2] = [12]
print(t[-1])
print(t)

fig = plt.figure()
ax = fig.add_subplot(111)

num_pegs = 4
num_discs = 6
x = list(range(0, num_pegs + 1))
for i in range(num_discs):
    disc_factor = 0.95
    rect1 = matplotlib.patches.Rectangle(((1-disc_factor/2) + 0.075*i, (i * 0.06)), disc_factor - 0.15*i, 0.05, color=(random.random(), random.random(), random.random()))
    ax.add_patch(rect1)


#rect2 = matplotlib.patches.Rectangle((0, 150), 300, 20, color='pink')

#rect3 = matplotlib.patches.Rectangle((-300, -50),40, 200,color='yellow')


#ax.add_patch(rect2)

print(chr(97))
plt.xlim([0, num_pegs + 1])
plt.ylim([0, 1])
plt.xticks([k+1 for k in range(num_pegs)], [chr(97 + i) for i in range(num_pegs)])

plt.show()

listu = [[1, 2], [2, 1]]
listo = [[[1, 2], [2, 1]]]

#t = CriticNN()

print(listu in listo)
