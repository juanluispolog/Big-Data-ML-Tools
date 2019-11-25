import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import time

start_read = time.time()
data = pd.read_csv('/data/Alumns.csv')
end_read = time.time()

print("Data Shape:", data.shape)
x_orig = data.iloc[:, 1:].values

y_orig = data.iloc[:, 0].values

print("Shape of Feature Matrix:", x_orig.shape)
print("Shape Label Vector:", y_orig.shape)
x_pos = np.array([x_orig[i] for i in range(len(x_orig)) if y_orig[i] == 1])

x_neg = np.array([x_orig[i] for i in range(len(x_orig)) if y_orig[i] == 0])

plt.scatter(x_pos[:, 0], x_pos[:, 1], color = 'blue', label = 'Positive')

plt.scatter(x_neg[:, 0], x_neg[:, 1], color = 'red', label = 'Negative')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Plot of given data')
plt.legend()

plt.show()
oneHot = OneHotEncoder()

oneHot.fit(x_orig)
x = oneHot.transform(x_orig).toarray()

y_orig = y_orig.reshape(400,1)

oneHot.fit(y_orig)
y = oneHot.transform(y_orig).toarray()

alpha, epochs = 0.0035, 500
m, n = x.shape
print('m =', m)
print('n =', n)
print('Learning Rate =', alpha)
print('Number of Epochs =', epochs)
X = tf.placeholder(tf.float32, [None, n])

Y = tf.placeholder(tf.float32, [None, 2])

W = tf.Variable(tf.zeros([n, 2]))

b = tf.Variable(tf.zeros([2]))
Y_hat = tf.nn.sigmoid(tf.add(tf.matmul(X, W), b))

cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = Y_hat, labels = Y)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(cost)

start_model_exec = time.time()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    cost_history, accuracy_history = [], []

    for epoch in range(epochs):
        cost_per_epoch = 0

        sess.run(optimizer, feed_dict={X: x, Y: y})

        c = sess.run(cost, feed_dict={X: x, Y: y})

        correct_prediction = tf.equal(tf.argmax(Y_hat, 1),
                                      tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                          tf.float32))

        cost_history.append(sum(sum(c)))
        accuracy_history.append(accuracy.eval({X: x, Y: y}) * 100)

        if epoch % 100 == 0 and epoch != 0:
            print("Epoch " + str(epoch) + " Cost: "
                  + str(cost_history[-1]))

    Weight = sess.run(W)
    Bias = sess.run(b)

    correct_prediction = tf.equal(tf.argmax(Y_hat, 1),
                                  tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                      tf.float32))
    end_model_exec = time.time()

    print("\nAccuracy:", accuracy_history[-1], "%")
    print("------------------------------")
    print("Read time: " + str(end_read - start_read))
    print("Train time: " + str(end_model_exec - start_model_exec))
    print("Total time: " + str(end_model_exec - start_read))
