import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

rng = numpy.random

learning_rate = 5e-5
training_epochs = 2000
display_step = 50

# Training Data
train_X = numpy.asarray([
    3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
    7.042,10.791,5.313,7.997,5.654,9.27,3.1])
#train_Y = numpy.asarray([
    #1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
    #2.827,3.465,1.65,2.904,2.42,2.94,1.3])
train_Y = numpy.asarray([
    0,0,0,1,1,0,1,1,1,0,
    1,1,0,1,0,1,0])
n_samples = train_X.shape[0]

X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(rng.randn(), name="weight")
#b = tf.Variable(rng.randn(), name="bias")

pred = tf.sigmoid(tf.multiply(X, W))
cost = -tf.reduce_sum(tf.add(
        tf.multiply(Y, tf.log(pred)),
        tf.multiply(tf.add(1., -Y), tf.log(tf.add(1., -pred)))))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

#with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for (x,y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X:x, Y:y})
        
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
            print("Epoch: {:04d}".format(epoch + 1),
                  "Cost = {:.9f}".format(c),
                  "W = {}".format(sess.run(W)))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
    print("Cost = {:.9f}".format(c),
          "W = {}".format(sess.run(W)))

    plt.plot(train_X, train_Y, 'ro', label='data')
    plt.plot(train_X, sess.run(pred, feed_dict={X:train_X}),
             label='Probability')
    plt.plot(train_X, numpy.full(train_X.shape, .5), 
             label='.5 line')
    plt.legend()
    plt.show()


