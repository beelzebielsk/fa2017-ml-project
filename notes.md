- Questions you can ask:
  - Which features describe the data the most? How important is each
    feature?
  - What linear combination of features explains the variation of the
    data the most?
  - Which models seem to predict the classifications the best? Perhaps
    consider figuring out why this might be the case.
  - Separate the data based on the values of some features. Does the
    behavior/prediction change in either group? Why might it change?
    For instance, consider data which describes matches in a sport,
    where the features include which team was the home team and which
    team was the away team. You could check the importance of this
    feature:
    - For the whole dataset.
    - For games beyond a certain year.
    - For games in a single year.

    And you could take that feature importance and perhaps try to
    visualize it. How has the importance of a feature changed over
    time?

  - Does it make sense to do a "bootstrapped" feature importance? If
    I'm going to make some judgments about how important a feature is
    based on the numbers that come out of this algorithm, it might be
    nice to figure out how consistent that number is going to be with
    different samples. If it varies wildly, then I probably can't
    trust it.

Sources
=======

- Face data: <https://www.kaggle.com/drgilermo/face-images-with-marked-landmark-points>
- Tutorials:
  - <http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/>
  - <https://github.com/aymericdamien/TensorFlow-Examples>
  - <https://github.com/pkmital/tensorflow_tutorials>
  - <https://github.com/nlintz/TensorFlow-Tutorials>

Problems?
=========

- Could be the loss function is inappropriate for faces.
- Somehow, even when the network has as many neurons in hidden layer
  as there are in input layer, the mapping isn't perfect.

Questions
==========

- If the regularization is over every weight in the network, doesn't
  that mean that the network size is a factor in the regularization?

Creation Notes
==============

- Tried to make tensorflow autoencoder, that went poorly for all but
  simplest images.
- Switched to Keras, models were better.
- Thought about problem some more and realized a more concrete version
  of what I want to do:
  - I want to try getting autoencoders to learn different
    repreentations of the feature space for some dataset.
  - Then, I want to run a different dataset that lives inside of the
    same feature space through the autoencoder.
  - My hope is that the encoding of feature space from the original
    dataset produces interesting results after data from a different
    dataset are encoded and then decoded.
- So my goal is now to learn to create different representations of
  the feature space of data:
  - Compressed
  - Sparse. I want this a little more, because it seems that sparse
    representations help more with each neuron learning a "high-level
    feature". That's really want I want. To look at different pictures
    with eyes trained to recognize the high-level features of a
    totally different object.
- To confirm my findings for learning different feature spaces, I want
  to also try to visualize the learned concepts of different neurons.

- On suggestion from professor, I started looking at my weights to see
  what regularization I should use.
- I had to look up the precise defn of l1 regularization to figure out
  what I should do. Found out that regularization used *all* of the
  weights in the *entire* network, as if it were one long vector. Once
  I saw that, I tried summing up all of the network weights to see how
  large the cost of the l1 norm of the weights was.

```
In [6]: [x.shape for x in weights]
Out[6]:
[(784, 512),
 (512,),
 (512, 256),
 (256,),
 (256, 128),
 (128,),
 (128, 256),
 (256,),
 (256, 512),
 (512,),
 (512, 784),
 (784,)]

In [7]: [np.abs(x).sum() for x in weights if len(x.shape) > 1]
Out[7]: [14079.087, 6261.0615, 2355.3237, 2363.5547, 6307.1421, 14264.825]

In [8]: sum([np.abs(x).sum() for x in weights if len(x.shape) > 1])
Out[8]: 45630.994140625
```

- Turns out that the total l1 norm of the weights of the matrix was on
  the order of 4E5, and my loss function was typically around 1E-1,
  1E-2. So, if I want my regularization to more-or-less evenly
  weighted with my loss function, then I have to use a regularization
  constant that makes the L1 norm around the same value as the base
  loss function, which is around 1E-7. That is the number that I found
  gave me sensible results when I first tried

- I think I understand the point of regularization a little bit
  better, now that I understand the idea of scaling your
  regularization and your original loss function. If there's a
  "typical magnitude" for your regularization, then you can set the
  regularization constant as something of a "threshold" for when your
  loss function stops mattering. When the value of the loss function
  outweighs your regularization, your loss function will focus on
  making moves that mainly reduce your loss function. When the loss
  function gets to around an equal or lower order of magnitude as
  compared to your regularization, the regularization term will start
  to matter more, and thus less will get done in terms of reducing the
  value of the original loss function. However, now that raises a
  question for me. Does it really matter what the regularziation used
  is? From that point of view, the regularization is just a threshold
  to stop optimizing beyond some point of "over-optimization". The
  regularization could have just been to stop working once the loss
  function drops beyond a certain point, or when it doesn't decrease
  much. Maybe different regularization techniques matter to different
  problems for different reasons, and I just haven't worked enough
  problems to really see this.

- Interesting. When I use a regularization of around 5E-7, the weights
  of the network are still pretty large. I wonder what the weights
  look like when I increase the regularization.

```
In [14]: [np.abs(x).sum() for x in weights if len(x.shape) > 1]
Out[14]: [13583.187, 6242.8652, 2443.0828, 2426.071, 6388.4077, 14366.514]

In [15]: sum([np.abs(x).sum() for x in weights if len(x.shape) > 1])
Out[15]: 45450.126953125
```

- It seems that the weights are around the same size, both with and
  without regularization, and both with 50 epochs and 20 epochs.

- It also seems that 1e-5 as L1 constant works with 512-256-128
  encoder. I wonder if regularization should be on the main encoder
  layer, or all the encoder layers. I wonder if it should be on any of
  the decoder layers.

- When I started using dropout, I had to change the way that I
  displayed info in my pictures, because that method didn't generalize
  to non-Dense layers.

- I managed to get the compression down to a 64 neuron layer. It
  worked pretty well. At this point, I'd like to start exploring
  sparse representations.

- Created an autoencoder with layers 1000-1200, and I'd like to make
  this sparse. The L1 norm for this autoencoder came out to what's
  listed below, though interestingly enough, this autoencoder had some
  regularization on it

~~~
In [4]: weights = autoencoder.get_weights()

In [5]: [np.abs(x).sum() for x in weights if len(x.shape) > 1]
Out[5]: [22807.045, 31479.285, 32072.518, 24184.715]

In [6]: sum([np.abs(x).sum() for x in weights if len(x.shape) > 1])
Out[6]: 110543.5625
~~~

- Upon further reflection, I realize that the regularization actually
  only applies to a layer, and not to the whole network, so I'm
  starting to measure the weights of the network as:

~~~
In [33]: [x.shape for x in autoencoder.layers[0].get_weights()]
Out[33]: [(784, 1000), (1000,)]

In [34]: [np.abs(x.get_weights()[0]).sum() for x in autoencoder.layers if x.activity_regularizer]
Out[34]: [31479.285]
~~~
