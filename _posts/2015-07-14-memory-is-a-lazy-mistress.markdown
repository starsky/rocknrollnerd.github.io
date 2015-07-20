---
layout: post
title:  "Memory is a lazy mistress"
date:   2015-07-14 15:05:00
categories: ml
tags:
comments: true
---

Trying to jump on the deep learning bandwagon, I often miss things. Sometimes I find my mind filled with models and algorihtms I hardly fully undestand: they become obscure concepts and fancy buzzwords. That actually bothers me, so I've decided to make a couple of detailed runs across the stuff I'm kinda already familiar with &mdash; this time, Restricted Boltzmann Machines. And turns out that the best way to understand something for me is to write it down in a stupidly oversimplified, tutorial style, leaving out intimidating equations and trying to make small code examples all the way through. So that's what I'm going to do now. Maybe someone else will find it helpful too.

What do I *already* know about RBMs? They are models that perform a kind of factor analysis on input data, extracting a smaller set of hidden variables, that can be used as data representation. First time when I encountered RBMs, I wasn't quite excited about it &mdash; after all, there are *lots* of representation algorithms in machine learning, including autoencoders, that are simple perceptrons and can be learned via already familiar backprop. RBM is different by being both stochastic (its neurons' values are calculated by tossing a probabilistic coin) and generative &mdash; meaning that it can generate data on its own after learning. Aaand basically that's what I start with.

So let's start with amazingly complete yet sometimes still-hard-for-me-to-grasp Hinton's course of [Neural Networks for Machine Learning](https://class.coursera.org/neuralnets-2012-001), lections 11 and 12. And surprisingly, the journey begins with Hopfield nets.

# Memory networks

Hopfield net is a bunch of binary neurons that can take values of 1 or -1. Each neuron is connected to every other neurons except itself, the weights are real-valued and symmetric &mdash; so if there are $$N$$ neurons, there going to be $$\frac{N(N -1)}{2}$$ weights.

Now, there's a function that calculates a special scalar property of a network, called the **energy function**. To obtain the energy for a single neuron, we do the following:

 * for all neighboring neurons (there are exactly $$N-1$$ of them), multiply each neighbor's value by the value of its weight and by the value of current neuron.
 * simply add them together with the minus sign.

The energy of the whole network is basically just the sum of these terms, calculated for each neuron. Let's forget about why exactly this value is called "energy" for a moment and think about how it depends on the state of the network. Suppose we have a neuron $$s_{i}$$, one of its neighbors $$s_{j}$$ and a weight $$w_{ij}$$. Their product $$s_{i}s_{j}w_{ij}$$ is a part of a global energy value, and when it's positive, global energy goes down (remember a minus sign). Now let's think about the weight as something we can adjust to be the way we want to and concentrate on neurons' values. The product is positive if both of them are positive or negative at the same time, and weight is positive too. Or otherwise, if both values are different and weight is negative. Now, that does actually look familiar... because that's the famous [Hebbian learning](https://en.wikipedia.org/wiki/Hebbian_theory) rule &mdash; neurons that fire together, wire together (i.e., have positive connection). So one thing we learn for now is that energy dynamics depend on neurons' behaviour &mdash; if neurons that "agree" with each other are connected by positive weights, and neurons that "disagree" are connected by negative weights, energy goes down, otherwise it goes up.

Now why does that matter? Because minimizing the energy function is exactly the thing we're going to do when learning weights for a Hopfield net. The purpose of training is, however, different from the usual supervised learning objective: we're not comparing data to some labeled target, but instead trying to *store* it in the network so that the state of the network corresponds to energy minimum. And the storage rule is quite simple: set the neurons to the values of our data vector (say, a single binary image), and update the weights the following way:

$$
    w_{ij} = w_{ij} + s_{i}s_{j}
$$

Time to write a piece of code!

{% highlight python %}
class HopfieldNet(object):

    def __init__(self, num_units):
        self.num_units = num_units
        self.w = np.zeros((num_units, num_units))
        self.b = np.zeros(num_units)

    def store(self, data):
        data = _data.reshape(data.shape[0], 1)
        activations = np.dot(data, data.T)
        np.fill_diagonal(activations, 0)  # because there are no connections to itself
        self.w += activations
        self.b += data.ravel()

    def get_energy(self, data):
        # first let's again compute product of activations
        data = _data.reshape(data.shape[0], 1)
        activations = np.float32(np.dot(data, data.T))
        np.fill_diagonal(activations, 0)
        # then multiply each activation by a weight elementwise
        activations *= self.w
        # total energy consists of weight and bias term
        weight_term = np.sum(activations) / 2  # divide by 2, because we've counted neurons twice
        bias_term = np.dot(self.b, data)[0]
        return - bias_term - weight_term
{% endhighlight %}

**Some things I've purposely left out:**

 * biases (notice `self.b = np.zeros(num_units)`). If you're already familiar with neural networks they work just the same way here. If not, look up a [great explanation](http://stackoverflow.com/questions/2480650/role-of-bias-in-neural-networks) why we need biases.
 * data is formatted as binary and takes values of 1 in -1.

Let's put a MNIST digit in it (resized to 8x8 px). If we try to visualize the network, it kinda looks like a messy hairball...

![][hopfield-full]

*Red weights are positive and blue weights are negative, but they're too messed up to see any pattern*

So let's instead show only positive weights between positive neurons.

![][hopfield-positive]

Active neurons are highlighted in purple. It seems that negative neurons *inside* the zero symbol have incoming positive connections too, but that's just an overlay &mdash; they really are not connected to any positive neurons.

Now, what do we see here? The storage rule has brought us exactly to the state Hebbian learning predicts: positive neurons "support" each other. Negative units, not shown here, also support each other, and (guess what) these two fractions have negative connections between each other's units too.

So let's think about any possible application of such a network, like, what can we do with all this support? Literally the first thing that comes to mind is error correction. For example, there are 16 positive neurons. Suppose one of them has flipped its state and become negative, and we want to check what state it actually should be in. Sometimes it's convinient for me to think about a neural network as a council of voters, and here's what happens in that case:

 * a neuron asks 15 of its *positive* neighbors something like "should I behave like you?" and they tell him "yes, you should".
 * then, a neuron asks 49 of its *negative* neighbors the same question and they tell him "no".
 * so, our neuron receives a total score of 63 votes that tell it to became positive and immediately does so, changing its initial value.

63 to 0 vote ratio leaves our neuron literally no doubt, and that also means we can afford to have more errors in our corrupted data. If another one of positive neurons flips and becomes negative, the ratio would be 62 to 1, and so on. So we actually can put just a small chunk of original data vector (or a piece of image) and still be able to correctly restore it. That is what's called *associative memory* &mdash; a kind of memory that can be restored by observing just a tiny part of it. It is believed to be the kind of memory (at least part of it) we humans have, because we're incredibly good at recognizing wholes by parts.

Let's formalize our voting procedure to get the restoring rule for Hopfield nets:

 * for each neuron, compute a weighted sum of all its inputs, i.e. $$\sum\limits_{j=1}^{N-1} s_{j}w_{ij}$$.
 * if this sum $$>0$$, set neuron's value to 1, else to -1...
 * ...or alternatively, compute and store network's energy $$E$$, then flip a neuron and compute the energy again after that state change ($$E_{new}$$). If $$E_{new}<E$$, remember the new energy minimum and keep the change, otherwise do nothing and pick another neuron to try.

The amazing thing here is that these two rules are equivalent (because we've defined energy to support the agreement between neurons that vote the same way and to repel neurons that disagree). The first rule, however, is much cheaper computationally because it only requires local information (and the second rule requires access to all the network).

Let's implement it:

{% highlight python %}
def restore(self, data):
    data = np.copy(_data)
    idx = range(len(data))
    # make 10 passes through the data
    for i in xrange(10):
        for _ in xrange(len(data)):
            j = np.random.choice(idx)
            inputs = np.sum(data * self.w[j])
            if inputs > 0:
                data[j] = 1
            else:
                data[j] = -1
    return data
{% endhighlight %}

And try to restore something corrupt:

<div class="photo_frame_center">
 <video width="650" height="500" controls preload="none"
  poster="/assets/article_images/2015-07-14-memory-is-a-lazy-mistress/poster.png">
  <source src="/assets/article_images/2015-07-14-memory-is-a-lazy-mistress/restore.webm" type='video/webm; codecs="vp8, vorbis"'>
 </video>
</div>
<br>

# How many errors we can make

If you are familiar with the concept if [error correction coding](https://en.wikipedia.org/wiki/Forward_error_correction), you should be, at least for now, a bit disappointed. Think about it this way: error correction becomes possible when we carry some extra information with our precious data. The amount of this information can be determined by the means of information theory and it's usually not much, because data transfer and storage cost money and resources. With a Hopfield net we can correct literally *any* amount of errors (you can start from blank image and still get your correct answer, when only one image is stored), but we pay the quadratic price of $$\mathcal{O}(N^2)$$ (each neuron is connected with each other neuron, remember), and that's quite a lot.

The cool thing that really surprised me is that we can store *multiple* memories in the same network, and we don't even have to modify our storage rule. Simply apply it again for a new piece of data, and than again and so on. To understand how does this work, let's get back to our voting example again:

 * Remember, we calculated $$\Delta w_{ij}$$ as the product of neurons' activations, so it could be either 1 or -1.
We can say that was "one weight one voice" model, meaning that absolute values were the same.
 * Now when we apply the storage rule one more time, weight values accumulate. Some of the weights (that connect the same active/inactive neurons for both images) will end up having values of +/- 2, meaning that their vote costs more now.
 * Think about these "doubled" weights as values the network is certain about. For example, if we'd like to continue storing different MNIST digits, each of them would have a negative border (the background). All the agreement connections between these background neurons will add up, and when it's time to ask neighbor voters about a certain neuron's state, it will receive strong support from every one of them. The network kinda tells us "I don't know what MNIST digit that neuron belongs to, but it really should be negative anyway".
 * Other neurons now start competing with each other by gathering votes in their support. If a certain neuron should be active in one image and inactive in the other, here's what happens: a neuron asks for support from its neighbors and they divide in two fractions, that tell it to switch on or off. The fraction which casts more votes wins and subdues a neuron to its collective will. And of course, if some of the neurons from both fractions are corrupted too, that messes up the output decision.

*Now* you can see that network's ability to correct errors has been decreased. We cannot restore images from almost random noise now, because we don't know which fraction is going to prevail. We have to show the network a relatively distinct piece of image to obtain the correctly restored version of it, and that really looks more like actual memory now.

# Learning backwards

There are additional complications, though: turns out, different memories can merge together when they correspond to the same local minimum (another cool thing about Hopfield nets is that we're not trying to escape local minima anymore, because they are memory storage locations). It's been shown that you can store about $$0.138N$$ memories in $$N$$-neuron net, but my MNIST example actually breaks at third &mdash; I guess, that's because some memories (0 and 2, for example) are partially similar (sic).

![][hopfield-mixed]

To avoid the issue, you can use a curious technique called *unlearning* or *reverse learning*, which is basically this: you set the network to a random state, and then apply the same Hebbian learning rule but with the minus sign. The idea of reverse learning actually was introduced before Hopfield nets by Crick (no less) and Mitchinson, who considered it to be a possible theory of dreams. I'd certainly like to read something on the matter, partially because (getting a little bit ahead of myself) unlearning takes an important part in Boltzmann Machine learning, but mostly because it looks like a awesomely cool concept. And, by the way, the whole concept of memories as energy minima, too! As George Carlin said, *"When I first heard of entropy in high school science I was attracted to it immediately"*.

I didn't try to implement the unlearning procedure, mostly because I felt already dug too much into Hopfield nets. Full implementation is available [here](https://github.com/rocknrollnerd/ml-playground). Next stop is Boltzmann Machine station!


[hopfield-full]: /assets/article_images/2015-07-14-memory-is-a-lazy-mistress/hopfield-full.png
[hopfield-positive]: /assets/article_images/2015-07-14-memory-is-a-lazy-mistress/hopfield-positive.png
[hopfield-mixed]: /assets/article_images/2015-07-14-memory-is-a-lazy-mistress/mixed.png
