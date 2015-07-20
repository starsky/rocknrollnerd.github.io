---
layout: post
title:  "Oversimplified introduction to Boltzmann Machines"
date:   2015-07-18 15:05:00
categories: ml
tags:
comments: true
---



This is a continuation of the [previous](http://rocknrollnerd.github.io/ml/2015/07/14/memory-is-a-lazy-mistress.html) post dedicated to (eventually) understand Restricted Boltzmann Machines. I've already seen Hopfield nets that act like associative memory systems by storing memories in local minima and getting there from corrupted inputs by minimizing energy, and now... to something completely different.

The first unexpected thing is understanding that Boltzmann Machines are nothing like Hopfield nets, yet bear a strong resemblance to them. So, let's start with similarities:

 * A Boltzmann Machine is a network of binary units, all of which are connected together &mdash; just like a Hopfield net
 * There's an energy function, which is exactly the same as Hopfield's
 * When we update units, we use kinda the same rule of summing up all weighted inputs and estimating unit's output by the sum. We're going to use different activation function, though, instead of Hopfield's binary threshold one.

As for the differences, there are plenty of them too:

 * The main one, I guess, is that a Boltzmann Machine is **not** a memory network. We're not trying to store things anymore. Instead we're going to look for *representations* of our input data (like MNIST digits), i.e., some probably more compact and meaningful way to represent all training examples, capturing their inner structure and regularities.
 * And we're going to do that by adding an extra bunch of neurons called *hidden units* to the network. They are essentially just the same neurons as the other (*visible units*), except their values aren't directly set from training data (but are, of course, influenced by it via input connections).
 * There's another global objective &mdash; instead of simply minimizing the energy function, we're trying to minimize the error between the input data and the reconstruction produced by hidden units and their weights.
 * Remember how having local minima was fine for Hopfield nets because different minima corresponded to different memories? Well, that's not the case anymore. When we do representation learning, there's one and final objective we're trying to achieve, and therefore, local minima becomes an issue. To avoid it, we add noise to our network by making neurons' activations stochastic (that's what I meant by having "different" activation function before), so they could be active/inactive with some probability.

So... well, having stochastic neurons is a novel thing, but actually doesn't change the inner logic of the model much, does it? These new hidden units, however, do. Remember how we used to minimize energy in Hopfield nets? Right, by using Hebbian rule for each pair of neurons, and this worked because we knew exactly the value of each neuron (because it was set (or "clamped") by our training data examples). Now when there are hidden units, they are free from external interference, and their values are unknown to us. Hence, we cannot update their weights, hence, a problem.

# Representative democracy of hidden units

I've suddenly discovered that my previous metaphor of "voting" neurons can actually be useful again. Remember how units in Hopfield net used to cooperate with each other, voting for their neighbors to change their value? They don't vote personally now &mdash; instead, they use a group of established hidden units to represent their collective will. Now initially, when there's no training data, hidden units don't have an opinion of their own, they're like slick politicians waiting to hear the voice of the crowd (this metaphor starts to get out of hand). But when a training example is available, they have to mimic it as good as they can.

Well, how do they do it? To find out, we should use gradient descent. Because that's not Hopfield net anymore, we're trying to minimize not the energy function, but the discrepancy between the input data and the reconstruction. The objective function can be written as KL-divergence between two probability dictributions (input and reconstruction), and it turns out, that it's derivative is quite simple and equals to $$(s_{i}s_{j})_{+} - (s_{i}s_{j})_{-}$$ (derivation is provided by [original paper](http://web.archive.org/web/20110718022336/http://learning.cs.toronto.edu/~hinton/absps/cogscibm.pdf), which is surprisingly easy to read). The first term of the derivative corresponds to mutual neurons' activations in a so-called "positive phase" (when a network observes the training data), and the second one is the same but for "negative phase" (when a network tries to reconstruct the input). So, if you are a slick politician hidden unit trying to capture the many voices of the crowd, you do the following:

 * say something at random (activate hidden unit by a random set of weights)
 * when listening to the crowd, strengthen the connections to the people you happen to agree with (units with the same state as the hidden unit), and similarly, weaken the connections to the people that disagree with you.
 * later at home, trying to rehearse your speech, do exactly the opposite &mdash; weaken the connections to the units that are different from the hidden one, and vice versa.

This is indeed a very interesting learning rule, and while I understand it numerically, I still can't construct a good intuitive explanation for, say, why do we need the negative phase. In Hinton's course the second term is described as the unlearning term, but is that the same unlearning that Hopfield nets perform, and if so, why now is it possible to measure it precisely (wait, I know, that's because Hopfield nets are not error-driven)? The other reason to have the negative phase is that Hebbian learning in its simplest form is unstoppable, so the weights will grow until the learning is stopped. The negative phase allows to control this growth &mdash; for example, imagine we've contructed the ideal representation, so that the input data is equal to the reconstruction. Then, $$(s_{i}s_{j})_{+} = (s_{i}s_{j})_{-}$$, i.e. derivative becomes zero and the learning naturally stops. Still, that's surely not the one and only way to cap the weights...

I've made a simple toy Boltzmann Machine with binary threshold units that performs *only positive* phase, just to see what's happening in that case. One interesting thing apart from endlessly growing weights is that opposite-valued units in different states force some weights to stay zero. For example, if there's one hidden unit and three visible units, weights are $$(0, 0, 0)$$ and the network observes two states $$(-1, 1, -1)$$ and $$(1, 1, -1)$$, the first weight will never change from zero.

{:.center}
![][1hidden]

*My example network. Notice that it's actually a Restricted Boltzmann Machine, because there are no connections other then visible-hidden*

If we choose a different set of weights, like $$(2, 0, 0)$$, the opposite thing happens &mdash; the first weight will grow suppressing the other two.

Here's the class to play with:

{% highlight python %}
import numpy as np


class PositiveToyRBM(object):

    def __init__(self, num_visible, num_hidden, w=None):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        if w is None:
            self.w = np.zeros((num_visible, num_hidden))
        else:
            self.w = np.float32(w)

    def threshold(self, arr):
        arr[arr >= 0] = 1
        arr[arr < 0] = -1
        return arr

    def hebbian(self, visible, hidden):
        # for each pair of units determine if they are both on
        return np.dot(visible.reshape(visible.shape[0], 1),
                      hidden.reshape(hidden.shape[0], 1))

    def pp(self, arr):
        # pretty print
        return list([list(i) for i in arr])

    def try_reconstruct(self, data):
        h = self.threshold(np.dot(data, self.w))
        recon = self.threshold(np.dot(h, self.w.T))
        return np.sum(data &mdash; recon) == 0

    def train(self, data, epochs=10):
        data = np.array(data)
        for e in xrange(epochs):
            delta_w = []
            for example in data:
                h = self.threshold(np.dot(example, self.w))
                delta_w.append(self.hebbian(example, h))
            # average
            delta_w = np.mean(delta_w, axis=0)
            self.w += delta_w
            result = self.try_reconstruct(data)
            print 'epoch', e, 'delta w =', self.pp(delta_w), 'new weights =', self.pp(self.w), 'reconstruction ok?', result
{% endhighlight %}

# How do we daydream

Time to add the negative phase (also called daydreaming phase) term! And surprisingly, this is not so simple, because in the negative phase we need the network to be *free* from external interference, and... well, how do we do that? We generally cannot just set hidden units to random values with some fixed probability, because there will be no learning in that case. Turns out we should do the following: set units to random values and let them settle down by updating the units one at a time so that each unit takes the most probable value considering its neighbors. For example, if a neuron is surrounded by positive neighbors with positive weights, it will most likely become positive itself.

You can imagine that the weights define an energy landscape, and each state of the network corresponds to a certain point on it. When the state is set by external data example, the network is "forced" to keep the high ground, but what it "wants" to do is to become free of external influence and fall down to the (hopefully global) energy minimum (this is also called thermal equilibrium). The learning algorithm tries to make these two states the same by terraforming energy landscape (modifying weights) &mdash; this is actually the same process as in Hopfield nets.

Now we can implement it:

{% highlight python %}
def collect_negative_stats(self):
    # we don't know in advance how many loops required to reach equilibrium
    stats = []
    for e in xrange(10):
        # initial random state
        visible = self.threshold(np.random.rand(self.num_visible), 0.5)
        hidden = self.threshold(np.random.rand(self.num_hidden), 0.5)
        idx = range(self.num_visible + self.num_hidden)

        # settling for equilibrium
        # again, number of loops is guessed
        for _ in xrange(50):
            i = np.random.choice(idx)  # selecting random neuron
            if i < self.num_visible:  # visible neuron
                visible[i] = self.threshold(np.sum(self.w[i] * visible[i]))
            else:  # hidden neuron
                i -= self.num_visible
                hidden[i] = self.threshold(np.sum(self.w[:, i] * hidden[i]))

        # hopefully done, now make a reconstruction and collect stats
        recon = self.threshold(np.dot(hidden, self.w.T))
        stats.append(self.hebbian(recon, hidden))
    # average
    return np.mean(stats, axis=0)
{% endhighlight %}

And subsctract the stats from the value of `delta_w`:

{% highlight python %}
delta_w -= self.collect_negative_stats()
{% endhighlight %}

And now it works pretty nice, requiring just one pass to learn the correct reconstruction:


    if __name__ == '__main__':
        rbm = ToyRBM(3, 1, w=[[0], [0], [0]])
        rbm.train([[1, -1, 1], [-1, 1, -1]], with_negative=True)

    >> epoch 0 delta w = [[-1.0], [-1.0], [-1.0]] new weights = [[-1.0], [-1.0], [-1.0]] reconstruction ok? True

# A little bit more serious example

There are still some things left before we can apply our Boltzmann Machine to a "real" problem like representing MNIST digits.

 * First of all, our toy examples use deterministic activation function. While technicaly there's nothing wrong with it, our network becomes vulnerable to local minima, meaning we won't be able to reach "relaxed" equilibrium state. So we're going to replace our activation function with the coin toss of the following probability $$p(s_{i}) = \frac{1}{1 + e^{-\Delta E}}$$, where $$\Delta E$$ is the weighted sum of neuron's inputs ($$\Delta E =\sum_{j}w_{ij}s_{j}+b_{i}$$). Why this function exactly? This is another question I still haven't intuitively understood, but the answer is because Boltzmann distribution has a property that allows to express the probability of a single unit turning on by a function of its energy gap. The function is derived step-by-step right in [Wikipedia](https://en.wikipedia.org/wiki/Boltzmann_machine).
 * Next, so far we've used only visible-to-hidden connections. The original model assumes units are connected to each other, so we're going to add these hidden-to-hidden and visible-to-visible connections to the network.
 * This will slightly change the procedure, namely, the positive phase, because now hidden neurons can influence each other. We're going to apply the same logic here, by letting the network to settle down to the minima, updating hidden units only (because visible units are fixed to training data).
 * The negative phase won't change at all, just don't forget to take into account these new weights.

I've also tried to switch to 0/1 binary unit values, which, I guess, adds a bit of extra computation to the Hebbian update (which still should be 1/-1). The need to update units one at a time makes learning quite slow (a hint: computing random choices at once speeds up things), so I've used just a small subset of MNIST digits restricted to 3 classes. And it seems we're learning something:

<div class="photo_frame_center">
 <video width="650" height="250" controls preload="none"
  poster="/assets/article_images/2015-07-18-general-boltzmann-machines/poster.png">
  <source src="/assets/article_images/2015-07-18-general-boltzmann-machines/learning.webm" type='video/webm; codecs="vp8, vorbis"'>
 </video>
</div>
<br>

The unpleasant surprise is that showing more digit classes to the network makes the weights merge together in these ugly blobs (you can see 4 on one of the weights, and 1 elsewhere, but there's also 2 which is kinda missing). I didn't try to use simulated annealing, mainly because it's mentioned to be a distraction from how Restricted Boltzmann Machines work. Playing with different parameters sometimes gives interesting outcomes: for example, if we don't let the network to settle down enough in the negative phase, we get these clumsy weights:

![][negative-short]

But strangely, when do let it settle down *too much* (how much?), the weights get weird too:

![][negative-long]

That's really not the way it's supposed to work, so I guess I should debug my implementation. Which is available [here](https://github.com/rocknrollnerd/ml-playground), by the way.

# Quick RBM note

Now it's actually quite easy to understand how Restricted Boltzmann Machines are trained. In RBMs there are no hidden-to-hidden or visible-to-visible connections, so influence flows just between hidden and visible units. Meaning we can now update them in parallel &mdash; first compute hidden units' activations, then visible, then hidden again, and so on until the network settles down to equilibrium. That's called contrastive divergence. And it turns out, this method works even if we make *one* iteration of it &mdash; when the network is certainly not in equilibrium, but still gets updated in the right direction.

# Summary

Whoa, that took longer than I expected. But the incredible feeling of finally understanding if not every part of it, but certainly the main idea &mdash; that's absolutely worth looking up Hopfield nets and original Boltzmann Machines. Next thing I want to try, is to implement some different RBMs (convolutional, gaussian, softmax) and maybe compare their results with autoencoders, because now I'm starting to favor RBMs more, perhaps just because of the beauty of the concept. They don't even use backprop, how cool is that? I wonder if there are any attemps to discover similar positive-negative cycles in real neurons, which, of course, don't have symmetric connections, but still may constitute the same kind of structures. Or, are real neurons tend to "settle down" and minimize their energy? Oh snap, now I'm going to google it for hours.

[1hidden]: /assets/article_images/2015-07-18-general-boltzmann-machines/1hidden.png
[negative-short]: /assets/article_images/2015-07-18-general-boltzmann-machines/negative_short.png
[negative-long]: /assets/article_images/2015-07-18-general-boltzmann-machines/negative_long.png
