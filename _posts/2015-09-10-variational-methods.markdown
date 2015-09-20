---
layout: post
title:  "Probability and variational methods, part 1"
date:   2015-09-20 20:11:00
categories: ml
tags:
comments: true
---

Apparently what's started as a couple of lectures and assignments has turned into completely different understanding of probability and machine learning I'm still trying to digest. So this is going too be a big (and possibly really oversimplified) post.

# Everything is random

We start from the idea that the world around us, natively represented by our sensors, can be compressed into some kind of collection of smaller representations.

The truth of this assumption is not actually self-evident. We can roughly estimate brain's storage capacity by counting possible number of connections between its neurons and maybe somehow estimate the amount of bits of information we experience on a daily basis (the latter seems kinda difficult, though) and compare them. We can also suggest that the universe around us is not by itself unique and is composed of some object of repeated and similar structure, so at least *in theory* there's some room for compression and it would be natural for the evolution to take the opportunity to exploit this structure and save computational power of the brain. We can also argue that the usual tasks our brain performs, including classification, recognition and pattern matching, assume existence of some kind of common ground for the objects in the same cathegories (Imagine you have to make a classification rule for completely different objects/data vectors. You'll need *extra* bits of information to explicitly map all the objects to their categories, if there's nothing common in the data *itself*).

Anyway, we're going to stick with the assumption for a while. There are some parameters that represent the structure of the data, they can be arranged in lots of different ways, maybe as a graphical model of some kind, maybe interacting with each other or maybe just a bunch of numbers that correspond to inner mechanics of the data. Doesn't matter for now. Let's call the parameters $$\theta$$ and the data $$X$$, without any specifics considering their form (scalar/vector/matrix/etc).

Now let's assume the data is random.

This also takes some time to digest. At least, it did for me, when I first thought about it, mostly because by "random" we usually mean something like complete mess, and the world we perceive seems solid, structured and deterministic (at least at large scale). This is, of course, a miscomprehension (actually, a single number can be a random variable, considering it corresponds to [delta distribution](https://en.wikipedia.org/wiki/Dirac_delta_function)). By saying that the data is random we just mean that there's some kind of noise in our observations, which may correspond to the flaws of our sensors, or to the unpredictable small changes in the world itself. When we see a bunch of handwritten examples of some digit, each image is unique, written slightly differently, and yet they all share the same structure, which corresponds to the underlying distribution of the data. Naturally, we'd like to configure the parameters $$\theta$$ so that they would describe this distribution and capture our knowledge of the world.

Suppose we toss a coin twice, not knowing anything about coins in advance, and it comes up first heads, then tails. Suppose we have the simplest parameter possible $$\theta$$ which is a number from 0 to 1 that denotes the probability of heads. What value of $$\theta$$ should we choose so that our model would be consistent with the data? Considering our one-parameter model, and the fact that two tosses are independent, we can express the probability of observed data given parameters as $$P(X\mid\theta)=P(H)P(T)=P(H)(1-P(H))=\theta(1-\theta)$$. One possible way to find a good value of $$\theta$$ then is to pick the value that *maximizes* that probability, which is actually the likelihood function (and the method is called **maximum likelihood estimation** or MLE). Let's do that by the dumbest way possible: by brute-force search.

{% highlight python %}
import numpy as np

ml = 0
theta = 0

for t in np.linspace(0, 1, 1000):
    likelihood = t * (1 - t)
    if likelihood > ml:
        ml = likelihood
        theta = t

print theta

# >> 0.49949949949949951
{% endhighlight %}

Okay, that was suddenly not surprising. And of course, there's no need to iterate over all possible values of $$\theta$$ when we can find the maximum simply by differentiating the function, like $$F'(X\mid\theta)=(\theta - \theta^{2})' = -2\theta + 1$$, which yields $$\theta=0.5$$. Plain and simple. It can be shown that lots of machine learning methods are performing maximum likelihood estimation, sometimes without using probabilistic vocabulary at all, like simple feed-forward neural networks, for example.

Now let's imagine a different example. Suppose a coin is tossed twise again, but now it results in two heads. Following the same path as before, we get the equation for the likelihood $$P(X\mid\theta)=\theta^{2}$$, and its derivative $$2\theta$$, which points to maximum at $$\theta=1$$. And this is perfectly right in terms of maximizing the likelihood, but doesn't look like a good answer in general (do we really expect the coin to show up heads only after just two tosses?). One solution to this problem is to always get lots of data before tweaking your model, and this is perfectly good advice we'd like to follow anytime possible, except sometimes it's just might not be possible. If we're dealing with the data that can result in multiple outcomes, each missing outcome would have probability 0 estimated by the model (like the probability of tails in the second coin example is 0), and that might be too strong of an assumtion to make. The other problem is that our data has to be *equally partitioned*. Think about the case when we perform 100 coin tosses and obtain 48 heads, when waximum likelihood estimate would be 0.48: good estimate, but still wouldn't the probability of 0.5 be more likely? (or at least as good?)

To tackle this issue we're going to allow some uncertainity in our answers. Concretely, we're going to make $$\theta$$ a random variable and instead of trying to find a single possible good value of it we're going to estimate its distribution given the observed data, or $$P(\theta \mid X)$$. This is what's called Bayesian inference, and naturally, the main tool we're going to use to estimate this distribution is Bayes' rule: $$P(\theta \mid X)=\frac{P(X \mid \theta)P(\theta)}{D(X)}$$.

So not only the data is random, but our model estimate is random *as well*. Nice. Let's try this approach for the second coin example.

*(Note: there are other reasons why full Bayesian approch may be better then MLE. A well-known example is a cancer test that detects cancer with very high probability, but still is practically useless without considering the prior probability of getting cancer. Usually, however, we don't have a good prior distribution for parameters $$\theta$$, or we start from the uniform prior, so that's not the main reason to go Bayes, I guess)*

# An impractical example

Since we don't know anything about coins, let's start from the uniform distribution as the initial estimate, meaning that we expect any value of $$\theta$$ between 0 and 1 with the same probability.

{:.center}
![][uniform]

After observing one head, here's what happens: $$P(\theta \mid \{H\})=\frac{P(H\mid \theta)P(\theta)}{P(H)}$$. To find the numerator, we have to multiply our initial distribution $$P(\theta)$$ by the likelihood of obtaining a heads given this particular value of theta. How do we get the likelihood, this $$P(H\mid \theta)$$ term? Well, we've chosen the $$\theta$$ wisely to define the probability of heads, so it's easy to see that $$P(H\mid \theta)=\theta$$. Let's plot the numerator expression:

{:.center}
![][unnormalized]

Notice that's not a probability distribution: this is an *unnormalized* posterior, since the area under the line doesn't sum to 1. We have to normalize it by dividing it by the denominator, $$P(H)$$, or the *evidence* term. This is simply the integral over $$\theta$$ of this unnormalized posterior triangle: $$\int_{0}^{1} P(H\mid \theta)P(\theta) d\theta$$, or in our case, this is simply the half of the unit-square, which has the area of $$\frac{1}{2}$$.

{:.center}
![][normalized]

Let's make another coin toss: and thanks to the fact both tosses are independent, we can perform updates sequentially, using $$P(\theta \mid \{H\})$$ as the new prior for calculating $$P(\theta \mid \{H, H\})$$. The likelihood stays the same, so basically we just multiply our distribution by $$\theta$$ again and normalize it:

{:.center}
![][two_tosses]

Look how much this is better then maximum likelihood estimate! This distribution is still biased towards probability of heads, but there's enough uncertainity to allow other explanations. You can make a couple of integrations to see the actual probabilities: for example, $$P(0.4 \leqslant \theta \leqslant 0.6)=0.152$$ and $$P(\theta \geqslant 0.9)=0.271$$, so the difference is sifnificant, but not quite one-vs-all.

Let's run it for some more steps, now using actual random number generator and see where it'll take us:

<div class="photo_frame_center">
 <video width="612" height="512" controls preload="none"
  poster="/assets/article_images/2015-09-10-variational-methods/uniform.png">
  <source src="/assets/article_images/2015-09-10-variational-methods/learning_coin.webm" type='video/webm; codecs="vp8, vorbis"'>
 </video>
</div>
<br>

Notice how the spread of the distribution stops changing after the certain amount of time, meaning that we're still allowing uncertainity to affect our decisions.

Okay, that was great, but it's actualy even better, because full Bayesian learing makes some problems specific to maximum likelihood estimation magically disappear. For example, we aren't afraid to be stuck in the local optima during gradient descent, because we're not looking for a single best set of parameters anymore. We can also make use of some prior knowledge of the model - for example, if we're dealing with a neural network, we can start with some assumption that the parameters, i.e. weights, shouldn't be very large.

Well, let's try to use it for something that looks more like a machine learning model.

# A slightly less impractical example

Consider this super-small neural network with two input units and one output unit:

{:.center}
![][toy_nn]

Where parameters of the model $$\theta$$ correspond to the weights of the network $$[\theta_0, \theta_1]$$. There's not much stuff this tiny network can do, but we'll give it a toy problem, for example, function as logic OR gate. But first, lets give this model a prior parameter distribution.

I use these [excellent slides](http://www.cs.cmu.edu/afs/cs/academic/class/15782-f06/slides/bayesian.pdf) by Aaron Courville as a source of the problem setup, but instead of using Gaussian prior, as he does, I'm going to stick with the uniform - just because I can. So,

$$
    P(\theta)=P(\theta_0, \theta_1) \sim Uniform(-5, 5)
$$

Looks just like our coin toss example prior, just 3D. Nothing fancy.

{:.center}
![][3d_prior]

As for the likelihood, that's easy too: we're going to use sigmoid function to output a number between 0 and 1 which can conviniently represent probability.

$$
    P([X_i, t_i] \mid \theta)=
    \begin{cases}
    \frac{1}{1+e^{-X_i \theta}}, & \text{if } t_i = 1\\
    1 - \frac{1}{1+e^{-X_i \theta}}, & \text{if } t_i = 0
    \end{cases}
$$

Where $$i$$ denotes $$i$$-th observation/dataset element. Let's pick an observation, for example, $$[(0, 1), 1]$$), and plot the unnormalized posterior (which is again, a product of the prior and the likelihood):

{:.center}
![][3d_unnormalized]

Nice! Then we're going to normalize, and again, and so on, and the process actually looks very similar to the coin example:

<div class="photo_frame_center">
 <video width="612" height="512" controls preload="none"
  poster="/assets/article_images/2015-09-10-variational-methods/3d_prior.png">
  <source src="/assets/article_images/2015-09-10-variational-methods/logic_or.webm" type='video/webm; codecs="vp8, vorbis"'>
 </video>
</div>
<br>

The only difference is that our dataset now is noise-free (I simply generated a bunch of number pairs and logical-OR-ed them), so that the posterior peak doesn't jump around.

At this point it might became obvious what is the main problem about Bayesian approach to machine learning: you have to evaluate that posterior for *every combination of parameters* which your model allows. And when we use graphical models and neural networks, especially deep ones, we usually have *lots*, hundreds and thousands of parameters. Think about it this way: when we're doing maximum likelihood learning via gradient descent, having an extra parameter means we have to compute an extra derivative. But if we're doing full Bayesian learning it means we have to compute all the combinations that this new parameter makes with the other parameters, so the total number of computations is

$$
    (possible \: parameter \: range)^{number \: of \: parameters}
$$

which grows exponentially. Not cool.

# Variational Bayes and Boltzmann machine example

So the idea is that instead of computing exact posterior, which is often (for sufficiently complicated models) intractable, why don't we resort to some approximation of the posterior, which doesn't involve exponential number of computations? This is called *variational approximation* and we're going to illustrate this idea with yet another example - now with Boltzmann machine model.

Just as a reminder, the likelihood function for Boltzmann machine is written with respect to the energy of the model and looks like this:

$$
    P(x \mid \theta) = \frac{e^{-E(x)}}{Z}
$$

where $$Z$$ is yet another partition function, which equals to the sum over $$e^{-E(x)}$$ for all possible data vectors $$x$$. So the likelihood alone involves $$2^N + number \: of \: hidden \: units$$ computations, and we still have to normalize it by $$P(X)$$, which is the sum over all parameters $$\theta$$. So not only Bayesian learning for Boltzmann machine is a nightmare; even exact maximum likelihood learning is often intractable (that's why we use Gibbs sampling and tricks like conrastive divergence).

So let's step back a bit and instead of trying to approximate the posterior first deal with the likelihood. After all, we can approximate any distribution we want. So there's $$P(X \mid \theta)$$ and some approximation distribution $$q(X \mid \mu)$$ where $$\mu$$ are variational parameters. For simplicity I'm going to denote the distributions just as $$P(x)$$ and $$q(x)$$, using little $$x$$ so that it means single observations and cannot be confused with the evidence term. So, to make the approximation close to the original, we're going to minimize the KL-divergence term:

$$
    \begin{align*}
    KL(q(x)||P(x)) & = \sum_{x} q(x) \log \frac{q(x)}{P(x)} dx \\
                   & = \sum_{x} q(x) \log q(x) dx - \sum_{x} q(x) \log P(x) dx \\
                   & = -H(q) - \sum_{x} q(x) \left[ -E(x) - \log Z \right] \\
                   & = \sum_{x} q(x) E(x) - H(q) + \log Z \\
                   & = \mathbb{E}_{q(x)} \left[ E(x) \right] - H(q) + \log Z

    \end{align*}
$$

where $$H$$ means information entropy, and $$\mathbb{E}_{q(x)}$$ means expected value with respect to $$q(x)$$ (I borrowed the derivation from these excellent [slides](http://cvn.ecp.fr/personnel/iasonas/course/DL4.pdf)).

Notice that minimization of KL-divergence allows us to ignore $$\log Z$$, because it doesn't depend on $$q(x)$$. A popular choice for $$q(x)$$ is a fully-factorized distribution, a.k.a *mean field* distribution, which assumes that the probability of an overall configuration is just the product of the probabilities of each binary unit to turn on:

$$
    q(x) = \prod_{i} q_{i}(x_{i}) = \prod_{i} \mu_{i}^{x_{i}} (1 - \mu_{i})^{1-x_{i}}
$$

So each unit is represented by a single value $$\mu_{i}$$ when it's on and $$1-\mu_{i}$$ when it's off. The it can be shown that mean field update equations look like this:

$$
    \mu_i=\sigma \left( \sum_{j\neq i} w_{ij}\mu_{i} \right)
$$

(I couldn't quite follow the derivation, although I really wanted to). The other quite obscured thing is the reason we cannot use these equations during the "negative phase" of Boltzmann machine learning: there's brief note on it in Hinton & Salakhutdinov [paper](http://jmlr.org/proceedings/papers/v5/salakhutdinov09a/salakhutdinov09a.pdf), but again I couldn't follow the reason why this is true. So to summarize:

1. Each hidden unit gets a parameter $$\mu_i$$ which is a number between 0 and 1 that denotes the probability of that unit to turn on.
2. During the positive phase we can update all units in parallel, using mean field equations.
3. During the negative phase we do the same as we used to (sequential updates). There are tricks to speed up the negative phase too (in fact those two tricks are the basics of Deep Boltzmann Machine learning algorithm I hope to get back to one day), but let's ignore them for now.

So I took all this and put it into my [previous attempt](http://rocknrollnerd.github.io/ml/2015/07/18/general-boltzmann-machines.html) to make a general toy Boltzmann machine. To monitor the performance, however, I cannot use visualized weights anymore (because our parameters are not just weights now, but mean-field values too), so the only way to see if the model actually learns something is to sample data from it.

{:.center}
![][bm_samples]

*Left: samples from the original Boltzmann machine. Right: samples from mean field-approximated Boltzmann machine. Both models observed a toy subset of MNIST digits that included only 3 classes*

# Summary

Okay, that took longer than expected. I sincerely hoped working through mean field for Boltzmann machines will get me closer to understanding variational methods, but apparently it was a poor choice to start with, because of all the difficulties: not being able to derive mean field equations (which was the most frustrating), the confusing part about still doing weight updates during the negative phase, and after all, that whole thing was a distraction from full Bayesian approach, because the model still does maximum likelihood learning (we end up with one single value of $$\mu_i$$ per each unit). Models I'm mostly interested in are [NADE](http://jmlr.csail.mit.edu/proceedings/papers/v15/larochelle11a/larochelle11a.pdf), [variational autoencoders](http://arxiv.org/pdf/1312.6114v10.pdf) and [Bayesian neural networks](http://arxiv.org/pdf/1505.05424v2.pdf), which were discussed on the DTU summer school and have really got my interest, so I hope I'll get to them in the next part.


[uniform]: /assets/article_images/2015-09-10-variational-methods/uniform.png
[unnormalized]: /assets/article_images/2015-09-10-variational-methods/unnormalized.png
[normalized]: /assets/article_images/2015-09-10-variational-methods/normalized.png
[two_tosses]: /assets/article_images/2015-09-10-variational-methods/two_tosses.png
[toy_nn]: /assets/article_images/2015-09-10-variational-methods/toy_nn.png
[3d_prior]: /assets/article_images/2015-09-10-variational-methods/3d_prior.png
[3d_unnormalized]: /assets/article_images/2015-09-10-variational-methods/3d_unnormalized.png
[bm_samples]: /assets/article_images/2015-09-10-variational-methods/bm_samples.png