---
layout: post
title:  "Finally, RBMs: part 1"
date:   2015-07-23 15:05:00
categories: ml
tags:
comments: true
---

Checking my progress: about 10 days dedicated to energy-based models, including Hopfield nets, Boltzmann machines and their little sisters RBMs. And I'm starting to notice that I can read and understand (some) machine learning papers now, not just scroll through all the equations in a panic. Another thing I've also discovered during the run, is that among other concepts I find probability ones to be the hardest to undestand. Until it was all single neurons and energy landscape, all was loud and clear, but "to get an unbiased sample from the posterior" still looks a bit vague to me. So now I'll try to watch the amazing Probabilistic Graphical Models course to get closer with sampling, Markov chains and belief nets.

And now, let's play with some RBMs.

**A short disclaimer**: I'm going to make the following implementations to be as simple as possible - mainly because RBM is a popular model and to add all the bells and whistles like momentum, mini-batch gradient descent and validation comes close to reinventing the wheel. So these pieces of code are mostly educational. However, I don't like the idea of concentrating on toy problems too much, so I'm going to try to make a full-sized implementation in Theano (since I've made a few baby steps with it already) and try to run it on GPU (since it can't handle Witcher 3 in all its glory).

# Bernoulli RBM

This is the simplest form of RBM with binary hidden and visible units (where Bernoulli means binary, naturally). There are a few difficulties, however, considering different ways of collecting statistics for contrastive-divergence learning: sometimes we can use activation probabilities, and sometimes sampled binary states. Hinton's [practical guide](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf) (I absolutely love the idea of making such a document) specifies it in details, but at first I've found the section on statistic collection kinda obscure (and I'm [not alone](http://stats.stackexchange.com/questions/93010/contrastive-divergence-making-hidden-states-binary)). But after googling some other implementations everything came in its place.

 * First, assume for the sake of simplicity, that we're going to use CD-1 (i.e., perform one pass of contrastive divergence)
 * When performing CD-updates, always sample the states except maybe for the last hidden units update (because you won't need them; that's just for economy. I've actually performed sampling on every update).
 * When calculating statistics, you can use probabilities everywhere except for the input data (which is fixed and binary) and the first hidden state vector $$h_{0}$$. Our you can just use states everywhere and be fine.

I've chosen 100 hidden units, clamped a subset of MNIST dataset to them, and here's the result:

![][dense-binary]

Looks kinda feature-ish, but a bit messed up. Let's try to add some sparsity to make hidden units represent more independent features. The basic way to to it is described by [Honglak Lee](http://web.eecs.umich.edu/~honglak/nips07-sparseDBN.pdf), and it's just adding $$\rho - mean(h)$$ to weights and hidden biases, where $$\rho$$ is the desired sparsity target, meaning average probability of a unit to fire. *Actually* it is possible to add that term only to hidden biases, and we'll get back to it in a moment.

Sparse RBM filters looks like this:

![][sparse-binary]

I'm still not sure wheter that's cool or rather not - on the one hand, average hidden unit's activation definitely go down, and units start to represent different (independent) features. On the other hand, too many of them look like "1" - maybe that's because those diagonal straight strokes are tend to occur in "1" and "7" quite often. I also would like to end up with nice penstrokes-like features like those pictured in Lee's paper, and after some digging discovered that whitening (preprocessing operation that make different image pixels less corellated with each other) helps to achieve something like that. I couldn't apply it right now, because I needed my data to be real-valued, and *that*, in turn, required my visible units to be Gaussian.

![][wdwhb]

# Gaussian RBM

At first I'd like to point out that I haven't found literally any analysis on the matter. Energy function for Gaussian RBM is just stated (by Hinton, Lee and some other papers I've managed to google), and no further comments are made. That's really disappointing *especially* because everyone keeps saying Gaussian RBMs are hard to train. And, by the way, this is so true.

First, it doesn't even work at all with uniformly initialized $$[-\frac{1}{m}, \frac{1}{m}]$$ weights. To make it learn, I had to change replace them with normaly distributed weights with zero mean and 0.001 standart deviation (thanks for practcal guide again). Any attempt to increate the std value breaks learning like completely.

Oh, and I forgot to mention the actual change: for visible units I've replaced the activation function with sampling from normal distribution of $$hw + b_{vis}$$ mean and unit variance. To be able to do that I had to rescale the input data to zero mean and unit variance (otherwise it's also possible to learn precise variance parameters per each unit). Also I guess I can't use "raw" $$hw + b_{vis}$$ value to collect learning statistic (as I did with Bernoulli probability), so I'm going to use the states everywhere.

According to my (kinda sloppy) observations sparsity doesn't work so good for Gaussian RBM either - adding sparsity penalty to the weights seems to push the gradient in the wrong direction maybe? Anyway, average hidden unit activation doesn't change properly. I've followed Lee's advice about adding sparsity penalty just to visible biases, and now it works better.

The results are... slightly better as well, I guess:

![][sparse-gaussian]

*(this is with sparsity enabled)*

I can distinguish specific digits, but not penstrokes, still. Let's whiten the data and try it again:

![][whitened-data]

*Whitened MNIST digits. Can't say they actually change a lot*

![][whitened-weights]

*Whoops*

...and this is where I'm stuck at the moment. This doesn't look like features in any way, and I don't understand what is the problem. The curious thing about it is that the model still learns, reconstruction error goes down - but it learns these pretty much random noise filters, still being able to make better reconstructions.

I'm going to pause for a bit and maybe look for external help. All the cases mentioned are uploaded in my playground repo.


[dense-binary]: /assets/article_images/2015-07-23-finally-rbms-part-1/dense-binary.png
[sparse-binary]: /assets/article_images/2015-07-23-finally-rbms-part-1/sparse-binary.png
[sparse-gaussian]: /assets/article_images/2015-07-23-finally-rbms-part-1/sparse-gaussian.png
[wdwhb]: /assets/article_images/2015-07-23-finally-rbms-part-1/wdwhb.jpg
[whitened-weights]: /assets/article_images/2015-07-23-finally-rbms-part-1/whitened.png
[whitened-data]: /assets/article_images/2015-07-23-finally-rbms-part-1/whitened_digits.png