---
layout: post
title:  "Deepdreaming pets"
date:   2015-08-22 9:00:00
categories: ml
tags:
comments: true
---

So I've tried to make an attempt against Kaggle's cats and dogs dataset as I [intended to](http://rocknrollnerd.github.io/ml/2015/08/20/i-want-my-corners.html). In short, some good and bad news:

 * I've managed to achieve 75% success recognition rate, which is, to be honest, not good at all for binary classification. Still I hoped that'll give me some features to look at.
 * Implementing a deconvolutional network in Theano is a tricky business. There's an [issue](https://github.com/Theano/Theano/issues/2022) on Github about in with no progress, it seems. People also suggest to use [Karen Simonyan et al.](http://arxiv.org/abs/1312.6034)'s technique for visualization, which is nice, but cannot give me a sneak peek into the second or third network layers (those are the ones I'm interested in, mostly). Still, better then nothing.
 * Apparently my network is still not good enough, because that's what I've managed to obtain:

{:.center}
![][catdog]

*Left: an image that maximizes "dog" class. Right: an image that maximizes "cat" class. Both: I sincerely hoped for deeper dreams.*

*Actually* after staring at them for several minutes it seems that I start to figure out some doggish patterns. Or it's just plain wishful thinking, I guess.

The technique is very simple and produces valid results for simpler images. For example, that's MNIST digits:

{:.center}
![][mnist]

Which is, again, very nice, but isn't what I wanted to see. I'd like to check out what's going on in layer 3 or somewhere around &mdash; what are the features that the network learns after it's done with edges and corners?

Maybe it's better to switch from Theano and Lasagne to something else. Also, to find a more powerful computing unit than my laptop: it runs fast and nice on MNIST digits, but this little catdog network took something like 3 hours to learn (images were resized to 60x60). Also, still trying to make CUDA work, but given the performance boost people report, there isn't going to be a miracle anyway.

[catdog]: /assets/article_images/2015-08-22-deepdreaming-pets/catdog.png
[mnist]: /assets/article_images/2015-08-22-deepdreaming-pets/mnist.png