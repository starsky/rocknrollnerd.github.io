---
layout: post
title:  "Occasional thoughts on spatial coding"
date:   2015-08-10 12:49:00
categories: ml
tags:
comments: true
---

I've been trying to read Norbert Wiener's *Cybernetics* recently, just to look up the origins of all the AI thing. Among other cases he talks about a problem of perceiving a geometric figure (a square) the same way regardless of its size, a.k.a. scale invariance problem. I thought about how is this problem solved nowadays, which is basically just using image pyramids &mdash; either calculating SIFT features or performing max/mean pooling by deep convnet layers. There are other approaches that try to separate object's location and content by using separate units encoding rotation and scale, such as Hinton's transforming autoencoders or quite recent DeepMind's [spatial transformer networks](http://arxiv.org/pdf/1506.02025v1.pdf), and personally I like those even more, but anyway, seems we've got scale invariance covered now.

There are a couple of things that still look a bit suspicious, however. All these methods can be divided in roughly two groups: "global", that scale or transform the entire image at once (Gaussian pyramids and transforming autoencoders), and "local", that perform pooling operations on a some local image region (convnet's max pooling). And you can notice that max pooling performs the same replicated operation across all the pooling regions, so we can call it global too. The point I'm trying to make here is that none of these methods can perform a heterogeneous scaling transformation on an image; for example, for a human figure, to scale up just its head in a caricature style. Transforming autoencoder can decode more complex affine transformation, but that's still quite a limited case of invariance. There are SIFT features still, that operate on all the pyramid levels at once &mdash; they'll match an enlarged human head/face part, but then again, if we encode our image using whole bunch of SIFT features, the spatial correspondence between the head and the body wil be different now, which is a problem.

Or to make a simpler example, something like these digits, when a part of a digit is scaled differently from the other parts:

{:.center}
![][digits]

Still pretty much recognizable, right?

Now, it looks like a heterogeneous scaling transform can *only* be applied to structurally separate object parts. We can think about an enlarged eye but not a half-eye &mdash; it doesn't even make sense, considering there's no way to connect two parts of the transformed eye in a realistic manner. So, we'd like to use some kind of constellation model to split a digit into separate parts, and how do we do *that*? There are multiple possible ways, like:

 * a [classic](https://www.cs.princeton.edu/courses/archive/fall07/cos429/slides/constellation_model.pdf) method: extracting interesting patterns, applying them to images, using EM to find proper combinations. Generally applicable, but I don't like the idea of processing a big dataset just to understand that "8" consists of two circles.
 * using something like time as a supervisor: if a part of an object moves in a separate (from other parts) direction, say, then we can safely assume some kind of structural independence. Sadly, our handwritten digits are pretty much static, although this method still can be used for some cases &mdash; for example, extracting parts from different images of "1" digit (with or without a bottom-horizontal bar) that are missing somethimes.
 * one-short learning approach: extracting salient regions and assuming they belong to constellation parts. Actually I'd like to see how that works for digits...
 * maybe going low-level a bit? We can safely assume that part of our digits' structure can be described by penstroke intersections/corners. Digits with no corners aren't prone to heterogenous scaling: you can't scale a part of "1" or a part of "0" without breaking a contour. Now: these corner features should be a well-known thing mentioned in Neocognitron network and convnets, but then again, convnet's max pooling doesn't look like the scalable solution we're looking for.

I guess before I'll start jumping to conclusions, I should investigate what features a convnet can actually learn on MNIST digits and how does it combine them together. But for now it looks like we can't achieve scale invariance simply by stacking together lots of pooling layers, let alone heterogenous scaling.

What if at hypothetical second layer, where we can represent an image by a significantly smaller subset of corners, we are no longer interested in features' absolute positions and instead look at corners' mutual relations? That would automatically solve *all* our invariance problems, including translation and rotation, but it seems that knowing corners only is simply not enough to represent an object...

[digits]: /assets/article_images/2015-08-10-spatial-coding/digits.png