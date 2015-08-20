---
layout: post
title:  '"I want my corners" (Avon Barksdale)'
date:   2015-08-20 00:50:00
categories: ml
tags:
comments: true
---

So, how do we learn feature hierarchies from images?

1. Cut a lot of patches from image data, put them into an unsupervised algorithm. Get a bunch of features that look like edges (which is well-known and reproduced many times).
2. Then we can make bigger patches, represent them in terms of the learned edges and put into the second layer using the same algorithm. Surprisingly, I've seen just [one paper](http://ai.stanford.edu/~ang/papers/nips07-sparsedeepbeliefnetworkv2.pdf) that shows that you can get corner-like features as a result (and also different contour parts). The key part here is that we look for first-level features that occuur together (close to each other). That's what spatial pooling does.
3. *Then* as different deep learning architectures suggest, we perform the same pooling/feature learning steps over and over. But wait a minute, if we do believe that second-level features are indeed corners, it doesn't make sense. We can find corners by looking at (different) edges that are active in some close proximity, but we can't use the same logic for features of higher order, because combining different corners together produces junstions and gratings, which are structurally the same. What we'd like to get at the abstract level 3 is maybe some simple geometric figures or [geons](https://en.wikipedia.org/wiki/Geon_%28psychology%29) and we **can't** do that by using spatial pooling simply because different corners that compose the figure can be far away from each other. Actually, isn't it the place where scale invariance kicks in? Maybe we can get away from absolute-valued spatial metric here by representing an image as some elastic corner graph?

An idea: look for coincident corners (pairs of corners that correspond to each other, i.e. can be connected together). That might include corners that doesn't have an actual edge between them - [illusory contours](https://en.wikipedia.org/wiki/Illusory_contours)? To do that, we have to know corner orientation (or somehow to learn all these corresponding pair which may be expensive). I hoped I can do that by using [factored sparse coding](http://arxiv.org/pdf/1109.6638v2.pdf) which ables angle parametrization for edge features, but I failed to understand and implement the algorithm, sadly.

Still, what features do convolutional networks learn, when they use spatial pooling approach? As [this paper](http://www.matthewzeiler.com/pubs/arxive2013/eccv2014.pdf) shows, it's mostly pieces of aligned image parts, if they are available (it also learns cornerse and junctions and gratings, no surprise). That reminds me of a simple sparse coding [hierarchy](https://github.com/rocknrollnerd/deep_hierarchy) I tried to implement some time ago which *didn't* learn higher-level features running against Kaggle's cats and dogs dataset. Does it mean this set doesn't contain any aligned parts (meaning each dog/cat is unique; kinda unlikely, I guess), or maybe my network/sample set wasn't big enough, or maybe sparse coding is just worse somehow? Now I want to build a convnet and run it against this dataset, using one of feature visualization techniques. An interesting question: what is we really construct a dataset of a lot of unique dogs with no repeated aligned image parts to learn the corresponding filters? Will a convnet fail there?

(if anybody's reading it, I apologize for that stream of consciousness)

As a side note: started reading "Machine Learning: probabilistic perspective". Great book, hope it'll bring some clarity into crazy probability-heavy papers from Toronto that I can't understand at all for now.