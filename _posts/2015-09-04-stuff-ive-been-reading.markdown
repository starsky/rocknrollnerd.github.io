---
layout: post
title:  "Stuff I've been reading"
date:   2015-09-04 19:00:00
categories: ml
tags:
comments: true
---

So before the summer school (which I'm going to briefly mention a little bit later) I was going to speedread a randomly chosen sample set of recent papers just to catch up with what's going on. Here are some short notes I made during the reading. Most of the papers are not really recent: it's mostly because I tried to choose the latest publications from researches/groups I haven't been paying close attention to (like Coates/Salakhutdinov/LeCun etc).

1. **Learning to Disentangle Factors of Variation with Manifold Interaction** by S. Reed, K. Sohn, Y. Zhang, H. Lee.

    The paper describes a model (a kind of RBM) designed to capture different factors of the input data: for example, human face pose and expression. Extremely interesting; after all, that's the big question &mdash; looking for independencies in the data and trying to find separate latent representations for them. The practical approach, however, looks like it requires a lot of hand engineering &mdash; the number of factor we're trying to extract is defined in terms of network architecture and cannot be adjusted during the learning. I also haven't got the math right, mainly because it seems to be based on variational learning which I'm still struggling to undestand.

2. **Dropout: A simple way to prevent neural networks from overfitting** by N. Srivastava, G. Hinton, A. Krizhevsky.

    Yeah, despite the fact that dropout is a famous technique now I never read the original paper, and actually shame on me, because it was awesome. Some people I tried to discuss dropout with were really skeptical about the concept, arguing that there's no theory behind it and it's purely a heuristic that works. From my perspective this looks like a great concept certainly worth investigating maybe even on a biological level (I particularly liked the section about evolutionary inspiration for dropout). Does it really make sense for a single neuron to follow the lonely wolf strategy by refusing to cooperate with its neighbor neurons that receive the same input? And if it does, how can it possibly do such a thing?

3. **Deep Boltzmann Machines** by R. Salakhutdinov and G. Hinton.

    Another famous paper; but now I couldn't understand *a thing* from it. That's really sad. Damn you variational learning!

4. **Deep generative stochastic networks trainable by backprop** by Y. Bengio, E. Thibodeau-Laufer, G. Alain.

    Again I understood almost nothing, *but* this paper was mentioned in a talk by Tapani Raiko during the summer school and it seems I've grasped a bit of intuition from it (the relationship between models wuth probabilistic hidden units and denoising models). So I'm definitely going to get back to it right after I'm done with all the stochastic stuff which is suddenly all over the neural networks.

5. **Deep learning of invariant features via simulated fixations in video** by W. Zou, S. Zhu, K. Yu, A. Ng.

    *Ahhh.* The first time I met machine learning was during Andrew Ng's Coursera class and now I'm starting to understand why I'm experiencing trouble trying to read about probability and neural networks: because that course didn't mention probability *at all*. Precisely by the same (I guess) reasons it's super-easy to read his papers: everything is loud and clear, moving on. The biggest mistery of this paper was the usage of the tracking algorithm to track video frames: doesn't it produce redundant sequences of replicated patches? Anyway, I absolutely like the idea of time as weak supervisor, although it seems that the model learns features not really different from the well-known edges and corners.

6. **Learning Feature Representations with K-means** by A. Coates and A. Ng.

    I used to consider this paper a some kind of blast: after all, being able to produce state-of-the-art unsupervised learning results with plain and simple K-means kinda implies that maybe we should throw off all the fancy algorithms like RBMs, autoencoders and sparse coding models. So I kept asking *everyone* from the summer school about it and suprisingly almost never encountered this reaction. There's a post on [reddit](https://www.reddit.com/r/MachineLearning/comments/1rsmlt/whats_wrong_with_kmeans_clustering_compared_to/) which conveys the tone of my question, and the answers are basically sound like "wellll, convolutional networks are better anyway". Interesting. As about the paper itself: I should try to implement the result one day, seems to be a useful tool to play with unsupervised learning methods.

7. **Emergence of Object-Selective Features in Unsupervised Feature Learning** by A. Coates, A. Karpathy, A. Ng

    A continuation of the previous paper, which introduces a deep hierarchy model built on K-means cells. I like the pooling approach they used (pooling "similar-looking" patches together), although I still believe that the idea of grouping edge-like features spatially trying to get higher-level features is wrong...

8. **The Importance of Encoding Versus Training with Sparse Coding and Vector Quantization** by A. Coates and A. Ng.

    That was... strange. I'm not sure I get the idea right: combining different learning/encoding methods produces comparable results? It's possible to encode an image using a linear combination of random code components? Something like this, I guess. The paper is from 2011, so not recent at all; and is it just me or the interest to unsupervised learning methods is lower now than it used to be? Maybe that's becayse we don't need unsupervised pre-training anymore with RELU and stuff.