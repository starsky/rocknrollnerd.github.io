---
layout: post
title:  "Suddenly, a leopard print sofa appears"
date:   2015-05-27 14:34:25
categories: ml
tags:
comments: true
---

If you have been around all the machine learning and artificial intelligence stuff, you surely have already seen this:

![][imagenet2012]

*Or, if you haven't, there are some deep convolutional network result samples from ILSVRC2010, by Hinton and Krizhevsky*

Let's look for a moment at the top-right picture. There's a leopard, recognized with substantial confidence, and then two much less probable choices are jaguar and cheetah.

And this is, if you think about it for a bit, kinda cool. Do *you* know how to tell apart those three big and spotty kitties? Because I totally don't. There must be differences, of course &mdash; maybe something subtle and specific, that only a skilled zoologist can perceive, like general body shape or jaw size, or tail length &mdash; or maybe is it context/background, because leopards inhabit forests and are more likely to be found laying on a tree, when cheetahs live in savanna? Either way, for a machine learning algorithm, this looks very impressive to me. After all, we're still facing lots of really annoying and foolish errors like [this one](http://weknowmemes.com/wp-content/uploads/2013/03/facial-recognition-fail.jpg). So, is that the famous deep learning approach? Are we going to meet human-like machine intelligence soon?

Well... turns out, maybe not so fast.

# Just a little zoological fact

Let's take a closer look at these three kinds of big cats again. Here's the jaguar, for example:

![][jaguar]

It's the biggest cat on both Americas, which also has a curious habit of killing its prey by puncturing their skull and brain (that's not really the little fact we're looking for). It's the most massive cat in comparison with leopard and cheetah, and its other distinguishing features are dark eyes and larger jaw. Well, that actually looks pretty fine-grained.

![][leopard]

Then, the leopard. It's a bit smaller then jaguar and generally more elegant, considering, for example, its smaller paws and jaw. And also yellow eyes. Cute.

![][cheetah]

And the smallest of the pack, the cheetah, that actually looks quite different from the previous two. Has a generally smaller, long and slim body, and a distinctive face pattern that looks like two black tear trails running from the corners of its eyes.

And now for the part I've purposely left out: black spotty print pattern. It's not completely random, as you might think it is &mdash; rather, black spots are combined into small groups called "rosettes". You can see that jaguar rosettes are large, distinctive and contain a small black spot inside, while leopard rosettes are significantly smaller. As for the cheetah, its print doesn't contain any, just a scatter of pure black spots.

![][spots]

*See how those three prints actually differ (also, thanks [Imgur](http://imgur.com/gallery/md8HT) for educating me and providing the pictures*).

# Suspicion grows

Now, I have a little bit of bad feeling about it. What if this is *the only thing* our algorithm does &mdash; just treating these three pictures like shapeless pieces of texture, knowing nothing about leopard's jaw or paws, its body structure at all? Let's test this hypothesis by running a pre-trained convolutional network on a very simple test image. We're not trying to apply any visual noise, artificial occlusion or any other tricks to mess with image recognition &mdash; that's just a simple image, which I'm sure everyone who reads this page will recognize instantly.

Here it is:

![][sofa]

We're going to use [Caffe](http://caffe.berkeleyvision.org/) and its pre-trained CaffeNet model, which is actually different from Hinton and Krizhevsky's AlexNet, but the principle is the same, so it will do just fine. Aaand here we go:

{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt

caffe_root = '../'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

MODEL_FILE = '../models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = '../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
IMAGE_FILE = '../sofa.jpg'

caffe.set_mode_cpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2, 1, 0),
                       raw_scale=255,
                       image_dims=(500, 500))
input_image = caffe.io.load_image(IMAGE_FILE)
prediction = net.predict([input_image])
plt.plot(prediction[0])
print 'predicted class:', prediction[0].argmax()
plt.show()
{% endhighlight %}

Here's the result:

![][plot]

{% highlight python %}
>> predicted class: 290
{% endhighlight %}

![][classes]
*Whoops.*

But wait, maybe that's just CaffeNet thing? Let's check something third-party:

**[Clarifai](http://www.clarifai.com/)** (those guys did great on the latest ImageNet challenge)

![][clarifai]

**[Brand new Stephen Wolfram's ImageIdentify](https://www.imageidentify.com/)**

![][wolfram]

Okay, I cheated a bit: on the last picture the sofa is rotated by 90 degrees, but that's really simple transformation that should not change the recognition output so radically. I've also tried [Microsoft](https://www.projectoxford.ai/demo/visions#Analysis) and [Google](https://images.google.com/) services and nothing have beaten rotated leopard print sofa. Interesting result, considering all the *"{Somebody}'s Deep Learning Project Outperforms Humans In Image Recognition"* headlines that's been around for a while now.

# Why is this happening?

Now, here's a guess. Imagine a simple supervised classifier, without going into model specifics, that accepts a bunch of labeled images and tries to extract some inner structure (a set of features) from that dataset to use for recognition. During the learning process, a classifier adjust its parameters using prediction/recognition error, and here's when dataset size and structure matter. For example, if a dataset contains 99 leopards and only one sofa, the simplest rule that tells a classifier to always output "leopard" will result in 1% recognition error while staying not intelligent at all.

And that seems to be exactly the case, both for our own visual experience and for ImageNet dataset. Leopard sofas are rare things. There simply aren't enough of them to make difference for a classifier; and black spot texture makes a very distinctive pattern that is otherwise specific to a leopard category. Moreover, being faced with *different* classes of big spotted cats, a classifier can benefit from using these texture patterns, since they provide simple distinguishing features (compared with the others like the size of the jaw). So, our algorithm works just like it's supposed to. Different spots make different features, there's little confusion with other categories and sofa example is just an anomaly. Adding enough sofas to the dataset will surely help (and then the size of the jaw will matter more, I guess), so there's no problem at all, it's just how learning works.

Or is it?

# What we humans do

Remember your first school year, when you learned digits in your math class.

When each student was given a heavy book of MNIST database, hundreds of pages filled with endless hand-written digit series, 60000 total, written in different styles, bold or italic, distinctly or sketchy. The best students were also given an appendix, "Permutation MNIST", that contained the same digits, but transformed in lots of different ways: rotated, scaled up and down, mirrored and skewed. And you had to scan through *all* of them to pass a math test, where you had to recognize just a small subset of length 10000. And just when you thought the nightmare was over, a language class began, featuring not ten recognition categories, but twenty-five instead.

So, are you going to say that was not the case?

It's an interesting thing: looks like we don't really need a huge dataset to learn something new. We perceive digits as abstract concepts, Plato's ideal forms, or actually rather a spatial combinations of ones, like "a straight line", "a circle", "an angle". If an image contains two small circles placed one above the other, we recognize an eight; but when none of the digit-specific elements are present, we consider the image to be not a digit at all. This is something a supervised classifier never does &mdash; instead, it tries to put the image into the closest category, even if likeness is negligible.

Maybe MNIST digits is not a good example &mdash; after all, we all have seen a lot of them in school, maybe enough for a huge dataset. Let's get back to our leopard print sofa. Have you seen a lot of leopards in your life? Maybe, but I'm almost sure that you've seen "faces" or "computers" or "hands" a lot more often. Have you actually seen such a sofa before &mdash; even once? Can't be one hundred percent confident for myself, but I think I have not. And nevertheless, despite this total lack of visual experience, I don't consider the image above a spotty cat in a slightest bit.

# Convolutional networks make it worse

![][convnet]

Deep convolutional network are long-time ImageNet champions. No wonder; they are designed to process images, after all. If you are not familiar with the concept of CNNs, here's a quick reminder: they are locally-connected networks that use a set of small filters as local feature detectors, convolving them across the entire image, which makes these features translation-invariant (which is often a desired property). This is also a lot cheaper than trying to put an entire image (represented by 1024x768=~800000 naive pixel features) into a fully-connected network. There are other operations involved in CNNs feed-forward propagation step, such as subsampling or pooling, but let's focus on convolution step for now.

Leopards (or jaguars) are complex 3-dimensional shapes with quite a lot of degrees of freedom (considering all the body parts that can move independently). These shapes can produce a lot of different 2d contours projected on the camera sensor: sometimes you can see a distinct silhouette featuring a face and full set of paws, and sometimes it's just a back and a curled tail. Such complex objects can be handled by a CNN very efficiently by using a simple rule: "take all these little spotty-pattern features and collect as many matches as possible from the entire image". CNNs local filters ignore the problem of having different 2d shapes by not trying to analyze leopard's spatial structure at all &mdash; they just look for black spots, and, thanks to nature, there are a lot of them at any leopard picture. The good thing here is that we don't have to care about object's pose and orientation, and the bad thing is that, well, we are now vulnerable to some specific kinds of sofas.

And this is really not good. CNN's usage of local features allows to achieve transformation invariance &mdash; but this comes with the price of not knowing neither object structure nor its orientation. CNN cannot distinguish between a cat sitting on the floor and a cat sitting on the ceiling upside down, which might be good for Google image search but for any other application involving interactions with actual cats it's kinda not.

If that doesn't look convincing, take a look at [Hinton's paper](http://www.cs.toronto.edu/~fritz/absps/transauto6.pdf) from 2011 where he says that convolutional networks are doomed precisely because of the same reason. The rest of the paper is about an alternative approach, his [capsule theory](http://www.kdnuggets.com/2014/12/geoffrey-hinton-talks-deep-learning-google-everything.html) which is definitely worth reading too.

# We're doing it wrong

Maybe not all wrong, and of course, convolutional networks are extremely useful things, but think about it: sometimes it almost looks like we're already there. We're using huge datasets like ImageNet, organize competitions and challenges, where we, for example, have decreased MNIST recognition error rate from 0.87 to 0.23 ([in three years](http://en.wikipedia.org/wiki/MNIST_database)) &mdash; considering that no one really knows what error rate a human brain can achieve. There's a lot of talk about GPU implementations &mdash; like it's just a matter of computational power now, and the theory is all fine. It's not. And the problem won't be solved by collecting even larger datasets and using more GPUs, because leopard print sofas are inevitable. There always going to be an anomaly; lots of them, actually, considering all the things painted in different patterns. Something has to change. Good recognition algorithms have to understand the structure of the image and to be able to find its elements like paws or face or tail, despite the issues of projection and occlusion.

So I guess, there's still a lot of work to be done.

[imagenet2012]: /assets/article_images/2015-05-27-leopard-sofa/imagenet2012.png
[jaguar]: /assets/article_images/2015-05-27-leopard-sofa/jaguar.jpg
[leopard]: /assets/article_images/2015-05-27-leopard-sofa/leopard.jpg
[cheetah]: /assets/article_images/2015-05-27-leopard-sofa/cheetah.jpg
[spots]: /assets/article_images/2015-05-27-leopard-sofa/spots.jpg
[sofa]: /assets/article_images/2015-05-27-leopard-sofa/sofa.jpg
[plot]: /assets/article_images/2015-05-27-leopard-sofa/plot.png
[classes]: /assets/article_images/2015-05-27-leopard-sofa/classes.png
[clarifai]: /assets/article_images/2015-05-27-leopard-sofa/clarifai.png
[wolfram]: /assets/article_images/2015-05-27-leopard-sofa/wolfram.png
[convnet]: /assets/article_images/2015-05-27-leopard-sofa/convnet.gif