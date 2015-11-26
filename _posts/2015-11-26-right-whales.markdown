---
layout: post
title:  "Wrong whales"
date:   2015-11-26 20:11:00
categories: ml
tags:
comments: true
---

OK, it's been quite some time since I wrote anything here: partly because of all the stuff going on with my life (thesis defending process and so on), but most importantly, because of a machine learning project that have been taking lots of my time. I mean [Kaggle's right whale recognition challenge](https://www.kaggle.com/c/noaa-right-whale-recognition).

{:.center}
![][whales_cover]

The challenge consists of two stages: first construct a "whale face detection" algorithm that can extract whale heads from an aerial view image like the one displayed on top, and than make a classification model that can discriminate between different whales based on one's head [callosity pattern](http://www.neaq.org/conservation_and_research/projects/endangered_species_habitats/right_whale_research/right_whale_projects/monitoring_individuals_and_family_trees/identifying_with_photographs/how_it_works/callosity_patterns.php) which is considered to be the main distinguishing feature.

I'm still struggling with the first problem, which, although presented by organizers as a quite simple step (they even included some [guidelines](https://www.kaggle.com/c/noaa-right-whale-recognition/details/creating-a-face-detector-for-whales) on that), appears to be quite non-trivial. So I'll try to explain the algorithm I'm working on and the path that lead me to it here in some detail without actually sharing the code.

# Why is it non-trivial

{:.center}
![][easy_whales]

So when you look at some of the images it actually appears to be quite easy to detect heads without any machine learning at all: there are large distinct white-pattern tiles that can be selected with some simple hand-crafted filter. The background is always water, different colours are possible, but still, no background clutter or extra objects that would require scene segmentation. There's always one point of view (from above), and one whale per picture. It looks like so simple compared to human face recognition, where human faces can appear on lots of different backgrounds and take lots of different 3d poses.

But then there are different pictures like these:

{:.center}
![][hard_whales]

Well, now... it all makes things a bit complicated. There are extra foam patterns which are bright and white and can mess with face detection, and the sun is clearly not helping with all the bright spots and reflections. Sometimes the head pattern is heavily obscured by water and foam (see the top-right picture, for example) and sometimes it seems to be really hard to distinguish it from some other body part. For example, which picture corresponds to a head here?

{:.center}
![][tail_head]

The left one is a lower body patch, the right one is the actual head; but the distinction is quite subtle, when you look at both patches without any context.

{:.center}
![][tail]

# The first attempt: naive sliding window detection

I've decided to ignore cascade detector approach suggested by organizers and go straight to convolutional network. I've annotated the subset of 500 whales (thought it'll take a lot of time, but actually I was done in a couple of nights), wrote a script that extracts head patches and samples a bunch of random background patches from a single image, and made a really simple CNN with Theano and Lasagne (basically just tried to build something like VGGNet: lots of layers with small 3x3 kernels). I've trained the network on my laptop: after *finally* installing the correct Nvidia driver (and by "correct" I mean "the one that doesn't make my Ubuntu boot in a black screen") I could fully use the almighty power of my GT 650M.

The results, hovewer, were quite bad: the network produced lots and lots of false positives, labeling positively almost every patch that was not a monochromatic background water. I managed to get something useful out of it, finally, just selecting the maximum certain patch (like the one that the network detected with 99% confidence; there were a lot of less-certain positives with the score of like 65%), and, well, some results were obtained:

{:.center}
![][right_whales]

Of course it was an extremely naive approach: I didn't even use different scales of an image, used a poorly augmented dataset (no arbitrary rotations, just horizontal and vertical flips), and yet it worked - at least partly. And, of course, the expected problems were immediately faced:

{:.center}
![][wrong_whales]

Damn you, foam patches. Look at the right-bottom picture: the head is barely seen, almost covered by water, and the brightest-white patches correspond to foam. Then it's just a matter of luck if the pattern looks similar to whale head, and bam, we're lost.

# Context matters

So it seems that the most important thing we understood so far is that local patch-based detectors handle the problem poorly. This is, I think, what makes the problem so interesting: for example, consider the difference between ImageNet localization challenge and this one. An object in ImageNet images usually occupies a lot of space, is quite visible, there are hardly any false positive similar-looking background patches and not much extra space to search. The main difficulty is that the objects can take a lot of different shapes, points of view, and can belong to different classes. This problem is quite different: sometimes the object we have to find is so occluded or looks like background, that even human eye can be confused, *when it looks directly at it*.

Well, but the said human eye is actually *able* to find the head in the last right-bottom ambiguous picture, right? We can do that simply by looking at the spot where whale body ends. Therefore, an interesting idea emerges: **to recognize the thing, you have to look somewhere near it, not directly at it**. In other words, context matters. To be able to locate whale head, we need to know the location of whale body first. And maybe *that* should be at least a little bit easier.

# Looking for body

Okay, the problem slightly changes, but is still object localization. [Forums](https://www.kaggle.com/c/noaa-right-whale-recognition/forums) seem to be full of pure computer-vision solutions for whale body detection - people use color filtering, histogram similarity; I've also tried local entropy and random walking segmentation (just for fun, basically).

{:.center}
![][entropy]

*Measuring local entropy with scikit-image. Sometimes it's quite good, other times waves and foam create enough entropy to render it meaningless.*

After all, I returned to convolutional networks, simultaneously looking up the more sophisticated approaches like [OverFeat](http://arxiv.org/abs/1312.6229) and [R-CNN](http://arxiv.org/abs/1311.2524). My search led me to a slide [comparison](https://www.google.ru/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwiMnZmM7a3JAhWIGCwKHVilC_0QFggfMAA&url=http%3A%2F%2Fcourses.cs.tau.ac.il%2FCaffe_workshop%2FBootcamp%2FLecture%25206%2520CNN%2520-%2520detection.pptx&usg=AFQjCNFlswHpK0kC6bVL1KRN1oDnl4kfyg&sig2=LqH0hdb0_ELK2uc4BiHDeg&bvm=bv.108194040,d.bGg&cad=rja) of these methods, which brought some clarity in what's going on with CNN-based object localization. Unfortunately, both R-CNN and OverFeat seemed to be not quite good enough for what I had in mind: detecting whale bodies with *oriented* (tight) bounding boxes. Like this (left) and *not* like this (right):

{:.center}
![][bboxes]

But then I stumbled upon that [Szegedy paper](http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf) that suggested to use convolutional network as a model that predicts black and white mask which corresponds to object's location. Which is really cool, because we're now allowed to use arbitrary "bounding-shapes" if we like, but also cool because it's so simple. We don't need sliding windows, different scales or extra feature extraction - just put the whole image into the network, and output a black and white version of it, maybe downscaled. So I decided to give it a try and began to annotate whale bodies.

(I use [Sloth](http://sloth.readthedocs.org/) for annotations, which doesn't allow to use oriented bounding boxes as annotation items; on the other hand, the docs suggest the possibility of creating a custom item from Qt graphic toolset. I tried that and failed, so my annotations ended up to be approximately-rectangular polygons, which I later fitted to rectangles using some geometry)

So, the pipeline looked somewhat like this:

 * annotate the dataset with polygons in Sloth
 * run the annotations throught a script that fits  oriented rectangles to polygons (nothing fancy here: estimating the center, locating two groups of far away vertices, calculating the slope and then width and height. Oriented bounding box is described by 5 parameters: center coordinates (ox, oy), width, height and angle)
 * paint black and white mask and store it alongside the original image

It's actually possible to skip the second step: after all, since we're allowed to use arbitrary shapes for object location, there's no need to make them strict rectangles. There were two reasons for that: first, after we locate the body, we need to crop it out of the image, so to fit a rectangle to a result mask, therefore it's better if the network outputs a somewhat close to retangular shape. The second reason is that I also tried a little bit different approach, described by other Szegedy's paper, when the network predicts not the mask, but directly 5 output parameters (ox, oy, width, height and angle). This didn't work out well, unfortunately.

So after my train set was ready I spend a week or so tinkering with the network parameters, and ended up with quite nice predictions:

{:.center}
![][mask]

*The last panel displays the resulting bounding box, fitted to a mask. Again nothing fancy, just some thresholding and PCA*

Some nuances and tips I've encountered during the training:

 * augmenting your dataset is the key. I started with 10 random rotations per image and quite poor results; extending that to 50 rotations per images improved things considerably. Unfortunately, my laptop couldn't handle it anymore, so I moved to Amazon EC2 GPU instance.
 * the paper suggests to use slightly different objective function to penalize "predicting black pixel where there's actually a white pixel" more then the other way around. This is based on the fact that masks are mostly black and just small parts of it are white, so the network can fall into the trivial solution of always predicting black images (maybe with centered bright spot). I couldn't quite understand Szegedy's function and came up with my own, which looks like this: `(prediction - target) ** 2 * (target * _lamda + 1)`, where `_lambda` controls the penalty value (and `target` is binary target mask). But at some point I've decided not to use it at all: standard mean square error was just fine. Maybe because whales are quite large and occupy enough white space so that the network can't just ignore them.
 * the VGGNet-like architecture (lots of layers, tiny kernels) showed worse results than the opposite approach: 4 convolutional layers and quite large kernels (9x9 and so on).
 * adaptive histogram equalization (CLAHE) applied to images sufficiently improves training score. Other than that and downscaling, I didn't try any preprocessing steps.
 * after the convolutional layers I placed a couple of fully-connected layers approximately twice as big as the mask size. This turned out to be good enough.

The "body pipeline" is basically predict a mask, fit an oriented bounding box (I also constrained it to 1:3 aspect ratio), rotate the image and crop the whale, and end up with something like this:

{:.center}
![][extracted_whales]

*There are two whales on the last picture: extremely rare case*

Not bad, huh.

# Next step: locating the head

Though I like the results so far (about 98% correctly cropped whales), we still have to locate the head. And we cannot do that (as I secretly hoped) by simply chopping off the leftmost and rightmost parts of the cropped image ("look at the end of the body"), since the cropping is still quite inaccurate and includes water background.

But we've still won something: there's much less space to search, we've thrown away lots of confusing foam patches and bright reflection spots, and most importantly, the whales are now rotated horizontaly, which greatly reduces the spatial variance. So why don't we use the same mask prediction technique again?

This is the point I'm stuck right now, because head detection still works worse then body detection (about 75% success rate). Average successfull prediction looks like this:

{:.center}
![][head_success]

But quite often the network is still confused by the bright foam regions:

{:.center}
![][head_failure]

And sometimes it, quite interestingly, makes two predictions (when it cannot decide which part looks more like head):

{:.center}
![][head_double]

I'm trying to solve the latter situation with [DBSCAN clustering](https://en.wikipedia.org/wiki/DBSCAN), which allows me to decide between multiple region predictions, but the decision rule is quite arbitrary (I always select the bigger region with some randomness, which is, strictly speaking, doesn't have to correspond to a head).

So, at the moment, I've got a couple of ideas on how to improve it:

 * try to make better body predictions. Can we, for example, make the box tighter, so that the head precisely touches the boundary? Maybe it's time to use some kind of histogram comparison technique: considering that the outer parts of "body" image are most likely to be water, we take the average histogram statistic for background and trying to eliminate that.
 * somehow rotate all the whales in the dataset so that they all share the same orientation. As you can notice, the body detection network cannot discriminate between right and left, but maybe we can make a separate (simpler) model that just tries to predict orientation as a binary number? That would reduce the variance even more.
 * just throw more images in the dataset! Although I'm already dead tired of manual annotations: last time it was about 1000 images. But after I see some wonderful person annotating the [whole training set](https://www.kaggle.com/c/noaa-right-whale-recognition/forums/t/17421/complete-train-set-head-annotations) the option starts to look feasible...

Okay, that's all for now. The interesting parts are still ahead (I haven't yet tried the classification itself and haven't made a single leaderboard submission), and I'm fascinated so far by switching from toy projects and papers to an actually big machine learning project. I hope I'll post an update on that soon.


[whales_cover]: https://kaggle2.blob.core.windows.net/competitions/kaggle/4521/media/ChristinKhan_RightWhaleMomCalf_640.png
[easy_whales]: /assets/article_images/2015-11-26-right-whales/easy_whales.jpg
[hard_whales]: /assets/article_images/2015-11-26-right-whales/hard_whales.jpg
[tail_head]: /assets/article_images/2015-11-26-right-whales/tail_head.png
[tail]: /assets/article_images/2015-11-26-right-whales/tail.jpg
[right_whales]: /assets/article_images/2015-11-26-right-whales/right_whales.jpg
[wrong_whales]: /assets/article_images/2015-11-26-right-whales/wrong_whales.jpg
[entropy]: /assets/article_images/2015-11-26-right-whales/entropy.jpg
[bboxes]: /assets/article_images/2015-11-26-right-whales/bboxes.jpg
[mask]: /assets/article_images/2015-11-26-right-whales/mask.png
[extracted_whales]: /assets/article_images/2015-11-26-right-whales/extracted_whales.jpg
[head_success]: /assets/article_images/2015-11-26-right-whales/head_success.jpg
[head_failure]: /assets/article_images/2015-11-26-right-whales/head_failure.jpg
[head_double]: /assets/article_images/2015-11-26-right-whales/head_double.jpg