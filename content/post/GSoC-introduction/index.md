---
title: Google Summer of Code 2021

# Summary for listings and search engines
summary: Introducing my project for the GSoC 2021.

# Link this post with a project
projects: []

# Date published
date: "2021-06-08"

# Date updated
lastmod: "2021-06-08"

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: true

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  caption: 'Image credit: [**MediaWiki**](https://www.mediawiki.org/wiki/Google_Summer_of_Code/2020)'
  focal_point: "Smart"
  placement: 3
  preview_only: false

authors:
- admin

tags:
- GSoC
- QuTiP

categories:
- GSoC
---


At the time of writing this post, I have just finished my Master's thesis in [Taminiau
Lab](http://taminiaulab.qutech.nl/) under the supervision of C. E. Bradley and T. H.
Taminiau. It has been a tough year due to the global pandemic and I would like to thank
Taminau's group for all the support during the project. I have learned a lot from
you!!

Now, I am ready for my next challenge: [Google Summer of
Code](https://summerofcode.withgoogle.com/) (GSoC). During the summer
of 2021 I will be participating in the GSoC working on one of the projects sponsored by
[numFOCUS](https://numfocus.org/). In particular, I will be contributing to
[QuTiP](https://qutip.org/), a python package for
simulating open quantum systems. My goal will be to implement a
[TensorFlow](https://www.tensorflow.org/) data backend
for QuTiP. Adapting QuTiP to use TensorFlow's tensors will allow QuTiP to benefit from
features such as auto differentiation and operating in a GPU.

The goal is to extend QuTiP's capabilities without compromising its simplicity. For
that, I believe it is necessary to include benchmarks that guide the user to choose when
it is worth using [qutip-tensorflow](https://github.com/qutip/qutip-tensorflow). During the first week, I will design a set of
benchmarks that can be used in the near future to compare the different data layers in
QuTiP and qutip-tensorflow. I will focus first on comparing the already existing
two data layers in QuTiP, dense and CSR. Then, I will use these benchmarks to test
and compare the TensorFlow data layer that I will implement.

If you found this post interesting stay tuned for the next posts and take a look at [my
proposal](QuTiP_GSoC_2021_Asier_Galicia.pdf) for the GSoC.
