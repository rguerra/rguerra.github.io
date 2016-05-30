---
file:        ml-softmax.md
title:       |
project:     |
keywords:    |
author:      rguerra
date:        2016-05-30 16:06
output: html_document
bibliography: ml.bib
---

# Soft Max
Many thanks to Charles Yang for his quora post [@quora-softmax] that I am basically pasting here because it is so well explained.

## Original definition according to [@quora-softmax]

### What is Max?
**Max** is a function from $\mathbb{R}^n \Rightarrow \mathbb{R}$.
$$max(x_1,...,x_n) = y$$

However, **max** is not differentiable.  Therefore in many applications one uses the function:

$$\text{softmax}(x_1,...,x_n)=log\big(\sum\limits_{i=1}^ne^{x_i}\big)$$

To understand what is happening, lets picture in the $x,y$ plane the graph of:
$$max(x, 5) = y$$
The graph an intersection of two lines (a horizontal one and a 45 degree one) at an elbow. For all values of $x \le 5$, $max(x,5)$ is equal to 5. In other words it is a horizontal line. The elbow of the graph is when $x=5$ because for all values of $x \gt 5$  $max(x,5)$  is equal to $x$. So it is a 45 degree straight line $x = y$. 

A graph with an elbow is not differentiable, so a differentiable approximation is **softmax**.

As [@quora-softmax] mentions, you can also rescale the softmax to better approximate the function.  For  $k \gt 0$, define

$$k−\text{softmax}(x_1,...,x_n)=\frac{1}{k}log\big(\sum\limits_{i=1}^ne^{kx_i}\big)$$

Then as  $k$  increases, you get a better approximation of max from softmax: however, this comes at a cost of increasing the roughness (size of derivatives) of the function.

## Machine Learning Definition
I am just copying the definition and explanation from [@quora-softmax] but I really like the way it is explained.

According to [@quora-softmax], in machine learning **softmax** is used for classification. Suppose you have $n$ classes.  For any given feature $x$, you want to estimate its probabilities  $p_i$  of being in class $i$. However, your algorithm doesn't directly produce probabilities.  Instead it first produces real-valued scores  $y_1,...,y_n$.From these scores you define the probabilities  $p_i$  using the softmax function.

$$(p_1,...,p_n) = \text{softmax}(y_1,...,y_n) = \Big(\frac{e^{y_1}}{\sum\limits_{j=1}^ne^{y_j}},...,\frac{e^{y_n}}{\sum\limits_{j=1}^ne^{y_j}}\Big) $$

Furthermore according to [@quora-softmax], why set up your classifier this way?  Because a probability vector  $(p_1,...,p_n)$ lives in a very constrained space (nonnegative, sums to 1), and it's hard to work with functions that map from the feature space to this constrained space.  In particular, the sum-to-one constraint means that you can't train learners for each class separately.  Instead, you work with functions that map to the unconstrained space of scores  $(y_1,...,y_n)$, and then map those scores to the space of probability vectors in the last step.  This allows you to divide up the problem into  $n$  subproblems of predicting  $y_1,...,y_n$, and it's also a generalization of logistic regression.

## Relation of both definitions
As you can note the formula for **softmax** in both definitions is different.

* Original Def. (**Softmax**):
$$\text{softmax}(x_1,...,x_n)=log\big(\sum\limits_{i=1}^ne^{x_i}\big)$$

* Machine Learning Def. (**Softmax Gradient**):
$$\text{softmax}(x_1,...,x_n) = \Big(\frac{e^{x_1}}{\sum\limits_{j=1}^ne^{x_j}},...,\frac{e^{x_n}}{\sum\limits_{j=1}^ne^{x_j}}\Big) $$

It turns out that one is the gradient of the other.

# REFERENCE
