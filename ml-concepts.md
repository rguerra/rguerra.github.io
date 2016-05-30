---
file:        |
title:       |  
project:     |  
keywords:    |
author:      rguerra
bibliography: ml.bib
---

## Probability
### Probability Concepts <!--{{{-->
* **Probability Distribution**: which describes the probabilities of something occurring over the range of possible feature values.

* **Central Limit Theorem**: which says that lots of small random numbers will add up to something Gaussian.

* **Conditional Probability**: is a measure of the probability of an event given that (by assumption, presumption, assertion or evidence) another event has occurred [@condprob-wiki].
    * **Two Variables**:
$$P(A,B) = P(A)P(B|A)$$
    * **Three Variables**:
$$P(U) = P(A,B) = P(A)P(B|A)$$
$$P(U,C) = P(U)P(C|U)$$
$$P(A,B,C) = P(A)P(B|A)P(C|A,B)$$
    * **More than three**:
 $$P(A,B,C,D) = P(A)P(B|A)P(C|A,B)P(D|A,B,C)$$
 
* **Chain Rule**: permits the calculation of any member of the joint distribution of a set of random variables using only conditional probabilities. The rule is useful in the study of Bayesian networks, which describe a probability distribution in terms of conditional probabilities [@chainrule-wiki].
$$P(x_1, x_2, x_3,...,x_n)    = P(x_1)P(x_2|x_1)P(x_3|x_1,x_2)...P(x_n|x_1,...,x_{n‐1})$$   

* **Markov Property**: A stochastic process has the Markov property if the conditional probability distribution of future states of the process (conditional on both past and present states) depends only upon the present state, not on the sequence of events that preceded it [@markovproperty-wiki].

* **Markov Assumption**: is used to describe a model where the Markov property is assumed to hold, such as a hidden Markov model [@markovproperty-wiki].

* **Maximum Likelihood Estimation**: probability of observing the given data as a function of the parameters $\theta$ [@mle-classnotes].

<!--}}}-->

## Statistics
### Statistics Concepts <!--{{{-->
* **Regression**: you want to predict one set of numbers given another set of numbers. You’ll also sometimes hear people refer to the set of numbers that are the inputs as predictors or features.

* **Correlation**: is just a measure of how well linear regression could be used to model the relationship between two variables.

* **Mean Square Error**: this is a way to combine all the squared errors from the points in the dataaset.

* **Expectation**: The expected value can be viewed as a weighted average. The expected value of a random variable is the average of all values it can take; thus the expected value is what one expects to happen on average.If the outcomes $x_i$ are not equally probable, then the simple average must be replaced with the weighted 
average, which takes into account the fact that some outcomes are more likely than the others. The intuition however remains the same: the expected value of $X$ is what one expects to happen on average.

* **Variance**: of the set of numbers is a measure of how spread out the values are. It is computed as the sum of the squared distances between each element in the set and the expected value of the set (the mean, µ. The variance looks at the variation in one variable compared to its mean.

* **Covariance**: is a generalization of variance instead of looking at the variation of one variable with respect to its mean it looks at how two variable vary together. The variance looks at the variation in one variable compared to its mean.

* **Covariance Matrix**: The covariance can be used to look at the correlation between all pairs of variables within a set of data. We need to compute the covariance of each pair, and these are then put together into what is imaginatively known as the covariance matrix.
    * *NOTE*: that the matrix is square, that the elements on the leading diagonal of the matrix are equal to the variances, and that it is symmetric since $cov(x_i , x_j ) = cov(x_j , x_i)$.

* **Multinomial Distribution**: it models the probability of counts for rolling a $k$ sided die n times. For $n$ independent trials each of which leads to a success for exactly one of $k$ categories, with each category having a given fixed success probability (these are the parameters of the distribution), the multinomial distribution gives the probability of any particular combination of numbers of successes for the various categories.
    * The simpler clase is the **binomial distribution** where instead of a $k$ sided die it is a two sided coin.

    * **Parameters**: are $n$ (the number of trials that the die was tossed) and $p$, where $p = (p_1, ..., p_k)$ these are probabilities $p_1, ..., p_k$ corresponding to the $k$ possible mutually exclusive outcomes.

* **Multivariate Normal Distribution**: the multivariate normal distribution or multivariate Gaussian distribution, is a generalization of the one-dimensional (univariate) normal distribution to higher dimensions. One possible definition is that a random vector is said to be $k$-variate normally distributed if every linear combination of its $k$ components has a univariate normal distribution. The multivariate normal distribution is often used to describe, at least approximately, any set of (possibly) correlated real-valued random variables each of which clusters around a mean value.
<!--}}}-->

## Machine Learning 
### Machine Learning Concepts: <!--{{{-->

* **False Positive** and **False Negative**: To understand this think of it as a binary classification problem in which the classifier will output either *positive* or *negative*. Now think of the term **False Positive** as a compound term that combines the gold standard answer with your classifier's answer. Just divided the term into two [Gold Standard] [Classifier's Answer]. If the classifier says that an instance is *positive* but it is *wrong* then the error is a [False] [Positive]. If the classifier says that an instance is *negative* but it is *wrong* then the error is a [False] [Negative].

* **Bias-Variance Dilemma**: more complex models do not necessarily result in better results. more complicated models have inherent dangers such as overfitting, and requiring more training data.More complex classifiers will tend to improve the bias, but the cost of this is higher variance, while making the model more specific by reducing the variance will increase the bias.

	* **Bias**: in a model means it is not accurate and doesn’t match the data well,
	
	* **Variance**: in a model means that it is not very precise and there is a lot of variation in the results.
	
* **Decision Boundary or Discriminant Function**: a straight line (in 2D, a plane in 3D, and a hyperplane in higher dimensions) where the neuron fires or the datapoints belong to class A on one side of the line, and doesn’t or belong to class B on the other.

* **Margin**: is defined as the distance between the separating hyperplane (decision boundary) and the training samples that are closest to this hyperplane, which are the so-called support vectors in a **Support Vector Machine**. 

* **Local Markov Assumption**: A variable X is independent of its non-descendants given its parents (and only its parents)

* **Gradient**: is a generalization of the usual concept of derivative of a function in one dimension to a function in several dimensions $f(x_1, ..., x_n)$ [@gradient-wiki].

* **Gradient Descent**:  update a set of parameters in an iterative manner to minimize an error function [@gd-quora].
    * **Gradient Descent**: you have to run through **ALL** the samples in your training set to do a single update for a parameter in a particular iteration [@gd-quora].
    * **Stochastic Gradient Descent**: you use **ONLY ONE** training sample from your training set to do the update for a parameter in a particular iteration [@gd-quora].

* **Soft-Max**: 
    * The more technical definition is: "Probability of choosing $1$ to $N$ discrete items. Mapping from vector space to a multinomial over words" [@web-lda2vec]. 

    * The less technical: "The machine learning softmax is used for classification. Suppose you have $n$ classes.  For any given feature $x$, you want to estimate its probabilities  $p_i$  of being in class $i$. However, your algorithm doesn't directly produce probabilities.  Instead it first produces real-valued scores  $y_1,...,y_n$.From these scores you define the probabilities  $p_i$  using the softmax function." [@quora-softmax]
    * For more info see [here](ml-softmax.html)

###### Important Algorithms
* **Expectation-Maximization [EM] Algorithm**:provides a general approach to the problem of maximum likelihood parameter estimation in statistical models with latent variables.

###### Types of Learning
* **Active Learning**: we are permitted to actively choose future training data based upon the data that we have previously seen. When we are given this extra flexibility, we can often reduce the need for large quantities of data.

###### Probabilistic Graphical Models
* The relationship between naive Bayes, logistic regression, HMMs, linear-chain CRFs, generative models, and general CRFs from McCallum's [introduction to CRFs](http://www.research.ed.ac.uk/portal/files/10482724/crftut_fnt.pdf):
    * Simple Generative Model -> Naive Bayes
    * Simple Discriminative Model -> Logistic Regression
    * Sequence Generative Model -> HMM
    * Sequence Discriminative Model -> Linear-chain CRF
    * General Graphs Generative -> Generative directed Models
    * General Graphs Discriminative -> General CRFs

<!--}}}-->

###### Neural Networks <!--{{{-->
* **Bias Term**:It is useful to think about this term in terms of a line.
	* **Line Equation**: $$y = mx + b$$
	* Without a bias term the line equation is y = mx and thus when x is 0 then y HAS TO BE 0. You cant have the line go through the point x=0, y=3. There is nothing you can multiply to zero to make it non-zero. To fix this a bias term is included. b in y=mx+b allows the line to to cut the y axis in a point different than (0,0).

* **Learning Rate**: parameter ? controlling how much to change/update the models weights by. We could miss it out, which would be the same as setting it to 1. If we do that, then the weights change a lot whenever there is a wrong answer, which tends to make the network unstable , so that it never settles down. The cost of having a small learning rate is that the weights need to see the inputs more often before they change significantly, so that the network takes longer to learn. However, it will be more stable and resistant to noise (errors) and inaccuracies in the data.
<!--}}}-->

###### Deep Learning <!--{{{-->
* **Tensor**: Multidimensional array.
<!--}}}-->

###### NLP <!--{{{-->
* **Language Model**: can be thought of as a function that takes as an input a sequence of words and returns a probability (likelihood) estimate of that sequence [@blogpost-yelpfilters]. The goal is to compute the probability of a sentence or sequence of words [@classstanford-ll].
<!--}}}-->

###### My own definitions (could be wrong) <!--{{{-->
* **kernel**: 
    * You only have some data and need more. So transform your current data to a richer feature space where two data points (x1, y1) and (x2, y2) in the original data transform to three points (x1, y1), (x2, y2), and (x1x2, y1y2).
    * Your problem is not linearly separable and you are using a learning a model that only learns linearly separable problems (like svm). Then transform your data to a feature space where it could be linearly separable. 
<!--}}}-->

### Relations Among Models <!--{{{-->
* Relation between **Perceptron** and **SVMs**.
Another powerful and widely used learning algorithm is the **support vector machine (SVM)**, which can be considered as an extension of the **perceptron**. 
    * Using the **perceptron** algorithm, we minimized misclassification errors. 
    * However, in **SVMs**, our optimization objective is to maximize the margin. 

* **Logistic regression versus SVM**: In practical classification tasks, linear logistic regression and linear SVMs often yield very similar results. 
    * **Logistic regression**: tries to maximize the conditional likelihoods of the training data, which makes it more prone to outliers than SVMs. Logistic regression has the advantage that it is a simpler model that can be implemented more easily.Furthermore, logistic regression models can be easily updated, which is attractive when working with streaming data.
    * The **SVMs**: mostly care about the points that are closest to the decision boundary (support vectors).  




<!--}}}-->

## REFERENCE
