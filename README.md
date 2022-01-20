# Gaussian Naive Bayes Classifer
Python implementation of Gaussian Naive Bayes Classifier.

## Model Description
Naive Bayes classifier model is a probabilistic machine learning model that uses [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) in as its core principle. The working of the model is with the assumption that every feature/input is totally independent of each other.

## Functioning and Mathematics:

### Bayes' Theorem
Bayes' theorem states that probability of occurrence of event A given event B has occurred is equal to the product of probability of occurrence of event B given event A has occurred and probability of event A whole divided by probability of occurrence of event B.

<img src="https://latex.codecogs.com/svg.latex?\Large&space;P(A|B)=\dfrac{P(B|A).P(A)}{P(B)}"/>

In this case, the equation would translate to

<img src="https://latex.codecogs.com/svg.latex?\Large&space;P(Y|X)=\dfrac{P(X|Y).P(Y)}{P(X)}"/>

Where,
* Y is the resultant vector.
* X is the feature matrix.

Since we assume that all the features are independent of each other, the probability of all of them occurring is simply the product of their individual probability. Hence, the above equation becomes

<img src="https://latex.codecogs.com/svg.latex?\Large&space;P(Y|X)=\dfrac{P(X_{0}|Y).P(X_{1}|Y).P(X_{2}|Y)...P(X_{n}|Y).P(Y)}{P(X)}"/>
For all n features

If observed carefully, you'll notice that higher the value for the probability of occurrence of X given Y, higher the chances of the corresponding Y value being the being the result.

### Gaussian Probability Density Function
* [Gaussian Probability Density Function](https://www.sciencedirect.com/topics/mathematics/gaussian-probability-density-function) (PDF) is function (that plots bell curve with mean as its center) used to obtain the probability of occurrence of a specific value. The area under the curve in a specific range would give you the probability of obtaining a value within the aforementioned range.
* PDF function can be defined as

    <img src="https://latex.codecogs.com/svg.latex?\Large&space;f(x)=\dfrac{e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}}{\sigma\sqrt{2\pi}}"/>

* Where,
    * x is the feature vector.
    * f(x) is the PDF function.
    * σ is the standard deviation of x.
    * μ is the mean of x.
    * e is Euler's number ≈ 2.718.
    * π is pi ≈ 3.142.

* The derivation of PDF is a tad bit lengthy, so I linked it (video) [here](https://www.youtube.com/watch?v=cTyPuZ9-JZ0).

## Example
```py
from model import NaiveBayesClassifier

model = NaiveBayesClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(f"Accuracy Score: {accuracy_score(predictions, y_test)}")
```