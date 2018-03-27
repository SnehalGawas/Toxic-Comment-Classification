# Toxic Comment Classification
## Goal
Study negative online behaviors, like toxic comments (comments that are rude, disrespectful or otherwise likely to make someone leave a discussion) and build a model to identify type of toxicity.


## Data Source:
Data is taken from kaggle Toxic Comment Classification Challenge.  

URL: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data


## Data Description:
In the given data large number of Wikipedia comments are given which are labeled by human raters for toxic behavior. The types of toxicity are:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

We will create a model which predicts a probability of each type of toxicity for each comment.


## Feature Extraction

We will be extracting word features from the text using CountVectorizer.
It takes collection of text and convert into matrix of token counts.
1. Here we are using analyzer as word which means our features will be made out of words.
2. We will be removing accents during the preprocessing step using strip_accents paremeter with unicode value. unicode is slightly slow than ascii but it will work with any characters.
3. We will be taking bigrams with  ngram_range= (1,2). These are words that often occur together in the text.
4. All words are converted into lower case by default in CountVectorizer method.

CountVectorizer takes care of basic preprocessing and cleaning so for now We won't be performing any extra cleaning on the data.

## Naïve Bayes
Let’s start with a naïve Bayes classifier. This will be our baseline model. scikit-learn includes several variants of this classifier, the one most suitable for word counts is the multinomial variant.

Naïve Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.

The disadvantage of this assumption is that In real life, it is rare that we get a set of predictors which are completely independent. Features often depend on one another, meaning multiple features often contain similar signals. This results in the bad probability.

In this problem we are required to predict probability and not the class. We will see how Naïve Bayes predicted probabilities perform on the test data.

Results:

- Cross Validation Score = 0.872
- Test Data Score (kaggle Private score) = 0.865

## Support Vector Machines

SVM is widely regarded as one of the best text classification algorithms. It is slower than Naïve Bayes.

Let's try to use SGDClassifier from sklearn which implements regularized linear models with stochastic gradient descent (SGD) learning.
We are using 100 learning iterations here and log loss.

- Cross Validation Score = 0.956
- Test Data Score = 0.954


