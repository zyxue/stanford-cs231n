import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    # print(scores)

    # for numerical stability purpose: http://cs231n.github.io/linear-classify/
    log_c = np.exp(- scores.max())

    exp_scores = log_c * np.exp(scores)

    scores_sum = np.sum(exp_scores)

    probs = exp_scores / scores_sum
    loss += - np.log(probs[y[i]])

    # inspired from
    # http://cs231n.github.io/linear-classify/#softmax
    # https://github.com/MyHumbleSelf/cs231n/blob/master/assignment1/cs231n/classifiers/softmax.py
    for j in xrange(num_classes):
      dW[:, j] += (probs[j] - (j == y[i])) * X[i]

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # 500 x 10
  scores = X.dot(W)

  # for numerical stability purpose: http://cs231n.github.io/linear-classify/
  log_c = np.exp(- scores.max())

  exp_scores = log_c * np.exp(scores)

  # 500,
  scores_sum = np.sum(exp_scores, axis=1)

  # 500 x 10
  probs = (exp_scores.T / scores_sum).T

  loss = np.sum(- np.log(probs[np.arange(num_train), y]))

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

