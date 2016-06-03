import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape)        # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

        # http://cs231n.github.io/optimization-1/
        dW[:,j] += X[i]
        dW[:,y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """

  loss = 0.0
  dW = np.zeros(W.shape)        # initialize the gradient as zero

  num_train = X.shape[0]
  num_classes = W.shape[1]
  delta = 1

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # 500 x 10
  scores = X.dot(W)
  # http://stackoverflow.com/questions/23435782/numpy-selecting-specific-column-index-per-row-by-using-a-list-of-indexes
  # 500 x 1
  correct_class_scores = scores[np.arange(num_train), y].reshape([-1, 1])
  # 500 x 10
  margins = np.maximum(0, scores - correct_class_scores + delta)
  margins[np.arange(num_train), y] = 0
  loss = np.sum(margins[margins > 0])
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  # # 3073 x 10
  # print(dW.shape)
  # # 500 x 3073
  # print(X.shape)
  # # 500 x 10
  # print(margins.shape)

  # print(margins[0:2])
  # print(y[0:2])

  # inspired from
  # https://github.com/huyouare/CS231n/blob/master/assignment1/cs231n/classifiers/linear_svm.py
  margins[margins > 0] = 1
  margins[np.arange(num_train), y] = -1 * margins.sum(axis=1)
  dW = X.T.dot(margins)

  # try to vectorize loop-by-loop
  # for i in xrange(num_train):
    # dW += X[i].reshape(1, -1).T.dot(margins[i].reshape(1, -1) > 0)
    # dW[:,y[i]] -= (X[i].reshape(1, -1).T.dot(margins[i].reshape(1, -1) > 0)).sum(axis=1)

    # for j in xrange(num_classes):
    #   if j == y[i]:
    #     continue
    #   marg = margins[i][j]
    #   if marg > 0:
    #     dW[:,j] += X[i]
    #     dW[:,y[i]] -= X[i]

  dW /= num_train
  dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return loss, dW
