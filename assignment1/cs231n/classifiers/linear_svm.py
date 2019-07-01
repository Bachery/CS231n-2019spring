from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    
#     for i in range(num_train):
#         scores = X[i].dot(W)
#         correct_class_score = scores[y[i]]
#         for j in range(num_classes):
#             if j == y[i]:
#                 continue
#             margin = scores[j] - correct_class_score + 1 # note delta = 1
#             if margin > 0:
#                 loss += margin

#     # Right now the loss is a sum over all training examples, but we want it
#     # to be an average instead so we divide by num_train.
#     loss /= num_train

#     # Add regularization to the loss.
#     loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#     pass
    #逐个计算每个样本的loss
    for i in xrange(num_train):
        scores = X[i].dot(W)                 # 每个样本在各个分类上的得分
        correct_class_score = scores[y[i]]   # 每个样本在其正确分类上的得分
        for j in xrange(num_classes):
            if j == y[i]:                    # 根据公式，跳过 j==y[i]
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,y[i]] += -X[i,:].T
                dW[:,j] += X[i, :].T
    
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#     scores = X.dot(W)
#     margin = scores - scores[np.arange(num_train), y].reshape(num_train, 1) + 1
#     margin[np.arange(num_train), y] = 0.0 #这一列不该计算，归零
#     margin = (margin > 0) * margin
#     loss += margin.sum() / num_train
#     loss += 0.5 * reg * np.sum(W * W)

#     pass
    scores = X.dot(W)          # scores (N,C)
    num_train = X.shape[0]

    # 利用 np.arange()将correct_class_score变成(num_train, y)的矩阵
    correct_class_score = scores[np.arange(num_train), y]
    correct_class_score = np.reshape(correct_class_score, (num_train,-1))
    margins = scores - correct_class_score + 1
    margins = np.maximum(0, margins)
    margins[np.arange(num_train),y] = 0     # 由于计算了 j=y[i] 的情况，置0
    
    loss += np.sum(margins) / num_train
    loss += reg * np.sum( W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#     margin = (margin > 0) * 1
#     row_sum = np.sum(margin, axis=1)
#     margin[np.arange(num_train), y] = -row_sum
#     dW = X.T.dot(margin)/num_train + reg * W
    
#     pass
    margins[margins > 0] = 1
    # 因为j=y[i]的那一个元素的grad要计算 >0 的那些次数次
    row_sum = np.sum(margins, axis=1)
    margins[np.arange(num_train), y] = -row_sum.T
    
    dW = np.dot(X.T, margins)
    dW /= num_train
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
