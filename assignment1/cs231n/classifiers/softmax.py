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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # https://zhuanlan.zhihu.com/p/21485970 参考
  num_train = X.shape[0]
  num_class = W.shape[1]

  for i in range(num_train):
    #计算得分
    scores = X[i].dot(W)

    # 考虑到数值稳定性，指数化之前要减去最大值
    sc_max = scores - np.max(scores)

    #对得分进行指数化
    sc_exp = np.exp(sc_max)

    #对分数进行归一化
    sc_rate = sc_exp/np.sum(sc_exp)

    # 计算 cross_entropy 这里用自然对数
    loss += -np.log(sc_rate[y[i]] / np.sum(sc_rate))

    # 开始计算梯度
    for j in range(num_class):
      if j == y[i]:
        dW[:, j] += (sc_rate[j] - 1) * X[i]
      else:
        dW[:, j] += (sc_rate[j] - 0) * X[i]

  loss = loss / num_train
  # Add regularization to the loss. L2 normal
  loss += reg * np.sum(W * W)

  dW = dW / num_train + reg * W
  pass
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X.dot(W)  #N by C

  # 将scores矩阵按行找出最大值做平移，实际上我觉得这里是不是可以直接用矩阵的最大值来平移
  sc_max = np.reshape(np.max(scores, axis = 1), (num_train, 1))

  # 将每行归一化
  prob = np.exp(scores - sc_max)/np.sum(np.exp(scores-sc_max), axis = 1, keepdims = True)

  # 创建一个特征矩阵, 只保留 Syi的softmax值
  keepProb = np.zeros_like(prob)
  keepProb[np.arange(num_train), y] = 1.0

  loss += -np.sum(keepProb * np.log(prob)) / num_train + reg * np.sum(W*W)

  # 这里两处的操作等价
  # for j in range(num_class):
  #   if j == y[i]:
  #     dW[:, j] += (sc_rate[j] - 1) * X[i]
  #   else:
  #     dW[:, j] += (sc_rate[j] - 0) * X[i]
  dW += -np.dot(X.T, keepProb - prob)/num_train + reg * W

  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

