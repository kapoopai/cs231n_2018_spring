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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

        """
        计算第i个样本的dw, Sj - Syi + 1(j!=yi)
        Sj = W[j] * X[i]， Syi = W[y[i]] * X[j]
        dw的计算公式
        if Sj - Syi + 1 > 0:
          L = W[j] * X[i] - W[y[i]] * X[j] + 1
          # dw[j] 就是 L对W[j]求偏导
          dw[j] = X[i]
          # dw[y[i]] 就是 L对W[y[i]]求偏导
          dw[y[i]] = -X[i]
        else :
          dw = 0
        """
        dW[:, y[i]] += -X[i].T    # dw中第y[i]列的元素
        dW[:, j] += X[i].T        # dw中第j列的元素

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss. L2 normal
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  #计算梯度
  # 参考原note中的微分梯度公式
  # https://note.youdao.com/web/#/file/recent/markdown/WEBe2b1f0ad20f1d8b93cffb3f424a6950b/
  dW /= num_train
  dW += reg * W

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
  # N是表示样本数木
  num_train = X.shape[0]
  # N x C 维矩阵，这里针对每个样本计算出打分，并保存到矩阵中
  scores = X.dot(W)
  # 两个矩阵相减，第一个矩阵是打分矩阵，第二个矩阵是Syi的矩阵，再加上安全边界1
  margin = scores - scores[range(0, num_train), y].reshape(-1, 1) + 1  # N x C
  # 将相减之后的矩阵的Syi变成0
  margin[range(num_train), y] = 0
  # margin矩阵中的每个元素求max函数的返回值，即求出了Li
  margin = (margin > 0) * margin
  # margin矩阵中的每个元素相加求平均值，即求出了L
  loss += margin.sum() / num_train
  # 计算L2正则化
  loss += reg * np.sum(W * W)
  pass
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
  # 针对margin函数中的每一个元素来求dw
  # margin[range(0, N), y] = -X[i]或者0, 其他的为X[i]或者0之和
  # 首先间隔margins小于0处赋值为0，其余赋值为1。dWj = X.T.dot(margins)。对应的dWyi = -X.T.dot(margins)。
  # 从这里分析，可以看出梯度的取值只与图像的原始数据有关系，与权重W的关系不大
  counts = (margin > 0).astype(int)
  counts[range(num_train), y] = - np.sum(counts, axis = 1)
  dW += np.dot(X.T, counts) / num_train + reg * W
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
