from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b2'] = np.zeros(num_classes)
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        num_train = X.shape[0]
        num_class = self.params['b1'].shape[0]
        num_hidden_dim = self.params['W1'].shape[0]
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # buf_H = np.reshape(X, (-1, num_hidden_dim)).dot(self.params['W1'])
        # buf_H += self.params['b1']
        # # ReLU
        # buf_H[buf_H < 0] = 0
        #
        # # layer2
        # scores = np.reshape(buf_H, (-1, self.params['W2'].shape[0])).dot(self.params['W2'])
        # scores += self.params['b2']

        caches = {}
        W1 = self.params['W1']
        b1 = self.params['b1']
        scores, caches[0] = affine_relu_forward(X, W1, b1)

        W2 = self.params['W2']
        b2 = self.params['b2']
        scores, caches[1] = affine_relu_forward(scores, W2, b2)
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # # 数值稳定化
        # # print('scores before stably: ', scores)
        # scores -= np.max(scores, axis=1).reshape(num_train, 1)
        # # print('scores after stably: ', scores)
        #
        # scores = np.exp(scores)
        # scores = scores / np.sum(scores, axis=1, keepdims=True)
        # loss = np.sum(-np.log(scores[range(num_train), y]))
        # loss = loss / num_train + \
        #        0.5 * self.reg * (np.sum(self.params['W1'] * self.params['W1']) +
        #                          np.sum(self.params['W2'] * self.params['W2']))
        # # print(loss)

        # scores[np.arange(num_train), y] -= 1
        # # [H * N]  * [N * C] >>> [H * C]
        # dW2 = np.dot(buf_H.T, scores) / num_train + self.reg * self.params['W2']
        # db2 = np.sum(scores, axis=0) / num_train
        # buf_hide = np.dot(self.params['W2'], scores.T)
        # # element > 0
        # buf_H[buf_H > 0] = 1
        # # relu buf_hide
        # buf_relu = buf_hide.T * buf_H
        # # 4*5  * 5*10 >>4*10
        # dW1 = np.dot(np.reshape(X, (-1, num_hidden_dim)).T, buf_relu) / num_train + self.reg * self.params['W1']
        # db1 = np.sum(buf_relu, axis=0) / num_train
        #


        loss, dout = softmax_loss(scores, y)
        dx, dW2, db2 = affine_relu_backward(dout, caches[1])
        dx, dW1, db1 = affine_relu_backward(dx, caches[0])

        # 这里要加上一个W的动量
        grads['W1'] = dW1 + self.reg * W1
        grads['W2'] = dW2 + self.reg * W2
        grads['b1'] = db1
        grads['b2'] = db2

        loss += 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        w_shape_0 = input_dim
        for i in range(self.num_layers):
            params_w_name = "W%s" % (i + 1)
            params_b_name = "b%s" % (i + 1)

            # 最后一层的维度是和类别数保持一致
            if i < (self.num_layers - 1):
                w_shape_1 = hidden_dims[i]
                if self.normalization == 'batchnorm':
                    params_scale_name = "gamma%s" % (i + 1)
                    params_shift_name = "beta%s" % (i + 1)
                    self.params[params_scale_name] = np.ones(w_shape_1)
                    self.params[params_shift_name] = np.zeros(w_shape_1)
            else:
                w_shape_1 = num_classes

            self.params[params_w_name] = np.random.randn(w_shape_0, w_shape_1) * weight_scale

            if i < (self.num_layers - 1):
                w_shape_0 = hidden_dims[i]

            b_dim = w_shape_1
            self.params[params_b_name] = np.zeros(b_dim)

        # for name in self.params:
        #     print('%s, shape: %s: ' % (name, self.params[name].shape))
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
        # print('dropout_param: ', self.dropout_param)
        # print('bn_params: ', self.bn_params)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        neurons = X
        reg = 0
        caches = {}
        dt_caches = {}
        # 按讲义中的内容是对FC层的神经元做dropout，并不包括输入层？
        for i in range(self.num_layers):
            params_w_name = "W%s" % (i + 1)
            params_b_name = "b%s" % (i + 1)
            params_scale_name = "gamma%s" % (i + 1)
            params_shift_name = "beta%s" % (i + 1)
            W = self.params[params_w_name]
            b = self.params[params_b_name]

            # print('loss function, params_w_name: ', params_w_name)
            # print('loss function, w.shape: ', W.shape)

            if i < self.num_layers - 1:
                # 对Hidden层的神经元做BN
                if self.normalization == 'batchnorm':
                    gamma = self.params[params_scale_name]
                    beta = self.params[params_shift_name]
                    # print('self.bn_params[i]: ', self.bn_params[i])
                    scores, caches[i] = affine_bn_relu_forward(neurons, W, b, gamma, beta, self.bn_params[i])
                elif self.normalization == 'layernorm':
                    gamma = self.params[params_scale_name]
                    beta = self.params[params_shift_name]
                    # print('self.bn_params[i]: ', self.bn_params[i])
                    scores, caches[i] = affine_ln_relu_forward(neurons, W, b, gamma, beta, self.bn_params[i])
                    pass
                else:
                    scores, caches[i] = affine_relu_forward(neurons, W, b)
                    pass

                # 对Hidden层的神经元做dropout
                if self.use_dropout:
                    scores, dt_caches[i] = dropout_forward(scores, self.dropout_param)
            else:
                scores, caches[i] = affine_forward(neurons, W, b)

            # 更新神经元
            neurons = scores

            # 求正则化
            reg += np.sum(W*W)

        # print('caches.len: ', caches)
        # for i in caches:
        #     dw, db, _ = caches
        #     print('dw.shape: %s, db.shape: %s' % (dw.shape, db.shape))
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * self.reg * reg

        # print('affine_bn_relu_cache.len: ', len(affine_bn_relu_cache))
        # 计算梯度
        affine_dx = dout
        for i in range(self.num_layers, 0, -1):
            if i < self.num_layers:
                # 对Hidden层的神经元做dropout
                if self.use_dropout:
                    affine_dx = dropout_backward(affine_dx, dt_caches[i - 1])

                if self.normalization == 'batchnorm':
                    affine_dx, affine_dw, affine_db, dgamma, dbeta = \
                        affine_bn_relu_backward(affine_dx, caches[i - 1])
                    grads['beta' + str(i)] = dbeta
                    grads['gamma' + str(i)] = dgamma
                elif self.normalization == 'layernorm':
                    affine_dx, affine_dw, affine_db, dgamma, dbeta = \
                        affine_ln_relu_backward(affine_dx, caches[i - 1])
                    grads['beta' + str(i)] = dbeta
                    grads['gamma' + str(i)] = dgamma
                    pass
                else:
                    affine_dx, affine_dw, affine_db = affine_relu_backward(affine_dx, caches[i - 1])

            else:
                affine_dx, affine_dw, affine_db = affine_backward(affine_dx, caches[i - 1])
                # print('affine_dw: ', caches[str(i - 1)])

            grads['W' + str(i)] = affine_dw + self.reg * self.params['W' + str(i)]
            grads['b' + str(i)] = affine_db
            # print('grads w%d shape: %s' % (i, grads['W' + str(i)].shape))
            # print('params w%d shape: %s' % (i, self.params['W' + str(i)].shape))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return loss, grads

    # def loss(self, X, y=None):
    #     """
    #     Compute loss and gradient for the fully-connected net.
    #
    #     Input / output: Same as TwoLayerNet above.
    #     """
    #     X = X.astype(self.dtype)
    #     mode = 'test' if y is None else 'train'
    #
    #     # Set train/test mode for batchnorm params and dropout param since they
    #     # behave differently during training and testing.
    #     if self.dropout_param is not None:
    #         self.dropout_param['mode'] = mode
    #     if self.normalization == 'batchnorm':
    #         for bn_param in self.bn_params:
    #             bn_param[mode] = mode
    #
    #     scores = None
    #     ############################################################################
    #     # TODO: Implement the forward pass for the fully-connected net, computing  #
    #     # the class scores for X and storing them in the scores variable.          #
    #     #                                                                          #
    #     # When using dropout, you'll need to pass self.dropout_param to each       #
    #     # dropout forward pass.                                                    #
    #     #                                                                          #
    #     # When using batch normalization, you'll need to pass self.bn_params[0] to #
    #     # the forward pass for the first batch normalization layer, pass           #
    #     # self.bn_params[1] to the forward pass for the second batch normalization #
    #     # layer, etc.                                                              #
    #     ############################################################################
    #     pass
    #     hidden_layers, caches = {}, {}
    #     dp_caches = range(self.num_layers - 1)
    #     hidden_layers[0] = X
    #     for i in range(self.num_layers):
    #         W, b = self.params['W' + str(i + 1)], self.params['b' + str(i + 1)]
    #         if i == self.num_layers - 1:
    #             hidden_layers[i + 1], caches[i] = affine_forward(hidden_layers[i], W, b)
    #         else:
    #             if self.normalization == 'batchnorm':
    #                 gamma, beta = self.params['gamma' + str(i + 1)], self.params['beta' + str(i + 1)]
    #                 hidden_layers[i + 1], caches[i] = affine_bn_relu_forward(hidden_layers[i], W, b, gamma, beta,
    #                                                                          self.bn_params[i])
    #             else:
    #                 hidden_layers[i + 1], caches[i] = affine_relu_forward(hidden_layers[i], W, b)
    #             if self.use_dropout:
    #                 hidden_layers[i + 1], dp_caches[i] = dropout_forward(hidden_layers[i + 1], self.dropout_param)
    #
    #     scores = hidden_layers[self.num_layers]
    #     ############################################################################
    #     #                             END OF YOUR CODE                             #
    #     ############################################################################
    #
    #     # If test mode return early
    #     if mode == 'test':
    #         return scores
    #
    #     loss, grads = 0.0, {}
    #     ############################################################################
    #     # TODO: Implement the backward pass for the fully-connected net. Store the #
    #     # loss in the loss variable and gradients in the grads dictionary. Compute #
    #     # data loss using softmax, and make sure that grads[k] holds the gradients #
    #     # for self.params[k]. Don't forget to add L2 regularization!               #
    #     #                                                                          #
    #     # When using batch normalization, you don't need to regularize the scale   #
    #     # and shift parameters.                                                    #
    #     #                                                                          #
    #     # NOTE: To ensure that your implementation matches ours and you pass the   #
    #     # automated tests, make sure that your L2 regularization includes a factor #
    #     # of 0.5 to simplify the expression for the gradient.                      #
    #     ############################################################################
    #     pass
    #     loss, dscores = softmax_loss(scores, y)
    #     dhiddens = {}   # range(self.num_layers + 1)
    #     dhiddens[self.num_layers] = dscores
    #     for i in range(self.num_layers, 0, -1):
    #         if i == self.num_layers:
    #             dhiddens[i - 1], grads['W' + str(i)], grads['b' + str(i)] = affine_backward(dhiddens[i], caches[i - 1])
    #         else:
    #             if self.use_dropout:
    #                 dhiddens[i] = dropout_backward(dhiddens[i], dp_caches[i - 1])
    #             if self.normalization == 'batchnorm':
    #                 dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dhiddens[i], caches[i - 1])
    #                 dhiddens[i - 1], grads['W' + str(i)], grads['b' + str(i)] = dx, dw, db
    #                 grads['gamma' + str(i)], grads['beta' + str(i)] = dgamma, dbeta
    #             else:
    #                 dx, dw, db = affine_relu_backward(dhiddens[i], caches[i - 1])
    #                 dhiddens[i - 1], grads['W' + str(i)], grads['b' + str(i)] = dx, dw, db
    #         loss += 0.5 * self.reg * np.sum(self.params['W' + str(i)] ** 2)
    #         grads['W' + str(i)] += self.reg * self.params['W' + str(i)]
    #     ############################################################################
    #     #                             END OF YOUR CODE                             #
    #     ############################################################################
    #
    #     return loss, grads
