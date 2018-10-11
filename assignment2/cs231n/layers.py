from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    # print('x.shape: ', x.shape)
    # print('w.shape: ', w.shape)
    # print('b.shape: ', b.shape)
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    out = np.reshape(x, [-1, w.shape[0]]).dot(w)
    out += b
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    dx = np.reshape(dout.dot(w.T), x.shape)
    dw = x.reshape(dout.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0)       # 偏导是1 这里按矩阵的维度来考虑即可
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # 不能用下面这条语句，这条语句把 x 的值改变了
    # out[out < 0] = 0.0
    # out = x
    out = np.maximum(0, x)
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # dx = dout
    # dx[x < 0] = 0
    dx = dout * (x > 0)
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    在训练过程中，我们通过归一化输入数据，来训练采样均值和采样方差。同时，我们需要训练一个滑动均值，
    这个运行是均值是通过每一个特征值的均值和方差来计算的，并且保持指数衰减，这个滑动均值用来在测试阶
    段归一化数据

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    运行时均值和方差在每个时间步时按以下公式进行更新，并且通过使用一个momentum动量参数来保障指数衰减

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    注意：batch normalization的论文中建议了一种不同的时间步行为：通过使用很大的图像训练数据，来计算
    样本均值和方差，而不是通过使用滑动平均。这个类的实现时我们使用的是滑动平均，因为这样就不需要额外增加
    一个判断操作

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """

    # print('x.shape: %s, gamma: %s, beta: %s' % (x.shape, gamma, beta))
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # 计算小批次输入数据的均值和方差，并且使用 gamma 和 beta 来缩放和位移          #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # 保存中间结果在cache中                                                  #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # 根据样本均值和方差以及动量来更新滑动均值和滑动方差                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # sample_mean.shape: (D,),
        # 注意这里是对批量的样本维度求均值和方差，不是对每个样本的所有数据求均值和方差
        sample_mean = np.mean(x, axis=0)
        # print('sample_mean.shape: ', sample_mean.shape)
        # 方差
        sample_var = np.var(x, axis=0)
        # sample_var = np.sum((x - sample_mean) ** 2, axis=0) / N
        # print('sample_var.shape: ', sample_var.shape)
        # 映射值 - 均值 / 标准差
        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
        # print('x_hat.shape: ', x_hat.shape)

        # 滑动均值
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        # 缩放和平移
        y = gamma * x_hat + beta

        # 入参已经做了 ReLU(X × W1) x W2 的操作
        out = y

        cache = (x, sample_mean, sample_var, x_hat, gamma, beta, eps)
        pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # 测试时使用滑动平均值和滑动方差
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta
        pass
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # 推不出来。。。。。。全靠论文
    N = dout.shape[0]
    x, sample_mean, sample_var, x_hat, gamma, beta, eps = cache
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)
    dx_hat = dout * gamma
    dx_var = np.sum(-1.0 / 2 * dx_hat * (x - sample_mean) / (sample_var + eps) ** (3.0 / 2), axis=0)
    dx_mean = np.sum(-1 / np.sqrt(sample_var + eps) * dx_hat, axis=0) + \
              1.0 / N * dx_var * np.sum(-2 * (x - sample_mean), axis=0)
    dx = 1 / np.sqrt(sample_var + eps) * dx_hat + dx_var * 2.0 / N * (x - sample_mean) + 1.0 / N * dx_mean
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    N = dout.shape[0]
    x, sample_mean, sample_var, x_hat, gamma, beta, eps = cache
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)
    dx_hat = dout * gamma
    dx_var = np.sum(-1.0 / 2 * dx_hat * (x - sample_mean) / (sample_var + eps) ** (3.0 / 2), axis=0)
    dx_mean = np.sum(-1 / np.sqrt(sample_var + eps) * dx_hat, axis=0)
    dx = 1 / np.sqrt(sample_var + eps) * dx_hat + dx_var * 2.0 / N * (x - sample_mean) + 1.0 / N * dx_mean
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    N, D = x.shape
    mue = np.mean(x, axis=1)
    theta = np.var(x, axis=1)
    # x_hat = ((x.T - mue.reshape(1, N)) / np.sqrt(theta.reshape(1, N) + eps)).T
    x_hat = (x - mue[:, None]) / np.sqrt(theta[:, None] + eps)
    out = gamma * x_hat + beta

    # print('gamma.shape: %s, beta.shape: %s, out.shape: %s' % (gamma.shape, beta.shape, out.shape))
    cache = (x, mue, theta, x_hat, gamma, beta, eps)
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    N = dout.shape[0]
    x, mue, theta, x_hat, gamma, beta, eps = cache
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)
    dx_hat = dout * gamma
    print('theta.shape: %s, eps: %s, dx_hat.shape: %s' % (theta.shape, eps, dx_hat.shape))
    dx_var = np.sum(-1.0 / 2 * dx_hat * (x - mue[:, None]) / (theta[:, None] + eps) ** (3.0 / 2), axis=1)
    dx_mean = np.sum(-1 / np.sqrt(theta[:, None] + eps) * dx_hat, axis=1)
    print('dx_var.shape: %s, dx_mean.shape: %s' % (dx_var.shape, dx_mean.shape))

    # dx = 1 / np.sqrt(theta + eps) * dx_hat + dx_var * 2.0 / N * (x - dx_mean) + 1.0 / N * dx_mean

    # dx 有点问题
    dx = 1 / np.sqrt(theta[:, None] + eps) * dx_hat +\
         dx_var[:, None] * 2.0 / N * (x - dx_mean[:, None]) + 1.0 / N * dx_mean[:, None]
    print('dx.shape: %s, dgamma.shape: %s, dbeta.shape: %s' % (dx.shape, dgamma.shape, dbeta.shape))
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *号是解元组操作，对元组中的每个参数作为入参执行函数
        mask = np.random.rand(*x.shape) < (1 - p)
        # mask.shape = x.shape
        mask = mask / (1 - p)
        out = mask * x
        pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        mask = np.array(range(x.shape[0]))
        out = x
        pass
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        pass
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)
    # print('stride: %d, pad: %d' % (stride, pad))

    input_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    # print('x: ', x)
    # print('input_pad: ', input_pad)
    # print('input_pad.shape: ', input_pad.shape)

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_o = 1 + (H + 2 * pad - HH) // stride
    W_o = 1 + (W + 2 * pad - WW) // stride
    out = np.zeros((N, F, H_o, W_o))
    # print('H_o: %d, W_o: %d' % (H_o, W_o))

    for i in range(W_o):
        for j in range(H_o):
            x_padded_mask = input_pad[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
            for k in range(F):
                out[:, k, i, j] = np.sum(x_padded_mask * w[k, :, :, :], axis=(1, 2, 3))

    out += b[None, :, None, None]

    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    dx = np.zeros_like(x)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    db = np.sum(dout, axis=(0, 2, 3))

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    for i in range(H_out):
        for j in range(W_out):
            x_pad_masked = x_pad[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
            for k in range(F):  # compute dw
                dw[k, :, :, :] += np.sum(x_pad_masked * (dout[:, k, i, j])[:, None, None, None], axis=0)
            for n in range(N):  # compute dx_pad
                dx_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] += np.sum((w[:, :, :, :] *
                                                                                                (dout[n, :, i, j])[:,
                                                                                                None, None, None]),
                                                                                               axis=0)
    dx = dx_pad[:, :, pad:-pad, pad:-pad]
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    out_height = H // pool_height
    out_width = W // pool_width
    out = np.zeros((N, C, out_height, out_width))
    for i in range(out_height):
        for j in range(out_width):
            mask = x[:, :, i * stride:i * stride + pool_height, j * stride:j * stride + pool_width]
            out[:, :, i, j] = np.max(mask, axis=(2, 3))
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    dx = np.zeros_like(x)
    out_height = H // pool_height
    out_width = W // pool_width
    for i in range(out_height):
        for j in range(out_width):
            # x, dx has the same dimension, so does x_mask and dx_mask
            x_mask = x[:, :, i * stride:i * stride + pool_height, j * stride:j * stride + pool_width]
            dx_mask = dx[:, :, i * stride:i * stride + pool_height, j * stride:j * stride + pool_width]
            # flags: only the max value is True, others are False
            flags = np.max(x_mask, axis=(2, 3), keepdims=True) == x_mask
            dx_mask += flags * (dout[:, :, i, j])[:, :, None, None]
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = x.shape
    # 这里用的transpose接口
    input = x.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    out, cache = batchnorm_forward(input, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = dout.shape
    dout_trans = dout.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    dx, dgamma, dbeta = batchnorm_backward_alt(dout_trans, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    # print('dx: %s, dgamma: %s, dbeta: %s' % (dx.shape, dgamma.shape, dbeta.shape))
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    N, C, H, W = x.shape
    trans_x = np.reshape(x, [N, G, C // G, H, W])
    x_mean = np.mean(trans_x, axis=(2, 3, 4), keepdims=True)
    x_var = np.var(trans_x, axis=(2, 3, 4), keepdims=True)
    x_hat = (trans_x - x_mean) / np.sqrt(x_var + eps)
    x_hat = np.reshape(x_hat, [N, C, H, W])
    out = x_hat * gamma + beta
    cache = (x, x_mean, x_var, x_hat, gamma, beta, G, eps, out)
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    x, x_mean, x_var, x_hat, gamma, beta, G, eps, out = cache
    # print('x.shape: ', x.shape)
    # print('x_mean.shape: ', x_mean.shape)
    # print('x_var.shape: ', x_var.shape)
    # print('x_hat.shape: ', x_hat.shape)
    # print('gamma.shape: ', gamma.shape)
    # print('beta.shape: ', beta.shape)
    # print('out.shape: ', out.shape)

    N, C, H, W = dout.shape

    dbeta = np.sum(dout, axis=(0, 2, 3)).reshape(1, C, 1, 1)
    dgamma = np.sum(dout * x_hat, axis=(0, 2, 3)).reshape(1, C, 1, 1)
    dx_hat = dout * gamma
    print('dx_hat.shape: ', dx_hat.shape)

    trans_dxhat = np.reshape(dx_hat, [N, G, C // G, H, W])
    trans_x = np.reshape(x, trans_dxhat.shape)
    print('trans_dxhat.shape: ', trans_dxhat.shape)

    dx_var = np.sum(-1.0 / 2 * trans_dxhat * (trans_x - x_mean) / (x_var + eps) ** (3.0 / 2),
                    axis=(2, 3, 4), keepdims=True)
    print('dx_var.shape: ', dx_var.shape)
    dx_mean = np.sum(-1 / np.sqrt(x_var + eps) * trans_dxhat, axis=(2, 3, 4), keepdims=True)
    print('dx_mean.shape: ', dx_mean.shape)

    temp1 = 1 / np.sqrt(x_var + eps)
    # print('temp1.shape: ', temp1.shape)
    temp2 = dx_var * 2.0 / G * (trans_x - x_mean)
    # print('temp2.shape: ', temp2.shape)
    temp3 = 1.0 / G * dx_mean
    # print('temp3.shape: ', temp3.shape)

    dx = temp1 + temp2 + temp3
    print('dx.shape: ', dx.shape)
    dx = dx.reshape(N, C, H, W)
    print('dx.shape: %s, dgamma.shape: %s, dbeta.shape: %s' % (dx.shape, dgamma.shape, dbeta.shape))
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
