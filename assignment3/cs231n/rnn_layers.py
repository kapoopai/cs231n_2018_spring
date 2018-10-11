from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    #公式 next_h = tanh(prev_h * Wh + x * Wx)
    # next_h.shape = (N, H)
    next_h = np.tanh(prev_h.dot(Wh) + x.dot(Wx) + b)
    cache = (x, prev_h, Wx, Wh, b, next_h)
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    #tanh(x)的导数为 1 - (tanh(x))^2
    x, prev_h, Wx, Wh, b, next_h = cache
    # dtanh.shape:(N,H)
    dtanh = dnext_h * (1 - next_h * next_h)
    # dx.shape:(N,D)
    dx = dtanh.dot(Wx.T)
    # dWx.shape:(D,H)
    dWx = x.T.dot(dtanh)
    # dprev_h.shape:(N,H)
    dprev_h = dtanh.dot(Wh.T)
    # dWh.shape:(H,H)
    dWh = prev_h.T.dot(dtanh)
    # db.shape:(H,)
    db = np.sum(dtanh, axis=0)
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    N, T, D = x.shape
    prev_h = h0
    h, cache = [], []
    for seq in range(T):
        input_x = np.transpose(x, (1, 0, 2))
        next_h, cur_cache = rnn_step_forward(input_x[seq], prev_h, Wx, Wh, b)
        h.append(next_h)
        cache.append(cur_cache)
        prev_h = next_h

    h = np.transpose(h, (1, 0, 2))
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H). 
    
    NOTE: 'dh' contains the upstream gradients produced by the 
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    N, T, H = dh.shape
    dout_h = np.transpose(dh, (1, 0, 2))
    D = cache[0][0].shape[1]
    dx, dWx, dWh, db = [], np.zeros((D, H)), np.zeros((H, H)), np.zeros(H)
    dh_cur = np.zeros((N, H))
    for step in range(T-1, -1, -1):
        # print('dprev_h: ', dprev_h)
        # 这里dout要把dh加起来
        dx_cur, dh_cur, dWx_cur, dWh_cur, db_cur = \
            rnn_step_backward(dout_h[step] + dh_cur, cache[step])
        dx.append(dx_cur)
        # 这里要把他们都加起来
        dWx += dWx_cur
        dWh += dWh_cur
        db += db_cur

    # dh0 这里直接等于最后一个dh，而不是所有的dh相加
    dh0 = dh_cur
    dx = np.transpose(np.array(dx[::-1]), (1, 0, 2))
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    out = W[x, :]
    cache = (x, W)
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    x, W = cache
    dW = np.zeros_like(W)
    # x占位的地方非0,其他为0
    np.add.at(dW, x, dout)
    #dW = np.add.at(dW, x, dout) # 这句话是错误的
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    N, H = prev_h.shape
    # 公式 a=Wx × xt + Wh × ht−1 + b
    # act_x.shape: (N, 4H)
    act_x = x.dot(Wx) + prev_h.dot(Wh) + b
    # print('act_x.shape: ', act_x.shape)

    # 计算四个门
    i_gate = sigmoid(act_x[:, :H])
    f_gate = sigmoid(act_x[:, H:2*H])
    o_gate = sigmoid(act_x[:, 2 * H:3 * H])
    g_gate = np.tanh(act_x[:, 3 * H:])
    # print('f.shape: %s, i.shape: %s, g.shape: %s, o.shape: %s' %
    #       (f_gate.shape, i_gate.shape, g_gate.shape, o_gate.shape))

    # 计算 next_c
    next_c = f_gate * prev_c + i_gate * g_gate

    # 计算 next_h
    next_h = np.tanh(next_c) * o_gate
    
    cache = (x, prev_h, prev_c, Wx, Wh, b, act_x, f_gate, i_gate, g_gate, o_gate, next_c, next_h)
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    x, prev_h, prev_c, Wx, Wh, b, act_x, f_gate, i_gate, g_gate, o_gate, next_c, next_h = cache

    # 这里都是有两部分组成，由 dnext_h 和 dnext_c 推导的相加得来
    do_gate = np.tanh(next_c) * dnext_h
    di_gate = (dnext_c + dnext_h * o_gate * (1 - np.tanh(next_c) ** 2)) * g_gate
    df_gate = (dnext_c + dnext_h * o_gate * (1 - np.tanh(next_c) ** 2)) * prev_c
    dg_gate = (dnext_c + dnext_h * o_gate * (1 - np.tanh(next_c) ** 2)) * i_gate
    dprev_c = (dnext_c + dnext_h * o_gate * (1 - np.tanh(next_c) ** 2)) * f_gate

    # dstack 这里将H一分为四之后，反向传播要把四者合并起来
    dstack = np.hstack([i_gate*(1-i_gate)*di_gate, f_gate*(1-f_gate)*df_gate,
                        o_gate*(1-o_gate)*do_gate, (1-g_gate*g_gate)*dg_gate])

    # 前面部分就容易推导了，将计算图从dstack处划分之后，前面就可以简化，这个是个很好的技巧，
    # 比上来就直接求 dx 要容易的多了
    dx = dstack.dot(Wx.T)
    dWx = x.T.dot(dstack)
    dprev_h = dstack.dot(Wh.T)
    dWh = prev_h.T.dot(dstack)
    db = np.sum(dstack, axis=0)
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    (N, T, D), H = x.shape, h0.shape[1]
    cell = np.zeros((N, H))
    cache = []
    h = []

    for step in range(T):
        h0, cell, prev_cache = lstm_step_forward(x[:, step, :], h0, cell, Wx, Wh, b)
        cache.append(prev_cache)
        h.append(h0)

    h = np.transpose(h, (1, 0, 2))
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    N, T, H = dh.shape
    dout_h = np.transpose(dh, (1, 0, 2))
    D = cache[0][0].shape[1]
    dx, dWx, dWh, db = [], np.zeros((D, 4 * H)), np.zeros((H, 4 * H)), np.zeros(4 * H)
    dh_cur = np.zeros((N, H))
    dc_cur = np.zeros((N, H))
    for step in range(T - 1, -1, -1):
        # print('dprev_h: ', dprev_h)
        # 这里dout要把dh加起来, dc_cur 这里不用加
        # 如果用 cache[step][-2] + dc_cur 替换入参 dc_cur 是不对的，暂时我也不知道原因。
        dx_cur, dh_cur, dc_cur, dWx_cur, dWh_cur, db_cur = \
            lstm_step_backward(dout_h[step] + dh_cur, dc_cur, cache[step])
        dx.append(dx_cur)
        # 这里要把他们都加起来
        dWx += dWx_cur
        dWh += dWh_cur
        db += db_cur

    # dh0 这里直接等于最后一个dh，而不是所有的dh相加
    dh0 = dh_cur
    dx = np.transpose(np.array(dx[::-1]), (1, 0, 2))
    pass
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
