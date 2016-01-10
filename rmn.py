import numpy
import theano

from blocks.roles import add_role, PARAMETER
from collections import OrderedDict
from theano import tensor

rng = numpy.random.RandomState(4321)


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
        add_role(tparams[kk], PARAMETER)
    return tparams


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


# some utilities
def ortho_weight(ndim):
    W = rng.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * rng.randn(nin, nout)
    return W.astype('float32')


# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru_rmn': ('param_init_gru_rmn', 'gru_rmn_layer'),
          }


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(params, prefix='ff', nin=None, nout=None, ortho=True,
                       add_bias=True):
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    if add_bias:
        params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')
    return params


def fflayer(tparams, state_below, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', add_bias=True, **kwargs):
    preact = tensor.dot(state_below, tparams[_p(prefix, 'W')])
    if add_bias:
        preact += tparams[_p(prefix, 'b')]
    return eval(activ)(preact)


# GRU-RMN layer
def param_init_gru_rmn(params, prefix='gru_rmn', nin=None, dim=None,
                       vocab_dim=None, memory_dim=None, memory_size=None):

    # first GRU params
    U = numpy.concatenate([ortho_weight(dim), ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U
    params[_p(prefix, 'Wx')] = norm_weight(nin, dim)
    params[_p(prefix, 'Ux')] = ortho_weight(dim)
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # memory block params
    params[_p(prefix, 'M')] = norm_weight(vocab_dim, memory_dim)
    params[_p(prefix, 'C')] = norm_weight(vocab_dim, memory_dim)
    params[_p(prefix, 'T')] = norm_weight(memory_size, memory_dim)

    # second GRU params
    params[_p(prefix, 'Wz')] = norm_weight(dim, memory_dim)
    params[_p(prefix, 'Wr')] = norm_weight(dim, memory_dim)
    params[_p(prefix, 'W2')] = norm_weight(dim, memory_dim)
    params[_p(prefix, 'Uz')] = ortho_weight(dim)
    params[_p(prefix, 'Ur')] = ortho_weight(dim)
    params[_p(prefix, 'U2')] = ortho_weight(dim)

    return params


def gru_rmn_layer(tparams, state_below, prefix='gru_rmn', mask=None,
                  memory_size=15, **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    steps = tensor.arange(state_below.shape[0])

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_ = tensor.dot(
        state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    state_belowx = tensor.dot(
        state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]

    def _step_slice(idx, m_, x_, xx_,
                    h1_, h2_,
                    ctx,  # input batch, 2D
                    U, Ux, M, C, T,
                    Wz, Uz, Wr, Ur, W2, U2):

        # first layer GRU
        preact = tensor.dot(h1_, U)
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        preactx = tensor.dot(h1_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        h1 = tensor.tanh(preactx)
        h1 = u * h1_ + (1. - u) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h1_

        # memory block
        n_back = tensor.max(idx - memory_size, 0)
        xi = ctx[n_back:idx, :]
        Mi = M[xi, :]
        Ci = C[xi, :]
        Ti = T[-n_back:, :]

        preact = tensor.dot((Mi + Ti), h1)
        pt = tensor.nnet.softmax(preact)
        st = tensor.dot(Ci.T, pt)

        # function g, as another GRU
        r2 = tensor.nnet.sigmoid(tensor.dot(Wr, st) + tensor.dot(Ur, h1))
        u2 = tensor.nnet.sigmoid(tensor.dot(Wz, st) + tensor.dot(Uz, h1))

        preactx = tensor.dot(U2, (r2 * h1))
        preactx = preactx + tensor.dot(W2, st)
        h2 = tensor.tanh(preactx)
        h2 = u2 * h2 + (1. - u2) * h1
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h2_

        return h1, h2

    seqs = [steps, mask, state_below_, state_belowx]
    _step = _step_slice

    rval, _ = theano.scan(_step,
                          sequences=seqs,
                          outputs_info=[tensor.alloc(0., n_samples, dim),
                                        tensor.alloc(0., n_samples, dim)],
                          non_sequences=[tparams[_p(prefix, 'U')],
                                         tparams[_p(prefix, 'Ux')],
                                         tparams[_p(prefix, 'M')],
                                         tparams[_p(prefix, 'C')],
                                         tparams[_p(prefix, 'T')],
                                         tparams[_p(prefix, 'Wz')],
                                         tparams[_p(prefix, 'Uz')],
                                         tparams[_p(prefix, 'Wr')],
                                         tparams[_p(prefix, 'Ur')],
                                         tparams[_p(prefix, 'W2')],
                                         tparams[_p(prefix, 'U2')]],
                          name=_p(prefix, '_layers'),
                          n_steps=nsteps,
                          strict=True)
    return rval
