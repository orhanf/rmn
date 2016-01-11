import ipdb
import numpy
import sys
import theano
import warnings

from collections import OrderedDict
from theano.ifelse import ifelse
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import (_p, norm_weight, ortho_weight)

rng = numpy.random.RandomState(4321)


# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru_rmn': ('param_init_gru_rmn', 'gru_rmn_layer'),
          }


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        state_before * 0.5)
    return proj


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
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')
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
                  memory_size=15, x=None, **kwargs):
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
        n_back = idx - memory_size
        n_back = ifelse(tensor.lt(n_back, 0),
                        tensor.zeros_like(n_back), n_back)
        xi = ctx[n_back:idx, :].flatten()
        Mi = M[xi, :]
        Ci = C[xi, :]
        Ti = T[-n_back:, :]  # so called slicing

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
                          non_sequences=[x,  # will be attended
                                         tparams[_p(prefix, 'U')],
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


class RMN(object):

    def __init__(self, options):
        self.options = options

        self.params = None
        self.tparams = None
        self.f_next = None
        self.f_log_probs = None

    def init_params(self):
        options = self.options
        params = OrderedDict()
        # embedding
        params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])

        # rmn layer
        params = get_layer(options['encoder'])[0](
            params, prefix='encoder', nin=options['dim_word'],
            dim=options['dim'], vocab_dim=options['vocab_dim'],
            memory_dim=options['memory_dim'],
            memory_size=options['memory_size'])

        # readout
        params = get_layer('ff')[0](params, prefix='ff_logit_lstm',
                                    nin=options['dim'],
                                    nout=options['dim_word'], ortho=False)
        params = get_layer('ff')[0](params, prefix='ff_logit_prev',
                                    nin=options['dim_word'],
                                    nout=options['dim_word'], ortho=False)
        params = get_layer('ff')[0](params, prefix='ff_logit',
                                    nin=options['dim_word'],
                                    nout=options['n_words'])
        self.params = params
        self.init_tparams()

    def load_params(self, saveto):
        if self.params is None:
            self.init_params()
        pp = numpy.load(saveto)
        for kk, vv in self.params.iteritems():
            if kk not in pp:
                warnings.warn('%s is not in the archive' % kk)
                continue
            self.params[kk] = pp[kk]

    def init_tparams(self):
        if self.params is None:
            self.init_params()
        tparams = OrderedDict()
        for kk, pp in self.params.iteritems():
            tparams[kk] = theano.shared(self.params[kk], name=kk)
        self.tparams = tparams

    def build_model(self):
        if self.tparams is None:
            self.init_tparams()
        tparams = self.tparams
        options = self.options

        opt_ret = dict()

        use_noise = theano.shared(numpy.float32(0.))

        # description string: #words x #samples
        x = tensor.matrix('x', dtype='int64')
        x_mask = tensor.matrix('x_mask', dtype='float32')

        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        # input
        emb = tparams['Wemb'][x.flatten()]
        emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
        emb_shifted = tensor.zeros_like(emb)
        emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
        emb = emb_shifted
        opt_ret['emb'] = emb

        # pass through gru layer, recurrence here
        proj = get_layer(options['encoder'])[1](
            tparams, emb, prefix='encoder', x=x, mask=x_mask,
            memory_size=options['memory_size'])

        proj_h = proj[1]
        opt_ret['proj_h'] = proj_h

        # compute word probabilities
        logit_lstm = get_layer('ff')[1](tparams, proj_h,
                                        prefix='ff_logit_lstm', activ='linear')
        logit_prev = get_layer('ff')[1](tparams, emb,
                                        prefix='ff_logit_prev', activ='linear')
        logit = tensor.tanh(logit_lstm+logit_prev)
        logit = get_layer('ff')[1](tparams, logit, prefix='ff_logit',
                                   activ='linear')
        logit_shp = logit.shape
        probs = tensor.nnet.softmax(
            logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

        # cost
        x_flat = x.flatten()
        x_flat_idx = tensor.arange(x_flat.shape[0]) * options['n_words'] + \
            x_flat
        cost = -tensor.log(probs.flatten()[x_flat_idx])
        cost = cost.reshape([x.shape[0], x.shape[1]])
        opt_ret['cost_per_sample'] = cost
        cost = (cost * x_mask).sum(0)

        return use_noise, x, x_mask, opt_ret, cost

    def build_sampler(self, trng=None):

        if trng is None:
            trng = RandomStreams(1234)
        if self.tparams is None:
            self.init_tparams()

        tparams = self.tparams
        options = self.options

        # x: 1 x 1
        y = tensor.vector('y_sampler', dtype='int64')
        init_state = tensor.matrix('init_state', dtype='float32')

        # if it's the first word, emb should be all zero
        emb = tensor.switch(y[:, None] < 0,
                            tensor.alloc(0., 1, tparams['Wemb'].shape[1]),
                            tparams['Wemb'][y])

        # apply one step of gru layer
        proj = get_layer(options['encoder'])[1](tparams, emb,
                                                prefix='encoder',
                                                mask=None,
                                                one_step=True,
                                                init_state=init_state)
        next_state = proj[0]

        # compute the output probability dist and sample
        logit_lstm = get_layer('ff')[1](tparams, next_state, options,
                                        prefix='ff_logit_lstm', activ='linear')
        logit_prev = get_layer('ff')[1](tparams, emb, options,
                                        prefix='ff_logit_prev', activ='linear')
        logit = tensor.tanh(logit_lstm+logit_prev)
        logit = get_layer('ff')[1](tparams, logit, options,
                                   prefix='ff_logit', activ='linear')
        next_probs = tensor.nnet.softmax(logit)
        next_sample = trng.multinomial(pvals=next_probs).argmax(1)

        # next word probability
        print 'Building f_next..',
        inps = [y, init_state]
        outs = [next_probs, next_sample, next_state]
        f_next = theano.function(inps, outs, name='f_next')
        print 'Done'

        return f_next

    def gen_sample(self, tparams=None, f_next=None, trng=None, maxlen=None,
                   argmax=False):
        if tparams is None:
            if self.tparams is None:
                self.init_tparams()
            tparams = self.tparams
        if trng is None:
            trng = RandomStreams(1234)
        if f_next is None:
            f_next = self.build_sampler(trng)
        if maxlen is None:
            maxlen = 30

        options = self.options
        sample = []
        sample_score = 0

        # initial token is indicated by a -1 and initial state is zero
        next_w = -1 * numpy.ones((1,)).astype('int64')
        next_state = numpy.zeros((1, options['dim'])).astype('float32')

        for ii in xrange(maxlen):
            inps = [next_w, next_state]
            ret = f_next(*inps)
            next_p, next_w, next_state = ret[0], ret[1], ret[2]

            if argmax:
                nw = next_p[0].argmax()
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score += next_p[0, nw]
            if nw == 0:
                break

        return sample, sample_score

    def pred_probs(self, stream, f_log_probs, prepare_data, verbose=True):

        options = self.options
        probs = []
        n_done = 0

        for x in stream:
            n_done += len(x)

            x, x_mask = prepare_data(x, n_words=options['n_words'])

            pprobs = f_log_probs(x, x_mask)
            for pp in pprobs:
                probs.append(pp)

            if numpy.isnan(numpy.mean(probs)):
                ipdb.set_trace()

            if verbose:
                print >>sys.stderr, '%d samples computed' % (n_done)

        return numpy.array(probs)
