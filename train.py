'''
Build a neural language model using GRU-RMN units
'''
import theano
import theano.tensor as tensor

import cPickle as pkl
import importlib
import ipdb
import logging
import numpy
import copy

import os
import time

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from data_iterator import TextIterator, prepare_data
from rmn import RMN
from utils import (zipp, unzip, itemlist)


logger = logging.getLogger(__name__)
profile = False


def train(dim_word=100,  # word vector dimensionality
          dim=1000,  # the number of GRU units
          encoder='gru',
          patience=10,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 weight decay penalty
          lrate=0.01,
          n_words=100000,  # vocabulary size
          vocab_dim=100000,  # Size of M, C
          memory_dim=1000,  # Dimension of memory
          memory_size=15,  # n_back to attend
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000,  # save the parameters after every saveFreq updates
          sampleFreq=100,  # generate some samples after every sampleFreq
          dataset='/data/lisatmp3/chokyun/wikipedia/extracted/wiki.tok.txt.gz',
          valid_dataset='../data/dev/newstest2011.en.tok',
          dictionary='/data/lisatmp3/chokyun/wikipedia/extracted/'
          'wiki.tok.txt.gz.pkl',
          use_dropout=False,
          reload_=False):

    # Model options
    model_options = locals().copy()

    # Theano random stream
    trng = RandomStreams(1234)

    # load dictionary
    with open(dictionary, 'rb') as f:
        worddicts = pkl.load(f)

    # invert dictionary
    worddicts_r = dict()
    for kk, vv in worddicts.iteritems():
        worddicts_r[vv] = kk

    # reload options
    if reload_ and os.path.exists(saveto):
        with open('%s.pkl' % saveto, 'rb') as f:
            model_options = pkl.load(f)

    print 'Loading data'
    train = TextIterator(dataset,
                         dictionary,
                         n_words_source=n_words,
                         batch_size=batch_size,
                         maxlen=maxlen)
    valid = TextIterator(valid_dataset,
                         dictionary,
                         n_words_source=n_words,
                         batch_size=valid_batch_size,
                         maxlen=maxlen)

    # initialize RMN
    rmn_ = RMN(model_options)

    print 'Building model'
    rmn_.init_params()

    # reload parameters
    if reload_ and os.path.exists(saveto):
        rmn_.load_params(saveto)

    # create shared variables for parameters
    tparams = rmn_.tparams

    # build the symbolic computational graph
    use_noise, x, x_mask, opt_ret, cost = rmn_.build_model()
    inps = [x, x_mask]

    print 'Buliding sampler'
    f_next = rmn_.build_sampler(trng)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'

    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # after any regularizer - compile the computational graph for cost
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    optimizer = importlib.import_module('optimizer.' + optimizer)
    f_grad_shared, f_update = optimizer(lr, tparams, grads, inps, cost)
    print 'Done'

    print 'Optimization'

    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        history_errs = list(numpy.load(saveto)['history_errs'])
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    # Training loop
    uidx = 0
    estop = False
    bad_counter = 0
    for eidx in xrange(max_epochs):
        n_samples = 0

        for x in train:
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            # pad batch and create mask
            x, x_mask = prepare_data(x, maxlen=maxlen, n_words=n_words)

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            ud_start = time.time()

            # compute cost, grads and copy grads to shared variables
            cost = f_grad_shared(x, x_mask)

            # do the update on parameters
            f_update(lrate)

            ud = time.time() - ud_start

            # check for bad numbers
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1.

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud

            # save the best model so far
            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving...',

                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_errs, **params)
                pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'))
                print 'Done'

            # generate some samples with the model and display them
            if numpy.mod(uidx, sampleFreq) == 0:
                # FIXME: random selection?
                for jj in xrange(5):
                    sample, score = rmn_.gen_sample(tparams, f_next,
                                                    model_options, trng=trng,
                                                    maxlen=30, argmax=False)
                    print 'Sample ', jj, ': ',
                    ss = sample
                    for vv in ss:
                        if vv == 0:
                            break
                        if vv in worddicts_r:
                            print worddicts_r[vv],
                        else:
                            print 'UNK',
                    print

            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_errs = rmn_.pred_probs(f_log_probs, prepare_data,
                                             model_options, valid)
                valid_err = valid_errs.mean()
                history_errs.append(valid_err)

                if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                    best_p = unzip(tparams)
                    bad_counter = 0
                if len(history_errs) > patience and valid_err >= \
                        numpy.array(history_errs)[:-patience].min():
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        estop = True
                        break

                if numpy.isnan(valid_err):
                    ipdb.set_trace()

                print 'Valid ', valid_err

            # finish after this many updates
            if uidx >= finish_after:
                print 'Finishing after %d iterations!' % uidx
                estop = True
                break

        print 'Seen %d samples' % n_samples

        if estop:
            break

    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    valid_err = rmn_.pred_probs(f_log_probs, prepare_data,
                                model_options, valid).mean()

    print 'Valid ', valid_err

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p,
                history_errs=history_errs,
                **params)

    return valid_err
