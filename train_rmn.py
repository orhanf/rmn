import logging
import os
import pprint

from train import train

logger = logging.getLogger(__name__)

isDebug = True


def main(job_id, params):
    logger.info("Model options:\n{}".format(pprint.pformat(params)))
    validerr = train(**params)
    return validerr

if __name__ == '__main__':
    homedir = os.getenv("HOME")
    if isDebug:
        params = {
            'saveto': homedir + '/models/model_rmn_small.npz',
            'dim_word': 200,
            'dim': 126,
            'vocab_dim': 100,
            'memory_dim': 126,
            'memory_size': 10,
            'n_words': 500,
            'encoder': 'gru_rmn',
            'optimizer': 'adadelta',
            'decay_c': 0.,
            'use_dropout': False,
            'lrate': 0.0001,
            'reload_': False,
            'maxlen': 20,
            'batch_size': 12,
            'valid_batch_size': 8,
            'validFreq': 1000,
            'dispFreq': 10,
            'saveFreq': 1000,
            'sampleFreq': 50,
            'dataset': homedir + '/data/europarl-v7.fr-en.en.tok',
            'valid_dataset': homedir + '/data/newstest2011.en.tok',
            'dictionary': homedir + '/data/europarl-v7.fr-en.en.tok.pkl',
            }
    else:
        params = {
            'saveto': homedir + '/models/model_rmn.npz',
            'dim_word': 620,
            'dim': 512,
            'vocab_dim': 300,
            'memory_dim': 512,
            'memory_size': 10,
            'n_words': 30000,
            'encoder': 'gru_rmn',
            'optimizer': 'adadelta',
            'decay_c': 0.,
            'use_dropout': False,
            'lrate': 0.0001,
            'reload_': True,
            'maxlen': 50,
            'batch_size': 32,
            'valid_batch_size': 16,
            'validFreq': 1000,
            'dispFreq': 10,
            'saveFreq': 1000,
            'sampleFreq': 200,
            'dataset': homedir + '/data/europarl-v7.fr-en.en.tok',
            'valid_dataset': homedir + '/data/newstest2011.en.tok',
            'dictionary': homedir + '/data/europarl-v7.fr-en.en.tok.pkl',
            }
    main(0, params)
