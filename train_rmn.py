import logging
import pprint

from train import train

logger = logging.getLogger(__name__)


def main(job_id, params):
    logger.info("Model options:\n{}".format(pprint.pformat(params)))
    validerr = train(**params)
    return validerr

if __name__ == '__main__':
    main(0, {
        'saveto': ['~/models/model_rmn.npz'],
        'dim_word': 128,
        'dim': 128,
        'vocab_dim': 1000,
        'memory_dim': 128,
        'memory_size': 10,
        'n_words': 1000,
        'encoder': 'gru_rmn',
        'optimizer': 'adadelta',
        'decay_c': 0.,
        'use_dropout': False,
        'lrate': 0.0001,
        'reload_': False,
        'maxlen': 30,
        'batch_size': 32,
        'valid_batch_size': 16,
        'validFreq': 5000,
        'dispFreq': 10,
        'saveFreq': 1000,
        'sampleFreq': 1000,
        'dataset': '/home/orf/data/europarl-v7.fr-en.en.tok',
        'valid_dataset': '/home/orf/data/newstest2011.en.tok',
        'dictionary': '/home/orf/data/europarl-v7.fr-en.en.tok.pkl',
        })
