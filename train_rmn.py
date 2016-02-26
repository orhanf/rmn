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
        'reload_': False,
        'maxlen': 50,
        'batch_size': 32,
        'valid_batch_size': 16,
        'validFreq': 1000,
        'dispFreq': 10,
        'saveFreq': 1000,
        'sampleFreq': 10,
        'dataset': '/home/ofirat/data/europarl-v7.fr-en.en.tok',
        'valid_dataset': '/home/ofirat/data/newstest2011.en.tok',
        'dictionary': '/home/ofirat/data/europarl-v7.fr-en.en.tok.pkl',
        })
