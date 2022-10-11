import os
import logging
import h5py
import numpy as np
from colorlog import ColoredFormatter

rs = np.random.RandomState(17)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s] %(message)s",
    #    datefmt='%H:%M:%S.%f',
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'white,bold',
        'INFOV': 'cyan,bold',
        'WARNING': 'yellow',
        'ERROR': 'red,bold',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
ch.setFormatter(formatter)

log = logging.getLogger('rn')
log.setLevel(logging.DEBUG)
log.handlers = []  # No duplicated handlers
log.propagate = False  # workaround for duplicated logs in ipython
log.addHandler(ch)

logging.addLevelName(logging.INFO + 1, 'INFOV')


def _infov(self, msg, *args, **kwargs):
    self.log(logging.INFO + 1, msg, *args, **kwargs)


logging.Logger.infov = _infov


class Dataset(object):

    def __init__(self, ids, dataset_path, name='default', num_k=10, is_train=True):
        self._ids = list(ids)
        self.name = name
        self.dataset_image_path = os.path.join(dataset_path, 'images')
        self.is_train = is_train
        self.num_k = num_k

        filename = 'data.hdf5'
        file = os.path.join(dataset_path, filename)
        log.info(f'Reading {file} ...')

        self.data = h5py.File(file, 'r')
        self.num_demo = int(self.data['data_info']['num_demo_per_program'].value)
        self.max_demo_len = int(self.data['data_info']['max_demo_length'].value)
        self.max_program_len = int(self.data['data_info']['max_program_length'].value)
        self.num_program_tokens = int(self.data['data_info']['num_program_tokens'].value)
        self.num_action_tokens = int(self.data['data_info']['num_action_tokens'].value)
        self.k = int(self.data['data_info']['num_demo_per_program'].value)
        self.test_k = int(self.data['data_info']['num_test_demo_per_program'].value)
        log.info(f'Reading Done: {file}')

    def get_data(self, id, order=None):
        # preprocessing and data augmentation

        # each data point consist of a program + k demo

        # dim: [one hot dim of program tokens, program len]
        program_tokens = self.data[id]['program'].value
        program = np.zeros([self.num_program_tokens, self.max_program_len], dtype=bool)
        program[:, :len(program_tokens)][program_tokens, np.arange(len(program_tokens))] = 1
        padded_program_tokens = np.zeros([self.max_program_len], dtype=program_tokens.dtype)
        padded_program_tokens[:len(program_tokens)] = program_tokens

        # get s_h and test_s_h
        demo_data = self.data[id]['s_h'].value[:self.num_k]
        test_demo_data = self.data[id]['test_s_h'].value

        sz = demo_data.shape
        demo = np.zeros([sz[0], self.max_demo_len, sz[2], sz[3], sz[4]], dtype=demo_data.dtype)
        demo[:, :sz[1], :, :, :] = demo_data
        sz = test_demo_data.shape
        test_demo = np.zeros([sz[0], self.max_demo_len, sz[2], sz[3], sz[4]], dtype=demo_data.dtype)
        test_demo[:, :sz[1], :, :, :] = test_demo_data

        # dim: [k, action_space, max len of demo - 1]
        action_history_tokens = self.data[id]['a_h'].value[:self.num_k]
        action_history = []
        for a_h_tokens in action_history_tokens:
            # num_action_tokens + 1 is <e> token which is required for detecting
            # the end of the sequence. Even though the original length of the
            # action history is max_demo_len - 1, we make it max_demo_len, by
            # including the last <e> token.
            a_h = np.zeros([self.max_demo_len, self.num_action_tokens + 1], dtype=bool)
            a_h[:len(a_h_tokens), :][np.arange(len(a_h_tokens)), a_h_tokens] = 1
            a_h[len(a_h_tokens), self.num_action_tokens] = 1  # <e>
            action_history.append(a_h)
        action_history = np.stack(action_history, axis=0)
        padded_action_history_tokens = np.argmax(action_history, axis=2)

        # dim: [test_k, action_space, max len of demo - 1]
        test_action_history_tokens = self.data[id]['test_a_h'].value
        test_action_history = []
        for test_a_h_tokens in test_action_history_tokens:
            # num_action_tokens + 1 is <e> token which is required for detecting
            # the end of the sequence. Even though the original length of the
            # action history is max_demo_len - 1, we make it max_demo_len, by
            # including the last <e> token.
            test_a_h = np.zeros([self.max_demo_len, self.num_action_tokens + 1], dtype=bool)
            test_a_h[:len(test_a_h_tokens), :][np.arange(len(test_a_h_tokens)), test_a_h_tokens] = 1
            test_a_h[len(test_a_h_tokens), self.num_action_tokens] = 1  # <e>
            test_action_history.append(test_a_h)
        test_action_history = np.stack(test_action_history, axis=0)
        padded_test_action_history_tokens = np.argmax(test_action_history, axis=2)

        # program length: [1]
        program_length = np.array([len(program_tokens)], dtype=np.float32)

        # len of each demo. dim: [k]
        demo_length = self.data[id]['s_h_len'].value[:self.num_k]
        test_demo_length = self.data[id]['test_s_h_len'].value

        demo_percept_data = self.data[id]['p_v_h'].value[:self.num_k]
        sz = demo_percept_data.shape
        demo_percept = np.zeros([sz[0], self.max_demo_len, sz[2]],
                                dtype=demo_percept_data.dtype)
        demo_percept[:, :sz[1], :] = demo_percept_data

        test_demo_percept_data = self.data[id]['test_p_v_h'].value
        sz = test_demo_percept_data.shape
        test_demo_percept = np.zeros([sz[0], self.max_demo_len, sz[2]],
                                     dtype=test_demo_percept_data.dtype)
        test_demo_percept[:, :sz[1], :] = test_demo_percept_data

        return (program, padded_program_tokens, demo, test_demo,
                action_history, padded_action_history_tokens,
                test_action_history, padded_test_action_history_tokens,
                program_length, demo_length, test_demo_length,
                demo_percept, test_demo_percept)

    @property
    def ids(self):
        return self._ids

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return 'Dataset (%s, %d examples)' % (
            self.name,
            len(self)
        )


def create_default_splits(dataset_path, num_k=10, is_train=True):
    ids_train, ids_test, ids_val = all_ids(dataset_path)

    dataset_train = Dataset(ids_train, dataset_path, name='train',
                            num_k=num_k, is_train=is_train)
    dataset_test = Dataset(ids_test, dataset_path, name='test',
                           num_k=num_k, is_train=is_train)
    dataset_val = Dataset(ids_val, dataset_path, name='val',
                          num_k=num_k, is_train=is_train)
    return dataset_train, dataset_test, dataset_val


def all_ids(dataset_path):
    with h5py.File(os.path.join(dataset_path, 'data.hdf5'), 'r') as f:
        num_train = int(f['data_info']['num_train'].value)
        num_test = int(f['data_info']['num_test'].value)
        num_val = int(f['data_info']['num_val'].value)

    with open(os.path.join(dataset_path, 'id.txt'), 'r') as fp:
        ids_total = [s.strip() for s in fp.readlines() if s]

    ids_train = ids_total[:num_train]
    ids_test = ids_total[num_train: num_train + num_test]
    ids_val = ids_total[num_train + num_test: num_train + num_test + num_val]

    rs.shuffle(ids_train)
    rs.shuffle(ids_test)
    rs.shuffle(ids_val)

    return ids_train, ids_test, ids_val
