import argparse
import h5py
import os
import threading
import tqdm
import multiprocessing as mp
import numpy as np
from typing import Set, Dict, Any
from taxi_env.dsl import TaxiProgramGenerator, str2int_seq, parse, INT2TOKEN
from taxi_env.taxi_env import TaxiEnv, ACTION_LIST
from utils.mp import run_parallel

PARALLEL = os.cpu_count()


class _DataThread(threading.Thread):

    def __init__(self, data_queue: mp.Queue, dir_name: str, total: int):
        super().__init__()
        self.total = total
        self.seen_programs: Set[str] = set()
        self.data_queue = data_queue

        # output files
        self.f = h5py.File(os.path.join(dir_name, 'data.hdf5'), 'w')
        self.id_file = open(os.path.join(dir_name, 'id.txt'), 'w')

    def run(self) -> None:
        max_demo_length_in_dataset = -1
        max_program_length_in_dataset = -1

        # progress bar
        bar = tqdm.tqdm(total=self.total)

        while True:
            data = self.data_queue.get()
            if data is None:
                break  # terminate if no more data

            _id: str
            grp_data: Dict[str, Any]
            _id, grp_data = data

            if 'program' not in grp_data:
                grp = self.f.create_group(_id)
                grp.update(grp_data)
                continue

            prog = str(grp_data['program'])
            if prog in self.seen_programs or len(self.seen_programs) >= self.total:
                continue

            max_demo_length_in_dataset = max(max_demo_length_in_dataset, np.max(grp_data['s_h_len']))
            max_program_length_in_dataset = max(max_program_length_in_dataset, grp_data['program'].shape[0])
            _id = f'no_{len(self.seen_programs)}' + _id
            self.seen_programs.add(prog)

            bar.update()

            self.id_file.write(_id + '\n')
            grp = self.f.create_group(_id)
            grp.update(grp_data)

        grp = self.f['data_info']
        grp.update({'max_demo_length': max_demo_length_in_dataset,
                    'max_program_length': max_program_length_in_dataset})

        print('Data processor finished')
        self.f.close()
        self.id_file.close()
        bar.close()


def _gen_proc(data_queue: mp.Queue, seed: int, num_total: int, config: argparse.Namespace):
    prog_gen = TaxiProgramGenerator(max_depth=config.max_program_depth,
                                    max_length=config.max_program_length,
                                    seed=seed)
    taxi_env = TaxiEnv(seed, max_steps=config.max_demo_length - 1)  # -1 to include initial state
    seen_programs = set()

    for _ in range(num_total):
        # generate a single random program
        program = prog_gen.random_code()

        # skip seen programs
        if program in seen_programs:
            continue
        seen_programs.add(program)
        program_seq = np.array(str2int_seq(program), dtype=np.int8)
        if program_seq.shape[0] > config.max_program_length:
            continue  # may have exceeded length

        exe = parse(program)
        if exe is None:
            print(f'Compile failure for program: {program}')
            raise RuntimeError('Program compile failure should not happen')

        num_demo = 0
        num_trial = 0
        s_h_list = []
        a_h_list = []
        p_v_h_list = []
        total_demos = config.num_demo_per_program + config.num_test_demo_per_program
        while num_demo < total_demos and num_trial < config.max_demo_generation_trial:
            try:
                taxi_env.init_game()
                done = False
                while not done:
                    action = exe(taxi_env, 0)
                    if action is None:
                        num_trial += 1
                        continue

                    done = taxi_env.state_transition(action)
            except RuntimeError:
                pass
            else:
                if config.max_demo_length >= len(taxi_env.s_h) >= config.min_demo_length:
                    s_h_list.append(np.stack(taxi_env.s_h, axis=0))  # shape: (trace_len, w, h)
                    a_h_list.append(np.array(taxi_env.a_h))  # shape: (trace_len, )
                    p_v_h_list.append(np.stack(taxi_env.p_v_h, axis=0))  # shape: (trace_len, n_percept)
                    num_demo += 1

            num_trial += 1

        if num_demo < total_demos:
            continue

        len_s_h = np.array([s_h.shape[0] for s_h in s_h_list])
        if np.max(len_s_h) < config.min_max_demo_length_for_program:
            continue

        # state shape: (n_demos, trace_len, w, h)
        h = s_h_list[0].shape[-1]
        w = s_h_list[0].shape[-2]
        demos_s_h = np.zeros([num_demo, np.max(len_s_h), w, h], dtype=np.uint8)
        for i, s_h in enumerate(s_h_list):
            demos_s_h[i, :s_h.shape[0]] = s_h

        len_a_h = np.array([a_h.shape[0] for a_h in a_h_list])

        # action shape: (n_demos, trace_len)
        demos_a_h = np.zeros([num_demo, np.max(len_a_h)], dtype=np.uint8)
        for i, a_h in enumerate(a_h_list):
            demos_a_h[i, :a_h.shape[0]] = a_h

        # percepts shape: (n_demos, trace_len, n_percept)
        demos_p_v_h = np.zeros([num_demo, np.max(len_s_h), taxi_env.num_percepts], dtype=bool)
        for i, p_v_h in enumerate(p_v_h_list):
            demos_p_v_h[i, :p_v_h.shape[0]] = p_v_h

        # save the state
        _id = f'prog_len_{str(program_seq)}_max_s_h_len_{np.max(len_s_h)}'
        grp = {'program': program_seq,
               's_h_len': len_s_h[:config.num_demo_per_program],
               's_h': demos_s_h[:config.num_demo_per_program],
               'a_h_len': len_a_h[:config.num_demo_per_program],
               'a_h': demos_a_h[:config.num_demo_per_program],
               'p_v_h': demos_p_v_h[:config.num_demo_per_program],
               'test_s_h_len': len_s_h[config.num_demo_per_program:],
               'test_s_h': demos_s_h[config.num_demo_per_program:],
               'test_a_h_len': len_a_h[config.num_demo_per_program:],
               'test_a_h': demos_a_h[config.num_demo_per_program:],
               'test_p_v_h': demos_p_v_h[config.num_demo_per_program:],
               }
        data_queue.put((_id, grp))


def generator(config):
    dir_name = config.dir_name
    num_train = config.num_train
    num_test = config.num_test
    num_val = config.num_val
    num_total = num_train + num_test + num_val

    # create and start data processor
    data_queue = mp.Manager().Queue()
    data_processor = _DataThread(data_queue, config.dir_name, num_total)
    data_processor.start()

    while len(data_processor.seen_programs) < num_total:
        missing = num_total - len(data_processor.seen_programs)
        num_gen = max(1, int(missing / PARALLEL))
        parallel = min(PARALLEL, missing)
        # print(f'Num unique progs: {len(data_processor.seen_programs)}')
        # print(f'Starting new batch of {parallel} parallel generators, each aiming for {num_gen} programs...')
        args = [(data_queue, config.seed + i, num_gen, config) for i in range(parallel)]
        run_parallel(_gen_proc, args, processes=parallel, use_tqdm=False)
        config.seed += parallel

    grp = {'num_program_tokens': len(INT2TOKEN),
           'num_action_tokens': len(ACTION_LIST),
           'num_demo_per_program': config.num_demo_per_program,
           'num_test_demo_per_program': config.num_test_demo_per_program,
           'num_train': config.num_train,
           'num_test': config.num_test,
           'num_val': config.num_val
           }

    data_queue.put(('data_info', grp))
    data_queue.put(None)
    data_processor.join()

    print(f'Dataset generated under {dir_name} with {num_total} samples ({num_train} '
          f'for training and {num_test} for testing and {num_val} for val')


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir_name', type=str, default='taxi_dataset')
    parser.add_argument('--num_train', type=int, default=25000, help='num train')
    parser.add_argument('--num_test', type=int, default=5000, help='num test')
    parser.add_argument('--num_val', type=int, default=5000, help='num val')
    parser.add_argument('--seed', type=int, default=123, help='seed')
    parser.add_argument('--max_program_length', type=int, default=50)
    parser.add_argument('--max_program_depth', type=int, default=6)
    parser.add_argument('--min_max_demo_length_for_program', type=int, default=2)
    parser.add_argument('--min_demo_length', type=int, default=8,
                        help='min demo length')
    parser.add_argument('--max_demo_length', type=int, default=20,
                        help='max demo length')
    parser.add_argument('--num_demo_per_program', type=int, default=10,
                        help='number of seen demonstrations')
    parser.add_argument('--num_test_demo_per_program', type=int, default=5,
                        help='number of unseen demonstrations')
    parser.add_argument('--max_demo_generation_trial', type=int, default=100)
    args = parser.parse_args()

    args.dir_name = os.path.join('datasets/', args.dir_name)
    check_path('datasets')
    check_path(args.dir_name)

    generator(args)


if __name__ == '__main__':
    main()
