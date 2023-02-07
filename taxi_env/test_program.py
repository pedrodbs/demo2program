import argparse
import io
import json
import os.path

import h5py
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from h5py import Group

from evaler import Evaler
from taxi_env import dataset_taxi
from taxi_env._tests.programs import MOVE_DEST_PROG
from taxi_env.dsl import str2int_seq, parse
from taxi_env.input_ops_taxi import create_input_ops
from taxi_env.taxi_env import TaxiEnv

TRAIN_DIR = 'train_dir/taxi-datasets_taxi_dataset-full-default-bs_32_lr_0.001_lstm_cell_512_k_10-20221007-203625'

SEED = 17
PROGRAM = MOVE_DEST_PROG


def main():
    with open(os.path.join(TRAIN_DIR, 'config.json'), 'r') as fp:
        config = argparse.Namespace(**json.load(fp))

    config.num_demo_per_program = config.num_k
    config.num_test_demo_per_program = 5
    config.min_max_demo_length_for_program = 2
    config.min_demo_length = 8
    config.max_demo_length = config.max_demo_len
    config.max_program_length = config.max_program_len

    config.dataset_split = 'test'
    config.train_dir = TRAIN_DIR
    config.checkpoint = ''
    config.max_steps = 1
    config.pred_program = True
    config.no_loss = False
    config.output_dir = 'eval_dir/taxi'
    config.result_data = True
    config.result_data_path = os.path.join(config.output_dir, 'result.hdf5')
    config.quiet = False
    config.write_summary = True
    config.summary_file = os.path.join(config.output_dir, 'report.txt')

    _, dataset, _ = dataset_taxi.create_default_splits(config.dataset_path, is_train=False, num_k=config.num_k)
    _id, demo_data = _gen_prog_demos(PROGRAM, config, SEED)
    data = h5py.File(io.BytesIO())
    grp = data.create_group(_id)
    grp.update(demo_data)
    dataset._ids = [_id]  # just this program
    dataset.data = data

    # config.id_list = _id

    evaler = Evaler(config, dataset)
    evaler.eval_run()

    # _, batch = create_input_ops(dataset, 1, is_training=False, shuffle=False)
    #
    # # --- create model ---
    # model = Evaler.get_model_class(config.model)
    # model = model(config, is_train=False)
    #
    # global_step = tf.contrib.framework.get_or_create_global_step(
    #     graph=None)
    # step_op = tf.no_op(name='step_no_op')
    #
    # # --- vars ---
    # all_vars = tf.trainable_variables()
    # slim.model_analyzer.analyze_vars(all_vars, print_info=True)
    #
    # tf.set_random_seed(123)
    #
    # session_config = tf.ConfigProto(
    #     allow_soft_placement=True,
    #     gpu_options=tf.GPUOptions(allow_growth=True),
    #     device_count={'GPU': 1},
    # )
    # session = tf.Session(config=session_config)
    #
    # # --- checkpoint and monitoring ---
    # saver = tf.train.Saver(max_to_keep=100)
    #
    # checkpoint = config.checkpoint
    # if checkpoint is None and TRAIN_DIR:
    #     checkpoint = tf.train.latest_checkpoint(TRAIN_DIR)
    #
    # # load checkpoint
    # if checkpoint:
    #     saver.restore(session, checkpoint)
    #
    # tuple_data = run_single_step(model, batch, session, global_step, step_op)
    # print(tuple_data)


def _gen_prog_demos(program, config, seed):
    taxi_env = TaxiEnv(seed, max_steps=config.max_demo_length - 1)  # -1 to include initial state

    program_seq = np.array(str2int_seq(program), dtype=np.int8)
    if program_seq.shape[0] > config.max_program_length:
        raise ValueError(f'Program {program} exceeds length')

    exe = parse(program)
    if exe is None:
        print(f'Compile failure for program: {program}')
        raise RuntimeError('Program compile failure should not happen')

    num_demo = 0
    s_h_list = []
    a_h_list = []
    p_v_h_list = []
    total_demos = config.num_demo_per_program + config.num_test_demo_per_program
    while num_demo < total_demos:
        try:
            taxi_env.init_game()
            done = False
            while not done:
                action = exe(taxi_env, 0)
                if action is None:
                    continue

                done = taxi_env.state_transition(action)
        except RuntimeError:
            pass
        else:
            if config.max_demo_length >= len(taxi_env.s_h) >= config.min_demo_length:
                s_h_list.append(np.stack(taxi_env.s_h, axis=0))  # shape: (trace_len, w, h, c)
                a_h_list.append(np.array(taxi_env.a_h, dtype=np.uint8))  # shape: (trace_len, )
                p_v_h_list.append(np.stack(taxi_env.p_v_h, axis=0))  # shape: (trace_len, n_percept)
                num_demo += 1

    if num_demo < total_demos:
        raise RuntimeError(f'Required {total_demos}, only {num_demo} produced')

    len_s_h = np.array([s_h.shape[0] for s_h in s_h_list])
    if np.max(len_s_h) < config.min_max_demo_length_for_program:
        raise RuntimeError(
            f'Demo length {np.max(len_s_h)} lower than required: {config.min_max_demo_length_for_program}')

    # state shape: (n_demos, trace_len, w, h, c)
    w = s_h_list[0].shape[-3]
    h = s_h_list[0].shape[-2]
    c = s_h_list[0].shape[-1]
    demos_s_h = np.zeros((num_demo, np.max(len_s_h), w, h, c), dtype=s_h_list[0].dtype)
    for i, s_h in enumerate(s_h_list):
        demos_s_h[i, :s_h.shape[0]] = s_h

    len_a_h = np.array([a_h.shape[0] for a_h in a_h_list])

    # action shape: (n_demos, trace_len)
    demos_a_h = np.zeros([num_demo, np.max(len_a_h)], dtype=a_h_list[0].dtype)
    for i, a_h in enumerate(a_h_list):
        demos_a_h[i, :a_h.shape[0]] = a_h

    # percepts shape: (n_demos, trace_len, n_percept)
    demos_p_v_h = np.zeros([num_demo, np.max(len_s_h), taxi_env.num_percepts], dtype=p_v_h_list[0].dtype)
    for i, p_v_h in enumerate(p_v_h_list):
        demos_p_v_h[i, :p_v_h.shape[0]] = p_v_h

    # save the state
    _id = f'prog_{"_".join([str(e) for e in program_seq])}_max_s_h_len_{np.max(len_s_h)}'
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
    return _id, grp


def run_single_step(model, batch, session, global_step, step_op):
    batch_chunk = session.run(batch)

    [step, loss, acc, hist,
     pred_program, pred_program_len, pred_is_correct_syntax,
     greedy_pred_program, greedy_program_len, greedy_is_correct_syntax,
     gt_program, gt_program_len,
     program_num_execution_correct, program_is_correct_execution,
     greedy_num_execution_correct, greedy_is_correct_execution, output, _] = \
        session.run(
            [global_step, model.report_loss,
             model.report_accuracy,
             model.report_hist,
             model.pred_program, model.program_len,
             model.program_is_correct_syntax,
             model.greedy_pred_program, model.greedy_pred_program_len,
             model.greedy_program_is_correct_syntax,
             model.ground_truth_program, model.program_len,
             model.program_num_execution_correct,
             model.program_is_correct_execution,
             model.greedy_num_execution_correct,
             model.greedy_is_correct_execution,
             model.output,
             step_op],
            feed_dict=model.get_feed_dict(batch_chunk)
        )

    return (step, loss, acc, hist,
            pred_program, pred_program_len, pred_is_correct_syntax,
            greedy_pred_program, greedy_program_len, greedy_is_correct_syntax,
            gt_program, gt_program_len, output, batch_chunk['id'],
            program_num_execution_correct, program_is_correct_execution,
            greedy_num_execution_correct, greedy_is_correct_execution)


if __name__ == '__main__':
    main()
