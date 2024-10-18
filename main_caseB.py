import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接 module 文件夹的路径，并将其添加到 sys.path
module_path = os.path.join(current_dir, 'DRL')
sys.path.append(module_path)

import gc
from MMBP_v1_1 import MMBPEnv
from MMBP_v0_2_heuristic_only import MMBPEnv_Heu
import torch
import time
import pandas as pd
import json
from DRL.PPO_model import PPO, Memory
import random
import numpy as np
import argparse
from main import setup_seed


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--problem', type=str, default='case_studyB', help='problem name')
parser.add_argument('--drl_batch_size', type=int, default=100, help='batch size of DRL model')
parser.add_argument('--seed', type=int, default=100, help='random seed')
parser.add_argument('--drl_cuda', type=int, default=0, help='cuda device')

args = parser.parse_args()
setup_seed(args.seed)


def test_MMBPEnv_Heu(problem='case_studyB'):
    """Example of test MMBPEnv_Heu"""
    print("Example: MMBPEnv_Heu")
    # torch.manual_seed(20199650253898)
    device = "cpu"
    default_path = './' + problem + '.fjs'
    env = MMBPEnv_Heu(path=default_path, batch=1, render_mode='p_d',
                      device=device, maintenance=[[10, 4, 6], [3, 5, 25], [8, 20, 45]])     # caseA [[10,4,6]]
    from MMBP.Heuristics import Heuristic
    rule_list = ['MOPNR_EET', 'MOPNR_SPT', 'MWRM_EET', 'MWRM_SPT', 'LWRM_EET', 'LWRM_SPT', 'FIFO_EET', 'FIFO_SPT']
    record = []
    for rule in rule_list:
        method = Heuristic(rule)
        # print(rule, " start")
        done = False  # Unfinished at the beginning
        i = 0
        t_start = time.time()
        while ~done:
            i += 1
            action = method.go(env)
            time_move = env.feature.feat_mas_batch[0, 1, action[1, 0]]
            if time_move >= env.schedule.time:
                env.time_move_to_t(torch.tensor([time_move]))
            state, rewards, dones, _, _ = env.step(action)
            done = dones.all()
        env.render(name=[rule, 't'])
        t_end = time.time()
        duration = t_end - t_start
        validation = env.validate_gantt()
        makespan = env.schedule.makespan_batch.numpy()[0]
        env.reset()
        print([rule, makespan, duration], 'validation:', validation[0])
        record.append([rule, makespan, duration])
    df = pd.DataFrame(record, columns=['rule', 'makespan', 'duration'])
    df.to_csv(f'Heu_result_dynamic_{problem[-1]}.csv')


def test_MMBPEnv_DRL(problem='case_studyB', batch=args.drl_batch_size):
    """Example of test MMBPEnv
    This method can adjust with Heu and RL method."""
    MODEL_CKPT = 'DRL/model/model_v1.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # torch.set_default_device(device)  # device
    ############################################## Environment ##############################################
    print("Example: MMBPEnv")
    default_path = problem + '.fjs'
    rnd = './' + problem + '.csv'
    env = MMBPEnv(device=device, batch=batch, ins_file=default_path, release_and_due=rnd,  # relation_stage_unit=rsu,
                  render_mode='p_d', time_slot=1,
                  maintenance=[[10, 4, 6], [3, 5, 25], [8, 20, 45]])

    ############################################## Loading Model ##############################################
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    gc.collect()
    torch.cuda.empty_cache()


    print("HGNN-DRL Model loading ...")

    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device for DRL: ", device)
    with open("./DRL/config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    train_paras = load_dict["train_paras"]
    model_paras = load_dict["model_paras"]
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]
    model_paras["device"] = device

    memories = Memory()
    model = PPO(model_paras, train_paras)

    # Load checkpoint
    if device.type == 'cuda':
        model_CKPT = torch.load(MODEL_CKPT, map_location='cuda:0')
    else:
        model_CKPT = torch.load(MODEL_CKPT, map_location='cpu')

    model.policy.load_state_dict(model_CKPT)
    model.policy_old.load_state_dict(model_CKPT)

    ############################################## Testing ##############################################

    state = env.state
    dones = env.schedule.done_batch
    done = False  # Unfinished at the beginning
    i = 0
    last_time = time.time()
    while ~done:
        i += 1
        with torch.no_grad():
            action = model.policy_old.act(state, memories, dones, flag_train=False)
            state, rewards, dones, _, _ = env.step(action)
            done = dones.all()
    spend_time = time.time() - last_time
    result_correct, schedule_batch = env.validate_gantt()
    min_makespan, idx_result_correct = env.schedule.makespan_batch[result_correct].min(0)  # not this index, but the index of result correct
    idx = torch.range(0,env.instance.batch_size-1)[result_correct][idx_result_correct].int()
    env.render(name=['RL','v'], selected_batch=idx)
    print("spend_time: %.2f" % spend_time, "result correct:", result_correct.any(), "makespan:", min_makespan.item())


def test_MMBPEnv_GA(problem='case_studyB'):
    """Example of test MMBPEnv_Heu"""
    print("Example: MMBPEnv_GA")
    # torch.manual_seed(20199650253898)
    device = "cpu"
    default_path = './' + problem + '.fjs'
    env = MMBPEnv(path=default_path, batch=1, render_mode='p_d',
                  device=device, maintenance=[[10, 4, 6], [3, 5, 25], [8, 20, 45]])

    # Genetic Algorithm (GA) method - a sequence of actions
    method = GA
    actions = GA.go(default_path)    # shape: (steps, batch_size, 3)

    done = False  # Unfinished at the beginning
    i = 0
    t_start = time.time()
    while ~done:
        action = actions[i]     # e.g. [[Opes],[Mas],[Jobs]], shape: (batch_size, 3)
        state, rewards, dones, _, _ = env.step(action)
        done = dones.all()
        i += 1
    env.render(name=['GA', 't'])
    t_end = time.time()
    duration = t_end - t_start
    validation = env.validate_gantt()
    makespan = env.schedule.makespan_batch.numpy()[0]
    env.reset()
    print('GA', [makespan, duration], 'validation:', validation[0])


class GA:
    def __init__(self):
        pass

    @staticmethod
    def go(env):
        """Genetic Algorithm
        :param env: MMBPEnv_Heu
        :return: action
        """
        pass


#test_MMBPEnv_Heu(args.problem)
#test_MMBPEnv_DRL(args.problem)
test_MMBPEnv_GA(args.problem)