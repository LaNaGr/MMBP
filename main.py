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

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--problem', type=str, default='case_studyB', help='problem name')
parser.add_argument('--num_iter', type=int, default=25, help='number of iterations')
parser.add_argument('--drl_batch_size', type=int, default=20, help='batch size of DRL model')
parser.add_argument('--seed', type=int, default=100, help='random seed')
parser.add_argument('--drl_cuda', type=int, default=1, help='cuda device')
parser.add_argument('--r_m', type=float, default=0.1, help='machine maintenance ratio')
parser.add_argument('--r_t', type=float, default=0.5, help='time maintenance ratio')
args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(1)


def random_main_test(problem='case_studyB', num_iter=args.num_iter, drl_batch_size=args.drl_batch_size, seed=args.seed, drl_cuda=args.drl_cuda):
    setup_seed(seed)
    print(problem)
    MODEL_CKPT = 'DRL/model/model_v1.pt'
    device = torch.device("cuda:"+str(drl_cuda) if torch.cuda.is_available() else "cpu")
    # torch.set_default_device(device)  # device
    ############################################## Environment ##############################################
    default_path = problem + '.fjs'
    rnd = './' + problem + '.csv'
    env = MMBPEnv(device=device, batch=drl_batch_size, ins_file=default_path, release_and_due=rnd,  # relation_stage_unit=rsu,
                  time_slot=1, render_mode='p_d')
    # time_slot=1, means operation start time before (Maintenance_start_time + 1) would not been influenced by maintenance

    env_heu = MMBPEnv_Heu(path=default_path, batch=1,
                      device='cpu', render_mode='p_d')
    from MMBP.Heuristics import Heuristic
    rule_list = ['MOPNR_EET', 'MOPNR_SPT', 'MWRM_EET', 'MWRM_SPT', 'LWRM_EET', 'LWRM_SPT', 'FIFO_EET', 'FIFO_SPT']

    ############################################## Loading Model ##############################################
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    gc.collect()
    torch.cuda.empty_cache()

    print("HGNN-DRL Method loading ...")

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
    from DRL.random_maintenance import random_maintenance as rm

    record = []
    with torch.no_grad():

        for ins in range(1, num_iter+1):
            '''Random maintenance and test DRL model and Heuristic model'''
            # Random maintenance
            CT = env_heu.schedule.makespan_batch[0] + 20
            MACHINES = env_heu.instance.num_mas
            maintenance = rm(CT=CT, MACHINES=MACHINES, ratio_machine=args.r_m, ratio_time=args.r_t)
            print(maintenance)
            gc.collect()
            torch.cuda.empty_cache()

            # DRL model
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=64'
            env.reset()
            env.update_main_info(maintenance)
            state = env.state
            dones = env.schedule.done_batch
            done = False  # Unfinished at the beginning
            i = 0
            last_time = time.time()
            while ~done:
                i += 1
                action = model.policy_old.act(state, memories, dones, flag_train=False)
                state, rewards, dones, _, _ = env.step(action)
                done = dones.all()
            spend_time = time.time() - last_time
            result_correct, schedule_batch = env.validate_gantt()

            if result_correct.any():
                min_makespan, idx_result_correct = env.schedule.makespan_batch[result_correct].min(0)
                idx = torch.range(0, env.instance.batch_size - 1)[result_correct][idx_result_correct]
                print(idx)

                print("spend_time: %.2f" % spend_time, "result correct:", result_correct.any(),
                      "makespan_min:", min_makespan)
                #env.render(name=['RL_m', ins], seleced_batch=idx)
            else:
                print("spend_time: %.2f" % spend_time, "result correct:", result_correct,
                      "makespan:", env.schedule.makespan_batch[result_correct])
                continue

            record.append(['DRL', env.schedule.makespan_batch[result_correct].min().cpu().item(), spend_time])

            # Heuristic model
            env_heu.update_main_info(maintenance)

            for rule in rule_list:
                method = Heuristic(rule)
                # print(rule, " start")
                done = False  # Unfinished at the beginning
                i = 0
                t_start = time.time()
                while ~done:
                    i += 1
                    action = method.go(env_heu)
                    time_move = env_heu.feature.feat_mas_batch[0, 1, action[1, 0]]
                    if time_move >= env_heu.schedule.time:
                        env.time_move_to_t(torch.tensor([time_move]))
                    state, rewards, dones, _, _ = env_heu.step(action)
                    done = dones.all()
                t_end = time.time()
                # env_heu.render(name=[rule+'_m', ins])
                duration = t_end - t_start
                validation = env_heu.validate_gantt()
                makespan = env_heu.schedule.makespan_batch.numpy()[0]
                env_heu.reset()
                print([rule, makespan, duration], 'validation:', validation[0])
                record.append([rule, makespan, duration])
            df = pd.DataFrame(record, columns=['rule', 'makespan', 'duration'])
            df.to_csv(f'{problem[-1]}_m_{args.seed}_{int(args.r_m*100)}_{int(args.r_t*100)}.csv', index=False)
        print('Finish')


if __name__ == "__main__":
    #test_MMBPEnv_Heu(problem='case_studyA')
    #test_MMBPEnv_random(problem='case_studyA')
    #test_MMBPEnv_DRL(problem='case_studyA')
    print(args)
    random_main_test(problem=args.problem, num_iter=args.num_iter, drl_batch_size=args.drl_batch_size, seed=args.seed)