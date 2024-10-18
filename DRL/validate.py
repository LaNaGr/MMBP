import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import PPO_model
import torch
import time
import os
import copy
from MMBP_env_old_ver import MMBPEnv
import random
import pandas as pd



def get_validate_env(device):
    '''
    Generate and return the validation environment from the validation set ()
    '''
    folders = os.listdir('./data_dev/')
    folders.sort()
    for x in folders:
        if x.endswith('.csv'):
            folders.remove(x)
        else:
            pass
    envs = []
    for fold in folders:
        file_path = "./data_dev/" + fold + "/"
        rnd = "./data_dev/"+fold+".csv"
        valid_data_files = os.listdir(file_path)
        for i in range(len(valid_data_files)):
            valid_data_files[i] = file_path+valid_data_files[i]
        env = MMBPEnv(ins_file=valid_data_files, device=device, render_mode=None, release_and_due=rnd, time_slot=0.1)
        envs.append(env)
        del env
    return envs

def validate(envs, model_policy):
    '''
    Validate the policy during training, and the process is similar to test
    '''
    makespan, makespan_batch = 0, []
    start = time.time()
    # batch_size = env_paras["batch_size"]
    memory = PPO_model.Memory()
    # print('There are {0} dev instances.'.format(batch_size))  # validation set is also called development set
    print('makespan:', end='\t')
    for env in envs:
        env.reset()
        state = env.state
        done = False
        dones = env.schedule.done_batch
        while ~done:
            with torch.no_grad():
                actions = model_policy.act(state, memory, dones, flag_sample=False, flag_train=False)
                # 这一步改变了proc_time, 为什么？
                #####################################################################################################BUG
            state, rewards, dones,_,_ = env.step(actions)
            done = dones.all()
        gantt_result = env.validate_gantt()[0]
        if not gantt_result:
            print("Scheduling Error！！！！！！")
        makespan_new = copy.deepcopy(env.schedule.makespan_batch.mean()).item()
        makespan += makespan_new
        makespan_batch.append(makespan_new)
        print(makespan_new, end='\t')
    print('validating time: ', time.time() - start, end='\t')
    makespan_batch = torch.tensor(makespan_batch)
    return makespan, makespan_batch


def get_test_envs(device, path='./data_test/'):
    '''
    Generate and return the test environment from the validation set ()
    '''
    folders = os.listdir(path)
    folders.sort()
    files = []
    for x in folders:
        if x.endswith('.csv'):
            files.append(x[:-4])
            folders.remove(x)
        else:
            pass
    envs = []
    for file in folders:
        file_path = "./data_test/" + file
        rnd = "./data_test/" + file[:-4] + ".csv"
        env = MMBPEnv(ins_file=file_path, batch=1, device=device, render_mode=None, release_and_due=rnd, time_slot=0.1)
        envs.append(env)
        del env
    print(folders)
    return envs, files


def test(envs, model_policy, TEST=[251,421,594,235,423,610,204,335,478,20.106,28.533,44.689,53.933,974,1878,492,914],
         flag_sample=True):
    '''
    Test the policy during training, and the process is similar to test
    default: sample method, flag-sample=True, so that the policy will sample actions and give different results
    '''
    makespan, makespan_batch, time_batch = 0, [], []
    start = time.time()
    # batch_size = env_paras["batch_size"]
    memory = PPO_model.Memory()
    # print('There are {0} dev instances.'.format(batch_size))  # validation set is also called development set
    print('makespan:', end='\t')
    for env in envs:
        timestep1 = time.time()
        env.reset()
        state = env.state
        done = False
        dones = env.schedule.done_batch
        while ~done:
            with torch.no_grad():
                actions = model_policy.act(state, memory, dones, flag_sample=flag_sample, flag_train=False)
            state, rewards, dones,_,_ = env.step(actions)
            done = dones.all()
        timestep2 = time.time()
        duration = timestep2 - timestep1
        gantt_result = env.validate_gantt()[0]
        if not gantt_result:
            print("Scheduling Error！！！！！！")
        makespan_new = copy.deepcopy(env.schedule.makespan_batch.mean()).item()
        makespan_new = round(makespan_new, 2)
        makespan += makespan_new
        makespan_batch.append(makespan_new)
        time_batch.append(duration)
        print(makespan_new, end='\t')
    print('validating time: ', time.time() - start, end='\t')
    np_duizhao = torch.tensor(TEST)
    makespan_batch_tensor = torch.tensor(makespan_batch)
    gap = (makespan_batch_tensor - np_duizhao) / np_duizhao
    gap = gap.cpu().numpy()
    makespan_batch = makespan_batch_tensor.cpu().numpy()
    return gap, time_batch, makespan_batch

def save_test_result(env_test, model, flag_sample=True):
    """save"""
    os.makedirs('test_result', exist_ok=True) if not os.path.exists('test_result') else None

    # vali_result, vali_result_100 = validate(env, model.policy_old)
    gap_chart, dur_chart, mkspan_chart = [], [], []
    for sample in range(100):
        print(f'##### Sample:{sample + 1} #####')
        gap, duration, makespan = test(env_test, model.policy_old, flag_sample=flag_sample)
        gap_chart.append(gap)
        dur_chart.append(duration)
        mkspan_chart.append(makespan)
        gap_mean = gap.mean()
        print('\n', gap_mean, '\n##################################')
        # save
        df_gap = pd.DataFrame(gap_chart)
        df_gap.to_csv('./test_result/gap_chart.csv', index=False)
        df_dur = pd.DataFrame(dur_chart)
        df_dur.to_csv('./test_result/dur_chart.csv', index=False)
        df_mkspan = pd.DataFrame(mkspan_chart)
        df_mkspan.to_csv('./test_result/mkspan_chart.csv', index=False)

    min_gap = df_gap.min(axis=0)
    print(min_gap)

if __name__== '__main__':
    # load model
    from use_RL_model import load_RL_model
    import pandas as pd

    model, _ = load_RL_model(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env_paras = {"num_jobs": 20, "num_mas": 30, "batch_size": 100}
    env = get_validate_env(device)
    env_test, env_names = get_test_envs(device)

    os.makedirs('test_result', exist_ok=True) if not os.path.exists('test_result') else None

    # vali_result, vali_result_100 = validate(env, model.policy_old)
    gap_chart, dur_chart, mkspan_chart = [],[],[]
    for sample in range(1):
        print(f'##### Sample:{sample+1} #####')
        gap, duration, makespan = test(env_test, model.policy_old, flag_sample=False)
        gap_chart.append(gap)
        dur_chart.append(duration)
        mkspan_chart.append(makespan)
        gap_mean = gap.mean()
        print('\n',gap_mean, '\n##################################')
        # save
        df_gap = pd.DataFrame(gap_chart)
        df_gap.to_csv('./test_result/gap_chart.csv', index=False)
        df_dur = pd.DataFrame(dur_chart)
        df_dur.to_csv('./test_result/dur_chart.csv', index=False)
        df_mkspan = pd.DataFrame(mkspan_chart)
        df_mkspan.to_csv('./test_result/mkspan_chart.csv', index=False)
