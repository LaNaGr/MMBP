from MMBP_for_heu import MMBPEnv
import torch
import pandas as pd
import os
from Heuristics import Heuristic, random_method
import time
import matplotlib.pyplot as plt
import numpy as np


def iter_rule(env, rule_list):
    record=[]
    for rule in rule_list:
        method = Heuristic(rule)
        # print(rule, " start")
        done = False  # Unfinished at the beginning
        i = 0
        t_start = time.time()
        while ~done:
            i += 1
            action = method.go(env)
            # print("action:", action)
            # print(action)
            # time
            time_move = env.feature.feat_mas_batch[0, 1, action[1, 0]]
            if time_move >= env.schedule.time:
                env.time_move_to_t(torch.tensor([time_move]))
            state, rewards, dones, _, _ = env.step(action)
            done = dones.all()
        t_end = time.time()
        duration = t_end - t_start
        validation = env.validate_gantt()
        makespan = env.schedule.makespan_batch.numpy()[0]
        env.reset()
        print([rule, makespan, duration],'validation:', validation[0])
        if validation[0]:
            record.append([rule, makespan, duration])
        else:
            record.append([rule, None, duration])
    df = pd.DataFrame(record, columns=['rule', 'makespan','duration'])

    return df


question = 'p15'
batch = 1
print(question, ' is calculating ...')

# default_path = '../Data/Mk10.fjs'#, '../Data/Mk04.fjs']
default_path = f'../MMBP/Data_G_FJSP_Version/{question}.fjs'
if os.path.exists(default_path):
    pass
else:
    from from_mmbp_to_fjsp import record_fjsp_add

    record_fjsp_add(question)

rnd_info = f'../MMBP/Data_G_FJSP_Version_RnD/{question}.csv'
rnd = pd.read_csv(rnd_info, index_col=[0, 1])  # release and due date
rsu = pd.read_csv(f'../MMBP/Data_G_FJSP_Version_StageUnit/{question}.csv', index_col=0)  # relation stage unit
torch.manual_seed(20199650253898)

device = "cuda"
env = MMBPEnv(device=device, batch=batch, ins_file=default_path, release_and_due=rnd, relation_stage_unit=rsu,
              # changeover_file='./Data_changeover/i07',    # 目前仅有i07具有changeover_file
              render_mode='p_d')  # maintenance=[[0, 6, 20], [3, 5, 25],[2, 40,60]],
print(env.instance.num_jobs)
env.color_type = plt.cm.rainbow(np.linspace(0, 1, env.instance.num_jobs))
state = env.state
dones = env.schedule.done_batch

rule_list = ['FIFO_EET', 'FIFO_SPT', 'MOPNR_EET', 'MOPNR_SPT','LWRM_EET','LWRM_SPT','MWRM_EET','MWRM_SPT']
df_target = iter_rule(env, rule_list)
os.makedirs('./table/iter_heu/', exist_ok=True)
df_target.to_csv(f'./table/iter_heu/{question}_iter_rule.csv', index=False)

"""
rule = 'MWRM_EET'
method = Heuristic(rule)"""

def random_calcu(env):
    done = False  # Unfinished at the beginning
    i = 0
    last_time = time.time()
    while ~done:
        i += 1
        eli = env.eli

        action = random_method(eli, ope_step_batch=env.state.ope_step_batch, batch_idxes=env.state.batch_idxes)
        state, rewards, dones, _, _ = env.step(action)
        done = dones.all()

    spend_time = time.time() - last_time  # The time taken to solve this environment (instance)
    env.render()


    result_correct, _ = env.validate_gantt()

    print("spend_time: %.2f" % spend_time, "result correct:", result_correct)
    pass