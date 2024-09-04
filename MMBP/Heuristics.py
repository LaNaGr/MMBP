import time
import torch
import random


'''
First In First Out (FIFO): The first job in is processed first. (earliest release date)
Last In First Out (LIFO): The last job in is processed first.
Shortest Processing Time (SPT): The job with the shortest processing time is processed first.
Longest Processing Time (LPT): The job with the longest processing time is processed first.
Shortest Total Processing Time (STPT): The job with the shortest total processing time is processed first. 	
Longest Total Processing Time (LTPT): The job with the longest total processing time is processed first. 	
Least Operation Remaining (LOR): The job with the least remaining operations to be completed is processed first.
Most Operation Remaining (MOR): The job with the most remaining operations to be completed is processed first.
Random: Job priority is randomly assigned.
critical ratio(CR) – this is the ratio of the time until the due date to the processing time: jobs with the lowest CR are processed first.'''

def get_action_prob(state):
    """find eligible O-M
    e.g. mk01:shape(batch, n_job, m_machine)"""
    # Uncompleted instances
    batch_idxes = state.batch_idxes
    # Raw feats
    raw_opes = state.feat_opes_batch.transpose(1, 2)[batch_idxes]
    raw_mas = state.feat_mas_batch.transpose(1, 2)[batch_idxes]
    proc_time = state.proc_times_batch[batch_idxes]
    # Normalize
    nums_opes = state.nums_opes_batch[batch_idxes]
    ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)
    
    # Matrix indicating whether processing is possible
    # shape: [len(batch_idxes), num_jobs, num_mas]
    eligible_proc = state.ope_ma_adj_batch[batch_idxes].gather(1, ope_step_batch[..., :, None].expand(-1, -1, state.ope_ma_adj_batch.size(-1))[batch_idxes])
    # Matrix indicating whether machine is eligible
    # shape: [len(batch_idxes), num_jobs, num_mas]
    ma_eligible = ~(state.mask_ma_procing_batch[batch_idxes] + state.mask_maintenance_ma_batch[batch_idxes]).unsqueeze(1)
    # Matrix indicating whether job is eligible
    # shape: [len(batch_idxes), num_jobs, num_mas]
    job_eligible = ~(state.mask_job_procing_batch[batch_idxes] +
                     state.mask_job_finish_batch[batch_idxes])[:, :, None]
    # shape: [len(batch_idxes), num_jobs, num_mas]
    eligible = job_eligible & ma_eligible & (eligible_proc == 1)    # 理论上屏蔽其中一个即可不选中
    if (~(eligible)).all():
        print("No eligible O-M pair!")
        return
    return eligible


def random_method(eligible_matrix, ope_step_batch, batch_idxes):
    rand_matrix = torch.rand(eligible_matrix.size())
    mul = rand_matrix * eligible_matrix
    point = mul.flatten(1).argmax(dim=1)
    mas = (point % eligible_matrix.size(2)).long()
    jobs = (point / eligible_matrix.size(2)).long()
    opes = ope_step_batch[batch_idxes, jobs]
    return torch.stack((opes, mas, jobs), dim=1).t()


def FIFO(env):
    # First In First Out
    next_opes = env.schedule.ope_step_batch
    bools = env.mask.mask_job_finish_batch[0]
    if env.release is not None:
        release_time = env.release
    else:
        release_time = torch.zeros(size=(env.instance.num_jobs,))
    sorted_indice = torch.argsort(release_time)
    for i in sorted_indice:
        if ~bools[i]:
            o = next_opes[0, i]
            break
    return o, i


def MOPNR(env):
    # same action; MOST OPE NUMBER REMAINING
    size = env.instance.num_jobs
    next_opes = env.schedule.ope_step_batch
    bools = env.mask.mask_job_finish_batch[0] # +~env.mask.mask_job_release_batch[0]
    sum_pj_mean_for_Ji = torch.ones(size=(size,))
    sum_pj_mean_for_Ji[bools] = 0
    sum_pj_mean_for_Ji[~bools] = env.feature.feat_opes_batch[0, 3, next_opes[0][~bools]]
    nums, idx_sort_descending = torch.sort(sum_pj_mean_for_Ji, dim=-1, descending=True, stable=False)
    # default: sort-stable is False, the order is not preserved.(random choosing)
    equal_list = []
    for i in range(size-1):
        if nums[i]==nums[i+1]:
            if i==size-2:
                equal_list.append(i)
            else:
                pass
        else:
            equal_list.append(i+1)
    last_equal = 0
    for equal in equal_list:
        init_list = [i for i in range(last_equal,equal)]
        random.shuffle(init_list)
        for i in init_list:
            o = next_opes[0, idx_sort_descending[i]]
            if ~env.mask.mask_job_finish_batch[0, idx_sort_descending[i]]:
                break
    return o, idx_sort_descending[i]


def LWRM(env):
    # Least work remaining sum(min(pijk))for O_ij in J_i
    size = env.instance.num_jobs
    next_opes = env.schedule.ope_step_batch
    bools = env.mask.mask_job_finish_batch[0]
    work_remaining_time = torch.zeros(size=(size,))
    work_remaining_time[~bools] = env.feature.feat_opes_batch[0, 4, next_opes[0][~bools]] - env.schedule.schedules_batch[0, next_opes[0][~bools], 2] # i.e. unschduled ope's ST
    work_remaining_time[bools] = torch.inf
    ts, idxs = torch.sort(work_remaining_time)
    for id in idxs:
        if ~env.mask.mask_job_finish_batch[0, id]:
            o = next_opes[0, id]
            break
    return o, id

def MWRM(env):
    # most work remaining
    size = env.instance.num_jobs
    next_opes = env.schedule.ope_step_batch
    bools = env.mask.mask_job_finish_batch[0]
    work_remaining_time = torch.zeros(size=(size,))
    work_remaining_time[~bools] = env.feature.feat_opes_batch[0, 4, next_opes[0][~bools]] - \
                                  env.schedule.schedules_batch[0, next_opes[0][~bools], 2]  # i.e. unschduled ope's ST
    work_remaining_time[bools] = torch.inf
    ts, idxs = torch.sort(work_remaining_time, descending=True)
    for id in idxs:
        if ~env.mask.mask_job_finish_batch[0, id]:
            o = next_opes[0, id]
            break
    return o, id

def EET(env):
    # first idle machine; Earliest End Time
    m_idle_time = env.feature.feat_mas_batch[0,1]
    m = torch.argsort(m_idle_time)
    return m


def choose_m(o, env, ms):
    # for EET
    adj = env.instance.ope_ma_adj_batch[0,o]
    m_choose_valid = torch.nonzero(adj)
    # print('m_valid_size:', m_choose_valid.size())
    """if m_choose_valid.size()==torch.Size([0, 1]):
        m_choose = m_choose_valid"""
    if ms.size()==torch.Size([]):
        m_choose = ms
    else:
        for i in ms:
            if i in m_choose_valid:
                m_choose = i
                break
    return m_choose


def SPT(env,o):
    # Shortest Processing Time
    adj = env.instance.ope_ma_adj_batch[0,o]
    proc_time = env.instance.proc_times_batch[0,o]
    pijk, idx_pijk = torch.sort(proc_time, stable=False)
    for id in idx_pijk:
        if adj[id]==0:
            pass
        else:
            m_MIN_pijk=id
            break
    # 寻找相同值
    ms = torch.nonzero(proc_time==proc_time[m_MIN_pijk])
    m = random.choice(ms).unsqueeze(0)
    return m

class Heuristic():
    def __init__(self, str_rule):
        self.rule = str_rule
        self.valid_rule_list = ['FIFO_EET', 'FIFO_SPT', 'MOPNR_EET', 'MOPNR_SPT','LWRM_EET','LWRM_SPT','MWRM_EET','MWRM_SPT']
        if str_rule == 'FIFO_EET':
            self.go = self.FIFO_EET
        if str_rule == 'FIFO_SPT':
            self.go = self.FIFO_SPT
        if str_rule == 'MOPNR_EET':
            self.go = self.MOPNR_EET
        if str_rule == 'MOPNR_SPT':
            self.go = self.MOPNR_SPT
        if str_rule == 'LWRM_EET':
            self.go = self.LWRM_EET
        if str_rule == 'LWRM_SPT':
            self.go = self.LWRM_SPT
        if str_rule == 'MWRM_EET':
            self.go = self.MWRM_EET
        if str_rule == 'MWRM_SPT':
            self.go = self.MWRM_SPT

    def FIFO_EET(self, env):
        action_o, job = FIFO(env)
        #job = env.instance.opes_appertain_batch[0, action_o]
        action_m_list = EET(env)
        m = choose_m(action_o, env, action_m_list)
        return torch.tensor([[action_o], [m], [job]])

    def FIFO_SPT(self, env):
        action_o, job = FIFO(env)
        m = SPT(env, action_o)
        return torch.tensor([[action_o], [m], [job]])

    def MOPNR_EET(self, env):
        action_o, job = MOPNR(env)
        #job = env.instance.opes_appertain_batch[0, action_o]
        action_m_list = EET(env)
        m = choose_m(action_o, env, action_m_list)
        return torch.tensor([[action_o], [m], [job]])

    def MOPNR_SPT(self, env):
        action_o, job = MOPNR(env)
        m = SPT(env, action_o)
        return torch.tensor([[action_o], [m], [job]])

    def LWRM_EET(self, env):
        action_o, job = LWRM(env)
        action_m_list = EET(env)
        m = choose_m(action_o, env, action_m_list)
        return torch.tensor([[action_o], [m], [job]])

    def LWRM_SPT(self, env):
        action_o, job = LWRM(env)
        m = SPT(env, action_o)
        return torch.tensor([[action_o], [m], [job]])

    def MWRM_EET(self, env):
        action_o, job = MWRM(env)
        action_m_list = EET(env)
        m = choose_m(action_o, env, action_m_list)
        return torch.tensor([[action_o], [m], [job]])

    def MWRM_SPT(self, env):
        action_o, job = MWRM(env)
        action_m_list = SPT(env, action_o)
        m = choose_m(action_o, env, action_m_list)
        return torch.tensor([[action_o], [m], [job]])


if __name__ == "__main__":
    # Dispatching rule应当脱离eligible矩阵存在
    from MMBP_v0_2_heuristic_only import MMBPEnv
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # torch.manual_seed(20199650253898)
    random.seed(1)
    question = 'i07'
    batch = 1
    print(question, ' is calculating ...')

    # default_path = '../Data/Mk10.fjs'#, '../Data/Mk04.fjs']
    default_path = f'../MMBP/Data_G_FJSP_Version/{question}.fjs'
    """if os.path.exists(default_path):
        pass
    else:
        """
    from from_mmbp_to_fjsp import record_fjsp_add
    record_fjsp_add(question)

    rnd_info = f'../MMBP/Data_G_FJSP_Version_RnD/{question}.csv'
    rnd = pd.read_csv(rnd_info, index_col=[0, 1])  # release and due date
    rsu = pd.read_csv(f'../MMBP/Data_G_FJSP_Version_StageUnit/{question}.csv', index_col=0)  # relation stage unit
    # torch.manual_seed(20199650253898)

    device = "cuda"
    env = MMBPEnv(device=device, batch=batch, ins_file=default_path, release_and_due=rnd, relation_stage_unit=rsu,
                  changeover_file='./Data_changeover/i07',
                  render_mode='p_d')  # maintenance=[[0, 6, 30], [3, 5, 25],[2, 40,60]],
    env.color_type = plt.cm.rainbow(np.linspace(0, 1, env.instance.num_jobs))
    print(env.instance.num_jobs)

    state = env.state
    dones = env.schedule.done_batch
    done = False  # Unfinished at the beginning
    i = 0
    valid_rule_list = ['FIFO_EET', 'FIFO_SPT', 'MOPNR_EET', 'MOPNR_SPT', 'LWRM_EET', 'LWRM_SPT', 'MWRM_EET', 'MWRM_SPT']
    rule = valid_rule_list[1]
    method = Heuristic(rule)
    t_start = time.time()
    while ~done:
        i += 1
        action = method.go(env)
        # time
        time_move = env.feature.feat_mas_batch[0,1,action[1,0]]
        if time_move>=env.schedule.time:
            env.time_move_to_t(torch.tensor([time_move]))
        state, rewards, dones, _, _ = env.step(action)
        done = dones.all()
    t_end = time.time()
    duration = t_end-t_start
    print(env.schedule.makespan_batch)

    env.render()
    env.validate_gantt()


