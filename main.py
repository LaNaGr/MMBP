from MMBP_v1_1 import MMBPEnv
from MMBP_v0_2_heuristic_only import MMBPEnv_Heu
import torch
import time


def test_MMBPEnv_Heu(problem='case_studyB'):
    """Example of test MMBPEnv_Heu"""
    print("Example: MMBPEnv_Heu")
    torch.manual_seed(20199650253898)
    device = "cpu"
    default_path = './' + problem + '.fjs'
    env = MMBPEnv_Heu(path=default_path, batch=1, render_mode='p_d',
                      device=device, maintenance=[[10, 4, 6], [3, 5, 25], [8, 20, 45]])
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
        env.render(name=rule)
        t_end = time.time()
        duration = t_end - t_start
        validation = env.validate_gantt()
        makespan = env.schedule.makespan_batch.numpy()[0]
        env.reset()
        print([rule, makespan, duration], 'validation:', validation[0])
        if validation[0]:
            record.append([rule, makespan, duration])
        else:
            record.append([rule, None, duration])
    df = pd.DataFrame(record, columns=['rule', 'makespan', 'duration'])
    df.to_csv('Heu_result_dynamic_B.csv')


def test_MMBPEnv(problem='case_studyA'):
    """Example of test MMBPEnv
    This method can adjust with Heu and RL method."""
    print("Example: MMBPEnv")
    batch = 2
    default_path = problem + '.fjs'
    rnd = './' + problem + '.csv'
    device = "cuda"
    env = MMBPEnv(device=device, batch=batch, ins_file=default_path, release_and_due=rnd,  # relation_stage_unit=rsu,
                  render_mode='p_d', time_slot=1,
                  maintenance=[[10, 4, 6], [0, 0.5, 1.5]])
    done = False  # Unfinished at the beginning
    i = 0

    print("First Test: Random Method")
    from MMBP.Heuristics import random_method
    last_time = time.time()
    while ~done:
        i += 1
        eli = env.eli
        action = random_method(eli, ope_step_batch=env.state.ope_step_batch, batch_idxes=env.state.batch_idxes)
        # print(action)
        state, rewards, dones, _, _ = env.step(action)
        done = dones.all()
    spend_time = time.time() - last_time  # The time taken to solve this environment (instance)
    env.render(name=problem+'_random')
    result_correct, _ = env.validate_gantt()
    print("spend_time: %.2f" % spend_time, "result correct:", result_correct, "makespan:", env.schedule.makespan_batch)

    print("Second Test: Heuristic Method")
    env.reset()
    done = False  # Unfinished at the beginning
    last_time = time.time()
    while ~done:
        i += 1
        eli = env.eli
        action = random_method(eli, ope_step_batch=env.state.ope_step_batch, batch_idxes=env.state.batch_idxes)
        # print(action)
        state, rewards, dones, _, _ = env.step(action)
        done = dones.all()
    spend_time = time.time() - last_time
    env.render(name=problem+'_RL')