import time
import json
from PPO_model import *
from random_maintenance import random_maintenance
import pandas as pd
from tqdm import tqdm
import random
import os

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_RL_model(i):
    """load model with paras in config"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type=='cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device.type)
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    train_paras = load_dict["train_paras"]
    model_paras = load_dict["model_paras"]
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]
    model_paras["device"] = device
    mod_files = os.listdir('./model/')[:]
    print("models: ", [str(x)+'-'+str(mod_files[x]) for x in range(len(mod_files))])
    i = int(input("your choice is: "))
    print("choose model: ", mod_files[i])

    memories = Memory()
    model = PPO(model_paras, train_paras)
    envs = []  # Store multiple environments
    if device.type == 'cuda':
        model_CKPT = torch.load('./model/' + mod_files[i])
    else:
        model_CKPT = torch.load('./model/' + mod_files[i], map_location='cpu')
    print('\nloading checkpoint:', mod_files[i])
    model.policy.load_state_dict(model_CKPT)
    """print("state state:\n")
    for param_tensor in model.policy.state_dict():
        print(param_tensor, "\t", model.policy.state_dict()[param_tensor].size())"""
    model.policy_old.load_state_dict(model_CKPT)

    return model, memories


def use_DRL_model(env, model, memories):
    """pack of use DRL"""
    state = env.state
    dones = env.schedule.done_batch
    done = False  # Unfinished at the beginning
    i = 0
    t_start = time.time()
    while ~done:
        i += 1
        with torch.no_grad():
            action = model.policy_old.act(state, memories, dones)
            ############### policy_old 也是 policy, 本质上使用的是HGNN_SCHEDULER ###############
            # print('ACTION NOW:', action)
        """if action[0,0]==26:
            print()"""
        state, rewards, dones, _, _ = env.step(action)
        done = dones.all()
    t_end = time.time()
    duration = t_end - t_start
    env.render()
    makespan = env.schedule.makespan_batch.cpu().numpy()[0]
    result_correct, _ = env.validate_gantt()
    # print(env.main_info.maintenance, 'makespan:', makespan, ', time:{:.4f}'.format(duration))
    # print("result correct:", result_correct)
    maint = env.main_info.maintenance if env.main_info is not None else None
    return maint, makespan,duration


def test_DRL_model(folder_name='../Data/', file_names=None, times_for_iter=10, percentage_of_machine=0.2, percentage_of_CT=0.1):
    """default folder = data public
    default files = first 40 samples"""
    file_names = os.listdir(folder_name)[:40] if file_names is None else [folder_name + i for i in file_names]
    root_file = os.path.abspath('../Data')
    # every file
    device = 'cuda'
    for sample in file_names:
        print(sample)
        df_current = pd.DataFrame(columns=['maintenance', 'makespan', 'duration'])
        env = MMBPEnv(path=sample, batch=1, render_mode=None, device=device)
        CT = env.schedule.makespan_batch[0]
        MACHINES = env.instance.num_mas
        # 随机生成dynamic event
        for batch in tqdm(range(times_for_iter)):
            maintenance = random_maintenance(CT.cpu(), MACHINES, ratio_machine=percentage_of_machine, ratio_time=percentage_of_CT)
            # print('maintenance: ', maintenance)
            env.update_main_info(maintenance)
            record = [use_DRL_model(env)]
            env.reset()
            df_new = pd.DataFrame(record, columns=['maintenance', 'makespan','duration'])
            df_current = pd.concat([df_current, df_new], ignore_index=True)
        df_current.to_csv('./table/test/'+sample[-7:-4]+'.csv')


if __name__=="__main__":
    model, memories = load_RL_model(-1)
    import sys
    sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
    from MMBP_v1_1 import MMBPEnv
    import matplotlib.pyplot as plt
    import numpy as np
    import gc
    import os

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    gc.collect()
    torch.cuda.empty_cache()

    setup_seed(1)
    # path = "../Data/Mk01.fjs"
    """path = "../MMBP/p11.txt"
    # path = "../Environment/1.txt"
    env = FJSPEnv(path, batch=1, render_mode='p_d',color_type=plt.cm.GnBu(np.linspace(0,1,10)))# maintenance=[[0, 7, 25], [3, 30, 50]]
    _, makespan,_ = use_DRL_model(env)
    print(makespan)"""
    #files=["Mk"+i+".fjs" for i in ['10']]# '03','04','05','06','07','08',00
    # test_DRL_model(file_names=files, times_for_iter=90)

    '''# question = 'p10_fake'
    question = 'p12'
    batch = 2
    print(question, ' is calculating ...')

    # default_path = '../Data/Mk10.fjs'#, '../Data/Mk04.fjs']
    default_path = f'../MMBP/Data_G_FJSP_Version/{question}.fjs'
    rnd_info = f'../MMBP/Data_G_FJSP_Version_RnD/{question}.csv'
    rnd = pd.read_csv(rnd_info, index_col=[0, 1])  # release and due date
    rsu = pd.read_csv(f'../MMBP/Data_G_FJSP_Version_StageUnit/{question}.csv', index_col=0)  # relation stage unit
    torch.manual_seed(20199650253898)'''
    batch = 100

    device = "cuda"
    env = MMBPEnv(device=device, batch=batch, ins_file=default_path, release_and_due=rnd, relation_stage_unit=rsu,
                  # changeover_file='../MMBP/Data_changeover/i07',
                  render_mode='p_d')  # maintenance=[[0, 6, 20], [3, 5, 25],[2, 40,60]],
    print(env.instance.num_jobs)
    env.color_type = plt.cm.rainbow(np.linspace(0, 1, env.instance.num_jobs))
    t1 = time.time()
    _, makespan, _ = use_DRL_model(env, model, memories)
    t2 = time.time()-t1
    print('Makespan',makespan, 'Time:', t2)