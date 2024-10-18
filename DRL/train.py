import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import time
from collections import deque
import pandas as pd
import numpy as np
from visdom import Visdom
from RL_model.random_maintenance import random_maintenance
from PPO_model import *
from validate import *
from MMBP_env_old_ver import MMBPEnv
from tqdm import tqdm
import gc


os.environ['CUDA_LAUNCH_BLOCKING']="1"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--cuda_number', help="set cuda number", type=int, default=0)
parser.add_argument('--if_go_on_training', help="if go on training", type=bool, default=False)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(if_go_on_training, if_add_maintenance, if_release_and_due, device=None,
         data_folder='./data_mmbp/',
         save_path='./model/save_best_10_10_20240522.pt'):	# before 15-8-560
    """
    Main function for training
    :param if_go_on_training: Whether to continue training
    :param if_add_maintenance: Whether to add random maintenance
    :param data_folder: The path of the data folder
    :param save_path: The path of the model to be loaded
    """
    # PyTorch initialization
    # gpu_tracker = MemTracker()  # Used to monitor memory (of gpu)
    gc.collect()
    torch.cuda.empty_cache()
    data_folder = data_folder
    print("Device", device)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('device:', device)
    else:
        device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print(f"Go on Training? {if_go_on_training} on model {save_path}\n Remember to run 'python -m visdom.server' in your terminal")
    print("###########################################################################################################")
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    # Load config and init objects
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]
    env_valid_paras = load_dict["env_valid_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    env_paras["device"] = device
    model_paras["device"] = device
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]

    # num_jobs = env_paras["num_jobs"]
    # num_mas = env_paras["num_mas"]  # this one is not useful, as machine num is determined by stage
    # opes_per_job_min = int(num_mas * 0.8)
    # opes_per_job_max = int(num_mas * 1.2)

    memories = Memory()

    # PPO_Model

    # load model and initialize environment
    model = PPO(model_paras, train_paras, num_envs=train_paras["parallel_iter"])  # 原为env_paras["batch_size"]   # PPO_Model
    print('build valid & test envs')
    env_valid = get_validate_env(device=device)  # Create an environment for validation
    env_test, _ = get_test_envs(device=device)  # Create an environment for testing
    maxlen = 1  # Save the best model
    best_models = deque()
    makespan_best = float('inf')

    # whether load old model to continue training
    if if_go_on_training:
        save_path = save_path
        model_CKPT = torch.load(save_path, map_location=device)
        model.policy.load_state_dict(model_CKPT)
        model.policy_old.load_state_dict(model_CKPT)
    # Use visdom to visualize the training process
    is_viz = train_paras["viz"]
    if is_viz:
        viz = Visdom(env=train_paras["viz_name"])

    # Generate data files and fill in the header
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = './save/train_{0}'.format(str_time)
    os.makedirs(save_path)
    valid_results = []
    valid_results_100 = []
    reward_results, loss_results = [], []

    # Start training iteration
    start_time = time.time()
    PATH_IN_DATA_FOLDER = os.listdir(data_folder)
    PATH_IN_DATA_FOLDER.sort()

    """for file_num in range(len(PATH_IN_DATA_FOLDER)):
        if str(num_jobs)+'j' in PATH_IN_DATA_FOLDER[file_num]:
            pass
        else:
            PATH_IN_DATA_FOLDER.remove(PATH_IN_DATA_FOLDER[file_num])"""
    print('files:', PATH_IN_DATA_FOLDER)
    number_of_ins = len(PATH_IN_DATA_FOLDER)
    tail = number_of_ins % train_paras['parallel_iter']
    if tail != 0:
        PATH_IN_DATA_FOLDER = PATH_IN_DATA_FOLDER[0:-tail]
        print(f'current num of file:{len(PATH_IN_DATA_FOLDER)}')
    # how_many_i_back_to_0 = len(PATH_IN_DATA_FOLDER) - len(PATH_IN_DATA_FOLDER) % train_paras["parallel_iter"]
    env = None

    for i in tqdm(range(1, train_paras["max_iterations"]+1)):
        # Replace training instances every x iteration (x = 30 in paper)
        if (i-1) % train_paras["parallel_iter"] == 0:
            # e.g. pi=30, trigger when i =1, 21, 41
            # instances use consistent operations to speed up training
            # series = i // number_of_ins # x_th series of files

            start_point = i % number_of_ins # 0, 200, 400 切换，sp=1-200
            filelist = PATH_IN_DATA_FOLDER[start_point-1:start_point-1+train_paras["parallel_iter"]]
            print(f'start:{start_point} -> ins: {len(filelist)}\n {filelist}\n', end='\t')
            # filelist = PATH_IN_DATA_FOLDER  # This is all the instances
            files = [data_folder + x for x in filelist]

            del env
            # initialize the environment
            env = MMBPEnv(ins_file=files, device=device, render_mode=None)
            num_mas = env.instance.num_mas
            num_jobs = env.instance.num_jobs
            CT = env.schedule.makespan_batch
            CT_MIN = torch.min(CT).cpu().numpy().astype(int)

            # set release and due
            # 每当重新开始parallel iter时，release&due更改
            if if_release_and_due:
                random_rd = np.random.randint(0, CT_MIN, size=(num_jobs,2))
                random_rd[:,1] = random_rd[:,0] + random_rd[:,1]
                release_and_due = random_rd
                env.if_due = env.set_release_and_due_of_job(release_and_due, device=device)
                # because the release and due time are set, time need to be move
                # move then update state
                env.set_release_and_due_of_job(release_and_due, in_proc=True)
            # set maintenance
            if if_add_maintenance:
                maintenance = random_maintenance(CT_MIN, num_mas)
                env.update_main_info(mt=maintenance)

            # print('num_job: ', num_jobs, '\tnum_mas: ', num_mas)
            """position = (i-1) % how_many_i_back_to_0
            filelist = PATH_IN_DATA_FOLDER[position:train_paras["parallel_iter"]+position]"""
        # print("iteration:" + str(i))
        # Get state and completion signal
        state = env.state
        done = False
        dones = env.schedule.done_batch
        last_time = time.time()

        # Schedule in parallel
        while ~done:
            with torch.no_grad():
                actions = model.policy_old.act(state, memories, dones)
            state, rewards, dones, _, _ = env.step(actions)
            done = dones.all()
            memories.rewards.append(rewards)
            memories.is_terminals.append(dones)
            # gpu_tracker.track()  # Used to monitor memory (of gpu)
        # print("spend_time: %.2f" % (time.time()-last_time))

        # Verify the solution
        gantt_result = env.validate_gantt()[0]
        if not gantt_result:
            print("Scheduling Error！！！！！！")
        # print("Scheduling Finish")
        env.reset()

        # if iter mod x = 0 then update the policy (x = 1 in paper)
        if i % train_paras["update_timestep"] == 0:
            loss, reward = model.update(memories, env_paras, train_paras)
            print(i, "\treward: ", '%.3f' % reward, "; loss: ", '%.3f' % loss, '####\t', end='')
            memories.clear_memory()
            if is_viz:
                viz.line(X=np.array([i]), Y=np.array([reward]),
                    win='window{}'.format(0), update='append', opts=dict(title='reward of envs'))
                viz.line(X=np.array([i]), Y=np.array([loss]),
                    win='window{}'.format(1), update='append', opts=dict(title='loss of envs'))  # deprecated
            reward_results.append(reward)   #.item())
            loss_results.append(loss)
        # if iter mod x = 0 then validate the policy (x = 10 in paper)
        if i % train_paras["save_timestep"] == 0:
            # print('\nValidating')
            # Record the average results and the results on each instance
            vali_result, vali_result_all = validate(env_valid, model.policy_old)
            test_gap, _, _ = test(env_test, model.policy_old)
            valid_results.append(vali_result)
            valid_results_100.append(vali_result_all)
            print('valid {:.2f} VS best {:.2f}'.format(vali_result, makespan_best), 'gap:', test_gap.mean(), test, end='\t')
            # Save the best model
            if vali_result < makespan_best:
                print('save model @ ', i, end='')
                makespan_best = vali_result
                if len(best_models) == maxlen:
                    delete_file = best_models.popleft()
                    os.remove(delete_file)
                save_file = '{0}/save_best_{1}_{2}_{3}.pt'.format(save_path, num_jobs, num_mas, i)

                best_models.append(save_file)
                torch.save(model.policy.state_dict(), save_file)

            if is_viz:
                viz.line(
                    X=np.array([i]), Y=np.array([vali_result]),
                    win='window{}'.format(2), update='append', opts=dict(title='makespan of valid'))

            # Save the data of training curve to files
            data = pd.DataFrame(np.array(valid_results).transpose(), columns=["res"])
            data.to_csv('{0}/training_valid_mean_{1}.csv'.format(save_path, str_time), sep=',',index=True, header=True)

            # column = [i_col for i_col in range(100)]
            data_train = pd.DataFrame(np.array(torch.stack(valid_results_100, dim=0).to('cpu')))
            data_train.to_csv('{0}/training_valid_all_{1}.csv'.format(save_path, str_time), sep=',',index=True, header=True)

            data_reward = pd.DataFrame(np.array(reward_results).transpose())
            data_reward.to_csv('{0}/reward_{1}.csv'.format(save_path, str_time))

            data_loss = pd.DataFrame(np.array(loss_results).transpose())
            data_loss.to_csv('{0}/loss_{1}.csv'.format(save_path, str_time))

    print("total_time: ", time.time() - start_time)

    save_test_result(env_test, model, flag_sample=True)


if __name__ == '__main__':

    setup_seed(101) #55
    args = parser.parse_args()
    cuda_number = args.cuda_number
    if_go_on_training = args.if_go_on_training
    main(if_go_on_training=if_go_on_training, if_add_maintenance=False, if_release_and_due=True,
         device='cuda:'+str(cuda_number) if cuda_number is not None else 'cuda:0')

    # use vizdom to visualize the training process in terminal
    # python -m visdom.server
