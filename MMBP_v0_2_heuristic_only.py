# No changeover time
# No release and due date settings
# Have maintenance window

import time
import os
import gym
import numpy as np
import torch
import copy
from MMBP.load_data_for_mmbp import instance
from MMBP.feat_ver_2 import feature
from MMBP.mask import mask
from MMBP.load_data_for_heu import schedule
from MMBP.state_makespan import EnvState
from MMBP.render_modes import Instance_for_render
import pandas as pd
import matplotlib.pyplot as plt

PIC_SETTINGS = {'machine_name': '', 'job_name': 'Order', 'operation_name': 'Stage'}


class MMBPEnv_Heu(gym.Env):
    """
    Flexible Job Shop Scheduling Problems
    Environment
    PATH: ['','',...] OR ''
    batch: None OR 1
    device: 'cpu','cuda'
    render_mode: default"human", while training, make it None
    maintenance: None OR [[mid, start, end]]
    plt.cm.tab20(np.linspace(0,1,20))
    """

    def __init__(self, path=None, batch=None, device="cpu", render_mode="human", maintenance=None,
                 color_type=None, pic_settings=PIC_SETTINGS):
        self.folder = None
        self.render_mode = render_mode
        self.device = device
        self.reward_batch = None
        self.done = False
        if path is None:
            path = default_path
        ins = instance(path, batch)
        feat = feature(ins)
        mask_at_this_time = mask(batch_size=ins.batch_size, num_jobs=ins.num_jobs, num_mas=ins.num_mas)
        schedule_result = schedule(batch_size=ins.batch_size, num_opes=ins.num_opes, num_mas=ins.num_mas,
                                   num_ope_biases_batch=ins.num_ope_biases_batch, feat_opes_batch=feat.feat_opes_batch,
                                   mask_job_finish_batch=mask_at_this_time.mask_job_finish_batch)
        self.instance = ins
        self.feature = feat
        self.mask = mask_at_this_time
        self.schedule = schedule_result
        # Uncompleted instances 即 batch_idxes 记得每一项都需要这个
        self.state = EnvState(batch_idxes=self.schedule.batch_idxes,
                              feat_opes_batch=self.feature.feat_opes_batch,
                              feat_mas_batch=self.feature.feat_mas_batch,
                              proc_times_batch=self.instance.proc_times_batch,
                              ope_ma_adj_batch=self.instance.ope_ma_adj_batch,
                              ope_pre_adj_batch=self.instance.ope_pre_adj_batch,
                              ope_sub_adj_batch=self.instance.ope_sub_adj_batch,
                              mask_job_procing_batch=self.mask.mask_job_procing_batch,
                              mask_job_finish_batch=self.mask.mask_job_finish_batch,
                              mask_ma_procing_batch=self.mask.mask_ma_procing_batch,
                              opes_appertain_batch=self.instance.opes_appertain_batch,
                              ope_step_batch=self.schedule.ope_step_batch,
                              end_ope_biases_batch=self.instance.end_ope_biases_batch,
                              time_batch=self.schedule.time,
                              nums_opes_batch=self.instance.nums_opes,
                              mask_maintenance_ma_batch=self.mask.mask_maintenance_ma_batch)
        self.old_state = copy.deepcopy(self.state)
        self.folder = self.make_folder(path)
        # maintenance time window
        self.main_info = main_information(maintenance) if maintenance is not None else None
        self.color_type = color_type if color_type is not None else plt.cm.rainbow(np.linspace(0, 1, self.instance.num_jobs))
        self.pic_settings = pic_settings
        self.release = None

    def add_main_info(self, mt):
        if self.main_info is None:
            self.main_info = main_information(mt)
        else:
            mt1 = self.main_info.maintenance
            mt_new = mt1 + mt
            self.main_info = main_information(mt_new)

    def update_main_info(self, mt):
        self.main_info = main_information(mt)

    def make_folder(self, path):
        if self.instance.batch_size == 1:
            folder_path = "./table/" + os.path.basename(path)[0:-4]  # time_now
        else:
            folder_path = "./table/" + os.path.basename(path[0])[0:-4]
        if self.render_mode in ['p_d', 'draw', 'print_table']:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        return folder_path

    def reset(self, **kwargs):
        """
        Reset the environment to its initial state
        """
        self.done = False
        self.reward_batch = None
        self.instance.reset_self()
        self.feature.reset_self()
        self.state = copy.deepcopy(self.old_state)
        self.schedule.reset_self()
        self.mask.reset_self()
        self.main_info = main_information(self.main_info.maintenance)
        return self.state

    def render(self, changeable_mode=None, name=None):
        """
        3 modes
        default "human": print schedules_batch
        """
        mode = self.render_mode if changeable_mode is None else changeable_mode
        if mode is None:
            pass
        elif mode == "human":
            for i in range(self.instance.batch_size):
                print(self.schedule.schedules_batch[i, :, :])
        else:
            # this 'ins' is in render_modes, but not load_data
            if name is None:
                ins = Instance_for_render(batch_size=self.instance.batch_size, file_name=self.instance.file_path,
                                          num_jobs=self.instance.num_jobs, num_mas=self.instance.num_mas,
                                          num_opes=self.instance.num_opes, nums_opes=self.instance.nums_opes,
                                          schedules_batch=self.schedule.schedules_batch,
                                          opes_appertain_batch=self.instance.opes_appertain_batch,
                                          num_ope_biases_batch=self.instance.num_ope_biases_batch,
                                          maintenance_info=self.main_info.maintenance,
                                          color_type=self.color_type, pic_settings=self.pic_settings, )
            else:
                ins = Instance_for_render(batch_size=self.instance.batch_size, file_name=self.instance.file_path,
                                          num_jobs=self.instance.num_jobs, num_mas=self.instance.num_mas,
                                          num_opes=self.instance.num_opes, nums_opes=self.instance.nums_opes,
                                          schedules_batch=self.schedule.schedules_batch,
                                          opes_appertain_batch=self.instance.opes_appertain_batch,
                                          num_ope_biases_batch=self.instance.num_ope_biases_batch,
                                          maintenance_info=self.main_info.maintenance,
                                          color_type=self.color_type, pic_settings=self.pic_settings,
                                          name=name)
            if self.color_type is not None:
                ins.color_type = self.color_type
            else:
                pass
                # default:plt.cm.tab20(np.linspace(0,1,20))
            if mode == 'p_d':
                ins.print_table()
                ins.draw()
            elif mode == 'print_table':
                ins.print_table()
            elif mode == 'draw':
                ins.draw()
            else:
                print("Unknown show_mode")

    def step(self, actions):
        """
        env step to next state
        give back obs
        """

        opes = actions[0, :]
        mas = actions[1, :]
        jobs = actions[2, :]
        self.schedule.N += 1
        batch_idxes = self.schedule.batch_idxes

        # Removed unselected O-M arcs of the scheduled operations
        last_opes = torch.where(opes - 1 < self.instance.num_ope_biases_batch[batch_idxes, jobs],
                                self.instance.num_opes - 1, opes - 1)
        self.instance.update_oma_pt_cca(mas, opes, last_opes, self.schedule.batch_idxes)  # time？？

        # Update for some O-M arcs are removed, such as 'Status', 'Number of neighboring machines' and 'Processing time'
        # Update 'Number of unscheduled operations in the job'
        proc_times = self.instance.proc_times_batch[batch_idxes, opes, mas]
        start_ope = self.instance.num_ope_biases_batch[batch_idxes, jobs]
        end_ope = self.instance.end_ope_biases_batch[batch_idxes, jobs]
        self.feature.update_fob_0123(opes, proc_times, start_ope, end_ope, self.schedule.batch_idxes)

        # Update 'Start time' and 'Job completion time'
        #################### This part is different from MMBPEnv ####################
        #################### Operation start time is decided by the machine idle time and the last operation
        # first, ensure start time ST >= last O in the same J & machine idle time
        if last_opes != self.instance.nums_opes - 1:
            time_now_schedule = torch.where(self.feature.feat_mas_batch[batch_idxes, 1, mas] >=
                                            self.schedule.schedules_batch[batch_idxes, last_opes, 3],
                                            self.feature.feat_mas_batch[batch_idxes, 1, mas],
                                            self.schedule.schedules_batch[batch_idxes, last_opes, 3])  # 这里用所有time
        else:
            time_now_schedule = self.feature.feat_mas_batch[batch_idxes, 1, mas]
        # second, update maintenance end time if there will be maintenance by ST calculated before
        machine_time = self.feature.feat_mas_batch[batch_idxes, 1, :][0]
        machine_time[mas] = time_now_schedule  # machine_time_update_with_t_now_schedule
        if self.main_info.if_maintenance:
            bool_main, main_info_next_step_used = self.main_info.bool_whether_start(machine_time)
            # decide with O start time, if overlap maintenance during O, check at the end of 'step'
            while True in bool_main and self.main_info.if_maintenance:
                self.update_mask_maintenance_m(self.main_info.if_maintenance, bool_main, main_info_next_step_used)
                # third (if schedule has been updated by maintenance), ST >= maintenance end time + O remain time
                time_now_schedule = max(self.feature.feat_mas_batch[batch_idxes, 1, mas], time_now_schedule)
                bool_main, main_info_next_step_used = self.main_info.bool_whether_start(machine_time)

        cca = self.instance.cal_cumul_adj_batch[batch_idxes, :, :]
        eob = self.instance.end_ope_biases_batch[batch_idxes, :]
        oa = self.instance.opes_appertain_batch[batch_idxes, :]
        self.feature.update_fob_45(time_now_schedule, opes, cca, eob, oa, self.schedule.batch_idxes)

        # Update partial schedule (state)
        fob_st = self.feature.feat_opes_batch[batch_idxes, 5, :]
        fob_pt = self.feature.feat_opes_batch[batch_idxes, 2, :]
        self.schedule.update_static(opes, mas, fob_st, fob_pt, proc_times, jobs, time_now_schedule)

        # Update feature vectors of machines
        self.feature.update_fmb(mas, self.instance.ope_ma_adj_batch, proc_times, time_now_schedule,
                                self.schedule.machines_batch, self.schedule.batch_idxes)

        # Update mask
        self.mask.update_with_step(jobs, mas, self.schedule.ope_step_batch, self.instance.end_ope_biases_batch,
                                   self.schedule.batch_idxes)

        # update env info d, r
        done_batch = self.mask.mask_job_finish_batch.all(dim=1)
        self.done = done_batch.all()
        max_time = torch.max(self.feature.feat_opes_batch[:, 4, :], dim=1)[0]
        self.reward_batch = self.schedule.makespan_batch - max_time
        # update schedule info
        self.schedule.update_info(done_batch, max_time)

        # Update the vector for uncompleted instances
        mask_finish = (self.schedule.N + 1) <= self.instance.nums_opes
        if ~(mask_finish.all()):
            self.schedule.batch_idxes = torch.arange(self.instance.batch_size)[mask_finish]

        # maintenance caused mask, mask after time point changed
        if self.main_info.if_maintenance:
            bool_main, main_info_next_step_used = self.main_info.bool_whether_start(
                self.feature.feat_mas_batch[batch_idxes, 1, :][0])
            while True in bool_main and self.main_info.if_maintenance:
                self.update_mask_maintenance_m(self.main_info.if_maintenance, bool_main, main_info_next_step_used)
                bool_main, main_info_next_step_used = self.main_info.bool_whether_start(
                    self.feature.feat_mas_batch[batch_idxes, 1, :][0])
                # self.time_move()
        # because of maintenance, machine_schedule will change end time
        # move schedule.time again
        # Update state of the environment
        self.state.update(batch_idxes=self.schedule.batch_idxes,
                          feat_opes_batch=self.feature.feat_opes_batch,
                          feat_mas_batch=self.feature.feat_mas_batch,
                          proc_times_batch=self.instance.proc_times_batch,
                          ope_ma_adj_batch=self.instance.ope_ma_adj_batch,
                          mask_job_procing_batch=self.mask.mask_job_procing_batch,
                          mask_job_finish_batch=self.mask.mask_job_finish_batch,
                          mask_ma_procing_batch=self.mask.mask_ma_procing_batch,
                          mask_maintenance_ma_batch=self.mask.mask_maintenance_ma_batch,
                          ope_step_batch=self.schedule.ope_step_batch,
                          time_batch=self.schedule.time)

        return self.state, self.reward_batch, self.schedule.done_batch, False, {}

    def time_move_to_t(self, target_time):
        # TARGET TIME: TENSOR OR LIST, same shape as schedule.T
        need_move = self.schedule.time < target_time
        self.schedule.time = torch.max(target_time, self.schedule.time)
        for idx in self.schedule.batch_idxes:
            if need_move[idx]:
                # machine feat change
                which_m = ~(self.feature.feat_mas_batch[idx, 1] >= target_time[idx] + self.schedule.machines_batch[idx,
                                                                                      :, 0])
                self.schedule.machines_batch[idx, which_m, 0] = 0
                utiliz = self.schedule.machines_batch[:, :, 2]
                cur_time = self.schedule.time[:, None].expand_as(utiliz)
                utiliz = torch.minimum(utiliz, cur_time)
                utiliz = utiliz.div(self.schedule.time[:, None] + 1e-5)
                self.feature.feat_mas_batch[:, 2, :] = utiliz
        new_finish_job = torch.where(self.schedule.time.unsqueeze(-1).expand_as(
            self.schedule.machines_batch[:, :, 1]) >= self.schedule.machines_batch[:, :, 1],
                                     self.schedule.machines_batch[:, :, 0], 0)
        new_finish_job = new_finish_job.bool()  # dtype long byte or bool
        # Detect the machines that completed (at above time), 1: idle
        jobs = torch.where(new_finish_job == 1, self.schedule.machines_batch[:, :, 3].double(), -1.0).float()
        jobs_index = np.argwhere(jobs.cpu() >= 0).to(self.device)  # size(batch_size, job_num) 目前空出来的job的坐标
        job_idxes = jobs[jobs_index[0], jobs_index[1]].long()  # 从以上坐标拿出序号
        target_batch_idxes = jobs_index[0]

        if_finish = self.schedule.ope_step_batch == self.instance.end_ope_biases_batch + 1

        self.mask.update_with_time(target_batch_idxes, job_idxes, new_finish_job, if_finish,
                                   num_opes=self.instance.num_opes)
        if self.main_info is not None:
            self.mask.update_with_maintenance_time(self.schedule.time, self.main_info.maintenance,
                                                   self.schedule.stuck_time)
        pass

    def update_mask_maintenance_m(self, if_maintenance, bool, info):
        if if_maintenance:  # When it is needed
            # First, detect whether we're in the time period
            # Note: If some job has already settled @ T_1, it will continue.
            i = 0
            for one_main_target in info:
                m_id, t_1, t_2 = one_main_target
                self.mask.mask_maintenance_ma_batch[i, m_id] = True
                m_available_time = self.schedule.machines_batch[i, m_id, 1]
                t = t_2 - t_1
                # if there is no working job, set it -1
                working_job = self.schedule.machines_batch[i, m_id, 3].long() if m_available_time >= t_1 else -1
                working_ope = (self.schedule.ope_step_batch[i, working_job] - 1).long() if working_job != -1 else -1
                if working_job > -1 and self.schedule.schedules_batch[i, working_ope, 3] > t_1:  # break into an ope
                    # f_mas
                    self.feature.feat_mas_batch[i, 1, m_id] += t
                    # f_ope
                    self.feature.feat_opes_batch[i, 2, working_ope] += t  # update fob_2 proc time
                    neighbor_ope_till_end_this_job = self.instance.end_ope_biases_batch[
                        i, working_job]
                    self.feature.feat_opes_batch[i, 4,
                    working_ope: neighbor_ope_till_end_this_job + 1] += t  # fob end time
                    self.feature.feat_opes_batch[i, 5,
                    working_ope + 1: neighbor_ope_till_end_this_job + 1] += t  # fob start time
                    # schedule s_batch
                    self.schedule.schedules_batch[i, working_ope, 3] += t
                    self.schedule.schedules_batch[i, working_ope + 1:neighbor_ope_till_end_this_job + 1, 2] += t
                    self.schedule.schedules_batch[i, working_ope + 1:neighbor_ope_till_end_this_job + 1, 3] += t
                    # schedule m_batch
                    self.schedule.machines_batch[i, m_id, 1] += t

                    # Update feature vectors of machines, available time & utilization
                    self.feature.feat_mas_batch[i, 1, m_id] = self.schedule.machines_batch[
                        i, m_id, 1]
                    utilization = self.schedule.machines_batch[i, :, 2]
                    cur_time = self.feature.feat_mas_batch[i, 1, :]
                    utilization = torch.minimum(utilization, cur_time)
                    utilization = utilization.div(cur_time + 1e-9)
                    self.feature.feat_mas_batch[i, 2, :] = utilization
                    # update env info d, r(no no no) && schedule info
                    max_time = torch.max(self.feature.feat_opes_batch[i, 4, :])
                    self.schedule.makespan_batch[i] = max_time
                else:
                    # machine feature changed
                    self.schedule.machines_batch[i, m_id, 1] += t  # fake proc time of Ope
                    self.schedule.machines_batch[i, m_id, 1] = max(self.schedule.machines_batch[i, m_id, 1], t_2)

                    # Update feature vectors of machines, available time & utilization
                    self.feature.feat_mas_batch[i, 1, m_id] = self.schedule.machines_batch[
                        i, m_id, 1]
                    utilization = self.schedule.machines_batch[i, :, 2]
                    cur_time = self.feature.feat_mas_batch[i, 1, :]
                    utilization = torch.minimum(utilization, cur_time)
                    utilization = utilization.div(cur_time + 1e-9)
                    self.feature.feat_mas_batch[i, 2, :] = utilization
                    # update env info d, r(no no no) && schedule info
                    max_time = torch.max(self.feature.feat_opes_batch[i, 4, :])
                    self.schedule.makespan_batch[i] = max_time

            self.main_info.update_main_pos(bool_position=bool)

    def validate_gantt(self):
        """
        Verify whether the schedule is feasible
        """
        ma_gantt_batch = [[[] for _ in range(self.instance.num_mas)] for __ in range(self.instance.batch_size)]
        for batch_id, schedules in enumerate(self.schedule.schedules_batch):
            for i in range(int(self.instance.nums_opes[batch_id])):
                step = schedules[i]
                ma_gantt_batch[batch_id][int(step[1])].append([i, step[2].item(), step[3].item()])
        proc_time_batch = self.instance.proc_times_batch

        # Check whether there are overlaps and correct processing times on the machine
        flag_proc_time = 0
        flag_ma_overlap = 0
        flag = 0
        for k in range(self.instance.batch_size):
            ma_gantt = ma_gantt_batch[k]
            proc_time = proc_time_batch[k]
            for i in range(self.instance.num_mas):
                ma_gantt[i].sort(key=lambda s: s[1])
                for j in range(len(ma_gantt[i])):  # j: 每个机器所作O的数量
                    if (len(ma_gantt[i]) <= 1) or (j == len(ma_gantt[i]) - 1):
                        break
                    if 0 > (ma_gantt[i][j][2] - ma_gantt[i][j + 1][1]) > -1e-4 or ma_gantt[i][j][2] > \
                            ma_gantt[i][j + 1][1]:  # 机器上某操作的完毕时间窗小于下一个开始
                        flag_ma_overlap += 1
                        print(f"Overlap:【machine{i}】【batch{k}】,"
                              f"【ope{ma_gantt[i][j][0]}】 time {ma_gantt[i][j][1]}-{ma_gantt[i][j][2]} while"
                              f"【ope{ma_gantt[i][j + 1][0]}】 start time {ma_gantt[i][j + 1][1]}-{ma_gantt[i][j + 1][2]}")
                    if ma_gantt[i][j][2] - ma_gantt[i][j][1] != proc_time[ma_gantt[i][j][0]][i]:
                        # flag_proc_time += 1
                        '''print(f"Wrong: processing time on【machine{i}】during {ma_gantt[i][j][2]} to {ma_gantt[i][j][1]},"
                              f"which should be {proc_time[ma_gantt[i][j][0]][i]}, 【batch{k}】【ope{ma_gantt[i][j][0]}】."
                              f"maintenance caused?")'''
                    flag += 1

        # Check job order and overlap
        flag_ope_overlap = 0
        for k in range(self.instance.batch_size):
            schedule_check = self.schedule.schedules_batch[k]
            nums_ope = self.instance.nums_ope_batch[k]
            num_ope_biases = self.instance.num_ope_biases_batch[k]
            for i in range(self.instance.num_jobs):
                if int(nums_ope[i]) <= 1:
                    continue
                for j in range(int(nums_ope[i]) - 1):
                    step = schedule_check[num_ope_biases[i] + j]
                    step_next = schedule_check[num_ope_biases[i] + j + 1]
                    if (step[3] - step_next[2]) > 0 or 1e-4 < (step[3] - step_next[2]) < 0:  # job got overlaps
                        flag_ope_overlap += 1
                        print(f"Overlap:【job{i}】【ope{j}】【batch{k}】 whose time{step[2]}-{step[3]} and "
                              f"{step_next[2]}-{step_next[3]}")
                        # 因为mask_job 过早释放导致派新工作
        # Check whether there are unscheduled operations
        flag_unscheduled = 0
        for batch_id, schedules in enumerate(self.schedule.schedules_batch):
            count = 0
            for i in range(schedules.size(0)):
                if schedules[i][0] == 1:
                    count += 1
            add = 0 if (count == self.instance.nums_opes[batch_id]) else 1
            flag_unscheduled += add

        if flag_ma_overlap + flag_ope_overlap + flag_proc_time + flag_unscheduled != 0:
            return False, self.schedule.schedules_batch
        else:
            return True, self.schedule.schedules_batch


class main_information():
    def __init__(self, mt):
        # e.g. [[m_id, start, end], [m_id, start, end]]
        if mt is not None:
            self.if_maintenance = True
        self.maintenance = mt
        self.count = len(mt)
        self.chart_main, self.machines_all, self.machines_pos_in_chart, self.machine_end_in_chart = self.df_maintenance(
            mt)
        # sort with m_id
        self.maintenance_plural = torch.tensor(self.maintenance)
        self.done_main = np.zeros(self.count, dtype=bool)
        # self.batch_idx = torch.tensor([i for i in range(batch_size)])

    def df_maintenance(self, mt):
        df = pd.DataFrame(mt, columns=['m_id', 'start', 'end'])
        for i in range(self.count):
            if df['start'][i] > df['end'][i]:
                t1 = df['start'][i]
                df['start'][i] = df['end'][i]
                df['end'][i] = t1
            # make sure t_1 < t_2
        list_df = df.values.tolist()
        self.maintenance = list_df
        df = df.sort_values(by=['m_id', 'start']).reset_index(drop=True)
        df2 = df.m_id.value_counts()
        machines_all = df2.index.values
        machines_numbers_in_chart = df2.values  # how many machine is maintained
        machines_pos_in_chart = [0]  # where this machine start
        for i in range(len(machines_all) - 1):
            now = machines_pos_in_chart[i]
            machines_pos_in_chart.append(machines_numbers_in_chart[i] + now)
        machines_end_in_chart = [machines_pos_in_chart[i] + machines_numbers_in_chart[i] for i in
                                 range(len(machines_all))]
        # where this machine end
        return df, machines_all, machines_pos_in_chart, machines_end_in_chart

    def update_main_pos(self, bool_position):
        for i in range(len(self.machines_all)):
            if bool_position[i]:
                if self.done_main[i]:
                    pass
                else:
                    self.machines_pos_in_chart[i] += 1
            self.done_main[i] = self.machines_pos_in_chart[i] >= self.machine_end_in_chart[i]
            # 判定结束有问题
        if self.done_main.all():
            self.if_maintenance = False

    def bool_whether_start(self, m_times):
        bool_start, info_use = [], []
        pos = 0
        for i in self.machines_all:
            if self.done_main[pos]:
                bool_start.append(False)
                pass
            else:
                time_m_idle = m_times[i]
                time_m_main_info = self.chart_main.loc[self.machines_pos_in_chart[pos]].values
                t_1, t_2 = time_m_main_info[1], time_m_main_info[2]
                if t_1 <= time_m_idle:
                    if self.done_main[pos]:
                        bool_start.append(False)
                    else:
                        bool_start.append(True)
                        info_use.append(time_m_main_info)
                else:
                    bool_start.append(False)
            pos += 1
        return bool_start, info_use


if __name__ == "__main__":
    device = "cpu"
    default_path = './case_studyB.fjs'
    env = MMBPEnv_Heu(path=default_path, batch=1, render_mode='p_d',
                      color_type=plt.cm.rainbow(np.linspace(0, 1, 50)),
                      device=device, maintenance=[[10, 4, 6], [3, 5, 25], [8, 20, 45]])

    torch.manual_seed(20199650253898)

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
            # print("action:", action)
            # print(action)
            # time
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
