# No changeover time
# Have maintenance window:
#   Rule_1 Machine offline and ongoing operation is paused (continue after maintenance)
#   Rule_2 The operation that is not started will be postponed to the end of the maintenance
#   Rule_3 The operation that is ongoing will be paused and continued after maintenance
#   Rule_4 The operation that is close to end (<= timeslot) will not be stopped, the maintenance will be postponed

import os
import gym
import copy
from MMBP.load_data_for_mmbp import instance
from MMBP.feat_ver_2 import feature
from MMBP.mask import mask
from MMBP.schedule import schedule
from MMBP.state_changeover import EnvState
from MMBP.render_modes import Instance_for_render
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import time


# default set
default_path = ['./Data_G_FJSP_Version/p11.fjs', './Data_G_FJSP_Version/p11.fjs']
PIC_SETTINGS = {'machine_name': '', 'job_name': 'Order', 'operation_name': 'Stage'}  # format of render
torch.manual_seed(20199650253898)


class MMBPEnv(gym.Env):
    """
    Multiproduct Multistage Batch Processes -- basic form as Flexible Job Shop Scheduling Problems
    Environment
    PATH: ['','',...] OR ''
    batch: None OR 1
    release_and_due: None OR pd.DataFrame
    device: 'cpu','cuda'
    render_mode: default"human", while training, make it None
    maintenance: None OR [[mid, start, end]]
    color_type: e.g. plt.cm.tab20(np.linspace(0,1,20))
    watching for debug: action which is processing
    """

    def __init__(self, time_slot=0.5, ins_file=None, batch=None, device="cpu",
                 release_and_due=None, relation_stage_unit=None,
                 maintenance=None,
                 render_mode="human", color_type=None, pic_settings=PIC_SETTINGS):
        """
        Initialize the environment
        time_slot: the time slot of the environment
        folder: the folder to save the results
        release_and_due: the release date and due date of each job
        if_due: whether the due date is given
        relation_stage_unit: the relation between stage and unit    type: pd.DataFrame
        render_mode: the mode to show the schedules
        device: the device to run the environment
        watching: the action which is processing
        reward_batch: the reward of each instance
        done: whether the environment is done
        instance: the instance of the environment - contain num_jobs, num_mas, num_opes, nums_opes, batch_size
        feature: the feature of the environment - contain feat_opes_batch, feat_mas_batch
        mask: the mask of the environment - contain mask_job_release_batch ,mask_job_procing_batch, mask_job_finish_batch, mask_ma_procing_batch, mask_maintenance_ma_batch
        schedule: the schedule of the environment - contain schedules_batch, ope_step_batch, makespan_batch, time, batch_idxes
        state: the state of the environment - many variables that should be monitored when applying the action
        old_state: the old state of the environment
        main_info: the maintenance information
        color_type: the color type of the environment
        pic_settings: the picture settings of the environment
        eli: the eligible matrix, size = (batch, num_mas, num_opes)
        """
        self.device = device
        self.watching = None
        self.time_slot = time_slot
        if ins_file is None:
            ins_file = default_path
        self.folder = self.make_folder(ins_file, batch)
        self.release, self.release_idx, self.due, self.release_matrix = None, None, None, None  # size = (num_jobs,)

        self.relation_stage_unit = relation_stage_unit
        self.render_mode = render_mode

        self.reward_batch = None
        self.done = False
        # build the instance, feature, mask, schedule, state, old_state
        ins = instance(ins_file, batch=batch, device=device)
        feat = feature(ins)
        mask_at_this_time = mask(batch_size=ins.batch_size, num_jobs=ins.num_jobs, num_mas=ins.num_mas)
        schedule_result = schedule(batch_size=ins.batch_size, num_opes=ins.num_opes, num_mas=ins.num_mas,
                                   num_ope_biases_batch=ins.num_ope_biases_batch, feat_opes_batch=feat.feat_opes_batch,
                                   mask_job_finish_batch=mask_at_this_time.mask_job_finish_batch)
        self.instance = ins
        self.feature = feat
        self.mask = mask_at_this_time
        self.schedule = schedule_result
        # maintenance time window
        self.main_info = maintenance_information(maintenance, batch_size=ins.batch_size) if maintenance is not None else None
        # update feat and mask because of due
        # Can be used after initialization of env
        self.if_rnd = False if release_and_due is None else True
        if self.if_rnd:
            self.set_release_and_due_of_job(release_and_due)
            if_no_eligible = self.if_no_eligible()  # 可能无已释放
            if 0 in if_no_eligible:
                self.time_move()
        else:
            # release all
            self.mask.mask_job_release_batch = torch.full_like(self.mask.mask_job_release_batch, True)
        self.eli = self.cal_eli()
        self.color_type = color_type if color_type is not None else plt.cm.rainbow(np.linspace(0, 1, self.instance.num_jobs))
        self.pic_settings = pic_settings
        # Uncompleted instances -> batch_idxes
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
                              mask_maintenance_ma_batch=self.mask.mask_maintenance_ma_batch,
                              mask_job_release_batch=self.mask.mask_job_release_batch)
        self.old_state = copy.deepcopy(self.state)

    def cal_eli(self):
        real_step_now = self.schedule.ope_step_batch
        ins_end_step = self.instance.end_ope_biases_batch
        ope_step = torch.where(real_step_now > ins_end_step, ins_end_step, real_step_now)
        op_proc_time = self.instance.proc_times_batch.gather(1, ope_step.unsqueeze(-1).
                                                             expand(-1, -1, self.instance.proc_times_batch.size(2)))
        ma_eligible = ~(self.mask.mask_ma_procing_batch + self.mask.mask_maintenance_ma_batch).unsqueeze(1).expand_as(
            op_proc_time)
        job_eligible = ~(~self.mask.mask_job_release_batch + self.mask.mask_job_procing_batch +
                         self.mask.mask_job_finish_batch)[:, :, None].expand_as(op_proc_time)
        self.eli = ma_eligible & job_eligible & (op_proc_time > 0).type(torch.bool)
        return self.eli

    def update_main_info(self, mt):
        """sometimes the main info will be added after the env is created, so we need to update it."""
        self.main_info = maintenance_information(mt, batch_size=self.instance.batch_size)

    def set_release_and_due_of_job(self, release_and_due, in_proc=False):
        """record release idx but don't change seq, keeping the initial sequence of r & d.
        release_and_due: pd.DataFrame, size = (num_jobs, 2) or numpy.array
        """
        if isinstance(release_and_due, str):
            release_and_due = pd.read_csv(release_and_due, header=None, index_col=None)
        if isinstance(release_and_due, pd.DataFrame):
            release_and_due = release_and_due.to_numpy()
        release_and_due = torch.Tensor(release_and_due)
        release = release_and_due[:, 0]
        due = release_and_due[:, 1]
        release_seq, idx = release.sort()
        # SELF.due keeps the initial index sequence
        self.release = release
        self.release_idx = idx
        self.due = due
        # job_2_ope
        release_matrix = release[self.instance.opes_appertain_batch]
        due_matrix = due[self.instance.opes_appertain_batch]
        # self.feature.feat_opes_batch[:, 4:6, :] += release_matrix
        # self.schedule.schedules_batch[:, :, 2:4] += release_matrix.reshape(1, -1, 1)
        self.feature.set_release_and_due_time(release_matrix, due_matrix)
        self.release_matrix = release[self.instance.opes_appertain_batch]
        if in_proc:
            self.if_rnd = True
            self.time_move()
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
                                  mask_maintenance_ma_batch=self.mask.mask_maintenance_ma_batch,
                                  mask_job_release_batch=self.mask.mask_job_release_batch)
            self.old_state = copy.deepcopy(self.state)
        _ = self.cal_eli()
        # if current time is already larger than some of the release dates
        if 0 in self.if_no_eligible():
            self.time_move()
        return True

    def make_folder(self, path, ins_batch):
        """make a folder to save charts and pictures"""
        # time_now = time.strftime('%Y%m%d_%H', time.localtime())
        if ins_batch == 1:
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
        self.main_info = maintenance_information(self.main_info.maintenance,
                                                 batch_size=self.instance.batch_size) if self.main_info is not None else None
        self.mask.reset_self()
        self.eli = self.cal_eli()
        return self.state

    def render(self, changeable_mode=None, name=None):
        """
        3 modes
            "human": (default) print schedules_batch
            "p_d": print table and draw
            "print_table": print table
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
        There are many mid_step parts are changing schedule clock,
            which are highlighted by ########### 'time_move'

        actions: sample:[[Ope],[Mas],[Job]]

        env step to next state
        give back obs
        """

        self.watching = actions
        opes = actions[0, :]
        mas = actions[1, :]
        jobs = actions[2, :]
        self.schedule.N += 1
        batch_idxes = self.schedule.batch_idxes

        # Removed unselected O-M arcs of the scheduled operations
        last_opes = torch.where(opes - 1 < self.instance.num_ope_biases_batch[batch_idxes, jobs],
                                self.instance.num_opes - 1, opes - 1)
        self.instance.update_oma_pt_cca(mas, opes, last_opes, self.schedule.batch_idxes)

        # Update for some O-M arcs are removed, such as 'Status', 'Number of neighboring machines' and 'Processing time'
        # Update 'Number of unscheduled operations in the job'
        proc_times, start_ope, end_ope = self.instance.proc_times_batch[batch_idxes, opes, mas], \
            self.instance.num_ope_biases_batch[batch_idxes, jobs], \
            self.instance.end_ope_biases_batch[batch_idxes, jobs]
        self.feature.update_fob_0123(opes, proc_times, start_ope, end_ope, self.schedule.batch_idxes)

        # Update 'Start time' and 'Job completion time'
        time_now_schedule = self.schedule.time  # start time will be the same as time_now_schedule
        cca, eob, oa = self.instance.cal_cumul_adj_batch[batch_idxes, :, :], \
            self.instance.end_ope_biases_batch[batch_idxes, :], \
            self.instance.opes_appertain_batch[batch_idxes, :]
        self.feature.update_fob_45(time_now_schedule, opes, cca, eob, oa, self.schedule.batch_idxes)

        # Update partial schedule (state)
        fob_st = self.feature.feat_opes_batch[batch_idxes, 5, :]
        fob_pt = self.feature.feat_opes_batch[batch_idxes, 2, :]
        self.schedule.update_static(opes, mas, fob_st, fob_pt, proc_times, jobs)

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

        # Check if there are still O-M pairs to be processed, otherwise the environment transits to the next time
        ########### 'time_move'
        self.time_move()
        ########### 'time_move'

        # Update the vector for uncompleted instances
        mask_finish = (self.schedule.N + 1) <= self.instance.nums_opes
        if ~(mask_finish.all()):
            self.schedule.batch_idxes = torch.arange(self.instance.batch_size)[mask_finish]

        # maintenance cause mask, mask after time point changed
        # Actually, it is an optinoal part before each action
        if self.main_info is not None:
            # in this part, m_times should be updated with schedule.time, if the clock has already moved after m_time
            while self.main_info.bool_whether_start(self.schedule.machines_batch[:,:,1]).any():
                ########### 'time_move'
                self.update_mask_maintenance_m(self.main_info.if_maintenance)
                ########### 'time_move'
                self.time_move()

            while self.main_info.bool_whether_end(self.schedule.time).any():
                ########### 'time_move'
                self.update_mask_maintenance_m(self.main_info.if_maintenance)
                ########### 'time_move'
                self.time_move()

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
                          time_batch=self.schedule.time,
                          mask_job_release_batch=self.mask.mask_job_release_batch)

        return self.state, self.reward_batch, self.schedule.done_batch, False, {}

    def time_move(self):
        """
        by self.next_time() to move time, until no eligible O-M pairs, error
        """
        flag_trans_2_next_time = self.if_no_eligible()
        time1 = time.time()
        while ~((~((flag_trans_2_next_time == 0) & (~self.schedule.done_batch))).all()):
            self.next_time(flag_trans_2_next_time)
            time2 = time.time()
            print('Time is Too Long') if time2-time1 >= 15 else None
            flag_trans_2_next_time = self.if_no_eligible()
            assert time2 - time1 <= 20, f'stay in loop, @time {self.schedule.time}, while doing {self.watching}'

    def time_move_to_t(self, target_time):
        """
        Only used in heuristic, schedule.t will be set to target_time
        """
        # TARTGET TIME: TENSOR OR LIST, same shape as schedule.T
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
        self.mask.update_with_time(target_batch_idxes, job_idxes, new_finish_job, if_finish, num_opes=self.instance.num_opes)
        if self.main_info is not None:
            self.mask.update_with_maintenance_time(self.schedule.time, self.main_info.maintenance,
                                                   self.schedule.stuck_time)

    def update_mask_maintenance_m(self, if_maintenance):
        """
        mask_maintenance_ma_batch, update stuck_job, stuck_time, feat_opes_batch, schedules_batch, feat_mas_batch...
        """
        if if_maintenance:  # When it is needed
            # First, detect whether we're in the time period
            # Note: If some job has already settled @ T_1, it will continue.
            m_time = self.schedule.machines_batch[:, :, 1]
            # whether walk through main, find start main point
            bool_start = self.main_info.bool_whether_start(m_time)
            if bool_start.any():
                batch_id_seq_to_main, m_id, t_1, t_2 = self.main_info.s_need(bool_start)
                m_id = m_id.long()
                batch_id_seq_to_main = batch_id_seq_to_main.long()
                self.mask.mask_maintenance_ma_batch[batch_id_seq_to_main, m_id] = True
                # make stuck job record
                m_available_time = self.schedule.machines_batch[batch_id_seq_to_main, m_id, 1]
                t = t_2 - t_1
                working_job = self.schedule.machines_batch[batch_id_seq_to_main, m_id, 3].long()

                # Rule_4 for Maintenance -- make self.already_maintenance different from self.main_info.maintenance
                bool_job_stuck = m_available_time - t_1 > self.time_slot    # Rule_4 of maintenance

                if bool_job_stuck.any():
                    # J not release / working ope, arrange job. Else, mask machine
                    working_ope = (self.schedule.ope_step_batch[batch_id_seq_to_main, working_job] - 1).long()
                    # next step of ope_step_batch，so -1
                    self.schedule.stuck_job[batch_id_seq_to_main, m_id] = working_job
                    self.schedule.stuck_time[batch_id_seq_to_main, m_id] = self.schedule.machines_batch[batch_id_seq_to_main, m_id, 1] - t_1

                    ##################### Problem 1
                    ##################### Main before ongoing Action
                    where_current_action_conflict = self.watching[2, batch_id_seq_to_main] == working_job

                    # current action is conflicting with maintenance
                    # Main before Action -- change start time of Action
                    batch_affected = t_1 <= self.schedule.schedules_batch[batch_id_seq_to_main, working_ope, 2]
                    use_id = batch_id_seq_to_main[where_current_action_conflict & batch_affected]
                    if use_id.any():
                        # use_id for Main before Action
                        dur = self.schedule.schedules_batch[use_id, working_ope, 3] - self.schedule.schedules_batch[use_id, working_ope, 2]
                        t = t_2 + dur - m_available_time
                        self.schedule.machines_batch[use_id, m_id, 1] = t_2 + dur
                        # self.feature.feat_opes_batch[i, 2, working_ope] don't need to change now
                        neighbor_ope_till_end_this_job = self.instance.end_ope_biases_batch[use_id, working_job]
                        self.feature.feat_opes_batch[use_id, 4, working_ope: neighbor_ope_till_end_this_job + 1] += t  # fob end time
                        self.feature.feat_opes_batch[use_id, 5, working_ope: neighbor_ope_till_end_this_job + 1] += t  # fob start time
                        # schedule s_batch
                        self.schedule.schedules_batch[use_id,
                        working_ope:neighbor_ope_till_end_this_job + 1, 2] += t
                        self.schedule.schedules_batch[use_id,
                        working_ope:neighbor_ope_till_end_this_job + 1, 3] += t
                    # Main after Action won't change the start time of Action
                    else:
                        self.schedule.machines_batch[batch_id_seq_to_main, m_id, 1] += t  # fake proc time of Ope
                        # the similar process as do an action
                        self.feature.feat_opes_batch[batch_id_seq_to_main, 2, working_ope] += t  # update fob_2 proc time
                        # influence fob of same job
                        neighbor_ope_till_end_this_job = self.instance.end_ope_biases_batch[batch_id_seq_to_main, working_job]
                        """if working_ope.size() is not torch.Size([1])"""
                        # only integer tensors of a single element can be converted to an index
                        if bool_job_stuck.shape[0] != 1:
                            for slice_idx in range(batch_id_seq_to_main.shape[0]):
                                # attention: mask_job_proc
                                # self.mask.mask_job_procing_batch[slice_idx, working_job[slice_idx]] = True
                                self.feature.feat_opes_batch[batch_id_seq_to_main[slice_idx], 4,
                                working_ope[slice_idx]:neighbor_ope_till_end_this_job[slice_idx] + 1] += t[
                                    slice_idx]  # fob end time
                                self.feature.feat_opes_batch[batch_id_seq_to_main[slice_idx], 5,
                                working_ope[slice_idx] + 1:neighbor_ope_till_end_this_job[slice_idx] + 1] += t[
                                    slice_idx]  # fob start time
                                # Update partial schedule (state), m_batch has already changed
                                self.schedule.schedules_batch[batch_id_seq_to_main[slice_idx], working_ope[slice_idx], 3] += t[slice_idx]
                                self.schedule.schedules_batch[batch_id_seq_to_main[slice_idx],working_ope[slice_idx] + 1:neighbor_ope_till_end_this_job[slice_idx] + 1, 2] \
                                    += t[slice_idx]
                                self.schedule.schedules_batch[batch_id_seq_to_main[slice_idx],working_ope[slice_idx] + 1:neighbor_ope_till_end_this_job[slice_idx] + 1, 3] \
                                    += t[slice_idx]
                        else:
                            self.feature.feat_opes_batch[batch_id_seq_to_main, 4,
                            working_ope: neighbor_ope_till_end_this_job + 1] += t  # fob end time
                            self.feature.feat_opes_batch[batch_id_seq_to_main, 5,
                            working_ope + 1: neighbor_ope_till_end_this_job + 1] += t  # fob start time
                            # schedule s_batch
                            self.schedule.schedules_batch[batch_id_seq_to_main, working_ope, 3] += t
                            self.schedule.schedules_batch[batch_id_seq_to_main,
                            working_ope + 1:neighbor_ope_till_end_this_job + 1, 2] += t
                            self.schedule.schedules_batch[batch_id_seq_to_main,
                            working_ope + 1:neighbor_ope_till_end_this_job + 1, 3] += t

                    # Update feature vectors of machines, available time & utiliz
                    self.feature.feat_mas_batch[batch_id_seq_to_main, 1, m_id] = self.schedule.machines_batch[
                        batch_id_seq_to_main, m_id, 1]
                    utiliz = self.schedule.machines_batch[batch_id_seq_to_main, :, 2]
                    cur_time = self.schedule.time[batch_id_seq_to_main, None].expand_as(utiliz)
                    utiliz = torch.minimum(utiliz, cur_time)
                    utiliz = utiliz.div((self.schedule.time[batch_id_seq_to_main] + 1e-9).unsqueeze(-1))

                    self.feature.feat_mas_batch[batch_id_seq_to_main, 2, :] = utiliz
                    # update env info d, r(no no no) && schedule info
                    max_time = torch.max(self.feature.feat_opes_batch[batch_id_seq_to_main, 4, :])
                    self.schedule.makespan_batch[batch_id_seq_to_main] = max_time

                else:   # No job is stuck by Rule_4

                    # 只改变machine相关特征，不动job
                    self.schedule.machines_batch[batch_id_seq_to_main, m_id, 1] += t  # fake proc time of Ope

                    # Update feature vectors of machines, available time & utiliz
                    self.feature.feat_mas_batch[batch_id_seq_to_main, 1, m_id] = self.schedule.machines_batch[
                        batch_id_seq_to_main, m_id, 1]
                    utiliz = self.schedule.machines_batch[batch_id_seq_to_main, :, 2]
                    cur_time = self.schedule.time[batch_id_seq_to_main, None].expand_as(utiliz)
                    utiliz = torch.minimum(utiliz, cur_time)
                    utiliz = utiliz.div((self.schedule.time[batch_id_seq_to_main] + 1e-9).unsqueeze(-1))
                    self.feature.feat_mas_batch[batch_id_seq_to_main, 2, :] = utiliz
                    # update env info d, r(no no no) && schedule info
                    max_time = torch.max(self.feature.feat_opes_batch[batch_id_seq_to_main, 4, :])
                    self.schedule.makespan_batch[batch_id_seq_to_main] = max_time

                    mask_finish = (self.schedule.N + 1) <= self.instance.nums_opes
                    if ~(mask_finish.all()):
                        self.schedule.batch_idxes = torch.arange(self.instance.batch_size)[
                            mask_finish]

                    ########### 'time_move'
                    self.time_move()
                mask_finish = (self.schedule.N + 1) <= self.instance.nums_opes
                if ~(mask_finish.all()):
                    self.schedule.batch_idxes = torch.arange(self.instance.batch_size)[
                        mask_finish]

                ########### 'time_move'
                self.time_move()

            bool_end = self.main_info.bool_whether_end(self.schedule.time)
            if bool_end.any():
                bid, m_ids, t_end = self.main_info.e_need(bool_end)
                m_ids = m_ids.long()
                bid = bid.long()
                self.mask.mask_maintenance_ma_batch[bid, m_ids] = False
                self.schedule.stuck_job[bid, m_ids] = -1
                self.schedule.stuck_time[bid, m_ids] = 0

    def if_no_eligible(self):
        """
        Check if there are still O-M pairs to be processed
        1 procing is not eli
        2 maintenance is not eli
        3 release date is not eli   # new part, conducting ##############

        """
        eli = self.cal_eli()
        flag_trans_2_next_time = torch.sum(eli.transpose(1, 2), dim=[1, 2])
        # shape: (batch_size)
        # An element value of 0 means that the corresponding instance has no eligible O-M pairs
        # in other words, the environment need to transit to the next time
        if self.main_info is not None:
            flag_trans_2_next_time += self.main_info.bool_whether_start(self.schedule.machines_batch[:, :, 1])
        return flag_trans_2_next_time

    def release_in_next_time_func(self):
        # new release job
        if self.if_rnd:
            # when due is on
            flag_release = self.release.expand_as(self.mask.mask_job_release_batch) <= self.schedule.time.unsqueeze(
                dim=1)
            self.mask.mask_job_release_batch = flag_release
            # release new job - now it can be detected

    def next_time(self, flag_trans_2_next_time):
        """This function is used in every #time_move#
        1 maintenance_time_chart change
        2 release date change       # new part, conducting ##############
        3 valid change after each new time point
        """
        # need transit
        flag_need_trans = (flag_trans_2_next_time == 0) & (~self.schedule.done_batch)
        batch_idx_using = torch.arange(self.instance.batch_size)
        # pos_job_len = self.schedule.ope_step_batch - 1  # size: (batch_size, job)
        # Pos is not the out time of former operation, when it is the 1st Ope of job.
        # So it is excluded by comparing with the num_opes_biases_batch
        pos_job_len = torch.where(self.schedule.ope_step_batch > self.instance.num_ope_biases_batch,
                                  self.schedule.ope_step_batch - 1, self.schedule.ope_step_batch)
        pos_batch_idx = batch_idx_using.unsqueeze(dim=0).T.expand(pos_job_len.size())  # (batch_size, job)
        # batch_idxes < batch_size, reshape
        job_feat = torch.where(self.schedule.ope_step_batch > self.instance.num_ope_biases_batch,
                               self.schedule.schedules_batch[pos_batch_idx, pos_job_len, 3],
                               self.schedule.schedules_batch[pos_batch_idx, pos_job_len, 2])
        # job end time, (batch_size, job)

        # Job
        if self.if_rnd and ~self.mask.mask_job_release_batch.all():
            # There are jobs that have not been released
            # 1 release later - release time
            un_release = torch.where(~self.mask.mask_job_release_batch, self.release, 1e5)
            job_min_release_later = torch.min(un_release, dim=1)[0]
            # 2 Already released jobs - out time
            job_already_release_out_time = torch.where(self.mask.mask_job_release_batch, job_feat, job_feat.max() + self.time_slot)    # 选择out time到已经释放的mask
            # min of release time and out time
            job_already_release_out_time = torch.where(job_already_release_out_time > self.schedule.time[:, None],
                                                       job_already_release_out_time,
                                                       self.schedule.time[:, None] + 1e5)
            job_ava_time = torch.min(job_already_release_out_time, dim=1)[0]
            j_min_avail_later = torch.min(job_min_release_later, job_ava_time)  # 统一到该值, 两者取小
        else:
            # all jobs have been released
            j_next_all = torch.where(job_feat > self.schedule.time[:,None], job_feat, job_feat.max() + 1)
            # [:,None] broadcast row->column
            j_min_avail_later = torch.min(j_next_all, dim=1)[0]

        # Machine
        # If masked, update schedule.machines_batch -> completion time
        m_available_time = self.schedule.machines_batch[:, :, 1]  # available_time of machines -- from maintenance -- 1
        # remain available_time greater than current time
        m_avail_time_later = torch.where(m_available_time > self.schedule.time[:, None], m_available_time,
                                         m_available_time.max()+1e5)  # in FJSP it is >
        # m_avail_time.max() might be zero, so add 1e5

        # Return the minimum value of available_time (the time to transit to)
        m_min_avail_later = torch.min(m_avail_time_later, dim=1)[0]

        # The time for each batch to transit to or stay in
        # 这里还需要考虑，用的job avail later其实是release later 而不是已经release的job可能空闲的时间
        # 若m_min_avail_later 很大说明m全都释放
        target_time = torch.where(flag_need_trans, torch.min(j_min_avail_later, m_min_avail_later),
                                  self.schedule.time)
        if target_time[flag_need_trans].any() >=1e4:
            print('Target time is Too Big, target_time > 1e4')
        if len(target_time[flag_need_trans]) == 1:
            if target_time[flag_need_trans] >= 1e4:
                print('target time is a scalar')
                _ = self.cal_eli()

        if (target_time[flag_need_trans] == self.schedule.time[flag_need_trans]).any():
            if (target_time[flag_need_trans] == 0).all():
                pass
            else:
                # print('There is problem about stucking in time_move, target_time == self.schedule.time')
                pass

        # print(f'@time {self.schedule.time}, target_time {target_time}', end='\t')

        # Detect the machines & job that completed (at above time), 1: idle
        m_idle_at_target_time_later = torch.where(
            (m_available_time <= target_time[:, None]) & (
                    self.schedule.machines_batch[:, :, 0] == 0) & flag_need_trans[:, None],
            True, False)

        # 将时间重置到c得到的当前最小空闲时间——容易导致错误
        self.schedule.time[flag_need_trans] = target_time[flag_need_trans]      # 重置时间，不要忘记加flag
        if self.if_rnd:
            self.release_in_next_time_func() if ~self.mask.mask_job_release_batch.all() else None# release job ############## 改变mask

        # 以下是更新时间后需要进行的步骤
        # Update partial schedule (state), variables and feature vectors
        aa = self.schedule.machines_batch.transpose(1, 2)
        aa[m_idle_at_target_time_later, 0] = 1
        self.schedule.machines_batch = aa.transpose(1, 2)

        utiliz = self.schedule.machines_batch[:, :, 2]
        cur_time = self.schedule.time[:, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.schedule.time[:, None] + 1e-5)
        self.feature.feat_mas_batch[:, 2, :] = utiliz

        # jobs 可能有一个batch没有正数?
        # 将已经执行完的job释放出来，m_b[3]: 正在执行的job_idx, m_idle: 空m_idx
        jobs = torch.where(m_idle_at_target_time_later, self.schedule.machines_batch[:, :, 3].double(), -1.0).float()
        jobs_index = np.argwhere(jobs.cpu() >= 0)  # .to(self.device)  # size(batch_size, job_num) 目前空出来的job的坐标
        job_idxes = jobs[jobs_index[0], jobs_index[1]].long()  # 从以上坐标拿出序号
        target_batch_idxes = jobs_index[0]

        if_finish = self.schedule.ope_step_batch == self.instance.end_ope_biases_batch + 1
        self.mask.update_with_time(target_batch_idxes, job_idxes, m_idle_at_target_time_later, if_finish,
                                   num_opes=self.instance.num_opes)
        """#############只做了job相关没有管mas"""

        if self.main_info is not None:
            self.mask.update_with_maintenance_time(self.schedule.time, self.main_info.maintenance,
                                                   self.schedule.stuck_time)

    def validate_gantt(self):
        '''
        Verify whether the schedule is feasible
        '''
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
                    if ma_gantt[i][j][2] > ma_gantt[i][j + 1][1]:  # 机器上某操作的完毕时间窗小于下一个开始
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
            schedule = self.schedule.schedules_batch[k]
            nums_ope = self.instance.nums_ope_batch[k]
            num_ope_biases = self.instance.num_ope_biases_batch[k]
            for i in range(self.instance.num_jobs):
                if int(nums_ope[i]) <= 1:
                    continue
                for j in range(int(nums_ope[i]) - 1):
                    step = schedule[num_ope_biases[i] + j]
                    step_next = schedule[num_ope_biases[i] + j + 1]
                    if step[3] > step_next[2]:  # job got overlaps
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


class maintenance_information():
    def __init__(self, mt, batch_size):
        # e.g. [[m_id, start, end], [m_id, start, end]]
        if mt is not None:
            self.if_maintenance = True
        self.maintenance = mt
        self.count = len(mt)
        self.chart_main, self.start_t_main_sequel, self.end_t_main_sequel = self.df_maintenance(mt)
        self.start_t_main_sequel = self.start_t_main_sequel.expand(batch_size, self.count)
        self.end_t_main_sequel = self.end_t_main_sequel.expand(batch_size, self.count)
        self.maintenance_plural = torch.tensor(self.maintenance).expand(batch_size, self.count, 3)
        self.main_position = torch.zeros(size=(batch_size, 2), dtype=torch.long)
        self.done_main = torch.zeros(size=(batch_size, 2), dtype=torch.bool)
        self.batch_idx = torch.tensor([i for i in range(batch_size)])

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
        df = df.sort_values(by=['start'])
        start_time_index = df.index.values
        df = df.sort_values(by=['end'])
        end_time_index = df.index.values
        return df, torch.tensor(start_time_index), torch.tensor(end_time_index)

    def update_main_pos(self, bool_position):
        self.main_position[bool_position] += 1
        self.done_main = self.main_position == self.count

    def bool_whether_start(self, m_time):
        """Which maintenance is needed to start at this time point?"""
        # m_time: (batch_size, machine_num)
        if self.done_main[:, 0].any():
            bool_pos_for_s = ~self.done_main[:, 0]
            b_clone = bool_pos_for_s.clone()
            gnmst = self.start_t_main_sequel[bool_pos_for_s, self.main_position[bool_pos_for_s, 0]]
            ss = self.maintenance_plural[bool_pos_for_s, gnmst]
            bool_pos_for_s[b_clone] = ss[:, 1] <= m_time[b_clone, ss[:, 0].long()]
        else:
            get_next_main_s_t = self.start_t_main_sequel[self.batch_idx, self.main_position[:, 0]]
            ss = self.maintenance_plural[self.batch_idx, get_next_main_s_t]
            bool_pos_for_s = ss[:, 1] <= m_time[self.batch_idx, ss[:, 0].long()]
        return bool_pos_for_s

    def bool_whether_end(self, time):
        if self.done_main[:, 1].any():
            bool_pos_for_e = ~self.done_main[:, 1]
            b_clone = bool_pos_for_e.clone()
            gnmet = self.end_t_main_sequel[bool_pos_for_e, self.main_position[bool_pos_for_e, 1]]
            ee = self.maintenance_plural[bool_pos_for_e, gnmet]
            bool_pos_for_e[b_clone] = ee[:, 1] <= time[b_clone]
        else:
            position = self.main_position[:, 1]
            get_next_main_e_t = self.end_t_main_sequel[self.batch_idx, position]
            ee = self.maintenance_plural[self.batch_idx, get_next_main_e_t]
            bool_pos_for_e = ee[:, 2] <= time
        return bool_pos_for_e

    def s_need(self, bool_s):
        get_pos_in_sequel = self.main_position[bool_s, 0]
        get_pos_in_plural = self.start_t_main_sequel[bool_s, get_pos_in_sequel]
        m_info_for_bools = self.maintenance_plural[bool_s, get_pos_in_plural]
        update_pos = torch.stack([bool_s, torch.zeros_like(bool_s)], dim=0)
        self.update_main_pos(update_pos.T)
        # b_idx, m_ids, t_1, t_2
        return self.batch_idx[bool_s], m_info_for_bools[:, 0], m_info_for_bools[:, 1], m_info_for_bools[:, 2]

    def e_need(self, bool_e):
        get_pos_in_sequel = self.main_position[bool_e, 1]
        get_pos_in_plural = self.end_t_main_sequel[bool_e, get_pos_in_sequel]
        m_info_for_bool_e = self.maintenance_plural[bool_e, get_pos_in_plural]
        update_pos = torch.stack([torch.zeros_like(bool_e), bool_e], dim=0)
        self.update_main_pos(update_pos.T)
        # b_idx, m_ids, t_2
        return self.batch_idx[bool_e], m_info_for_bool_e[:, 0], m_info_for_bool_e


if __name__ == "__main__":
    batch = 2
    default_path = 'case_studyA.fjs'
    rnd = './case_studyA.csv'

    device = "cuda"
    env = MMBPEnv(device=device, batch=batch, ins_file=default_path, release_and_due=rnd, # relation_stage_unit=rsu,
                  render_mode='p_d', time_slot=1,
                  maintenance=[[10,4,6],[0,0.5,1.5]])
    # print(env.instance.num_jobs)
    state = env.state
    dones = env.schedule.done_batch
    done = False  # Unfinished at the beginning
    i = 0

    # j = env.get_job_num(ope=10)

    last_time = time.time()

    from MMBP.Heuristics import random_method
    while ~done:
        i += 1
        eli = env.eli
        action = random_method(eli, ope_step_batch=env.state.ope_step_batch, batch_idxes=env.state.batch_idxes)
        # print(action)
        state, rewards, dones, _, _ = env.step(action)
        done = dones.all()
        env.render(changeable_mode='draw')

    spend_time = time.time() - last_time  # The time taken to solve this environment (instance)
    env.render(name='CASE_A')
    result_correct, _ = env.validate_gantt()

    print("spend_time: %.2f" % spend_time, "result correct:", result_correct)
