import torch
import numpy as np
import copy

from torch import Tensor

"""
read data files

data structure:
    instance
    feature
    mask
    schedule
"""


class mask:
    def __init__(self, batch_size, num_jobs, num_mas, device='cpu'):
        # Masks of current status, dynamic
        self.device=device
        self.batch_size, self.num_jobs, self.num_mas = batch_size, num_jobs, num_mas
        self.mask_job_procing_batch = torch.full(size=(batch_size, num_jobs), dtype=torch.bool, fill_value=False, device=device)
        self.mask_job_finish_batch = torch.full(size=(batch_size, num_jobs), dtype=torch.bool, fill_value=False, device=device)
        self.mask_ma_procing_batch = torch.full(size=(batch_size, num_mas), dtype=torch.bool, fill_value=False, device=device)
        self.mask_maintenance_ma_batch = torch.full(size=(batch_size, num_mas), dtype=torch.bool, fill_value=False, device=device)
        # job maintenance直接东job proc

    def reset_self(self):
        # mask for job, shape: (batch_size, num_jobs), True for jobs in process
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool,
                                                 fill_value=False, device=self.device)
        # mask for job, shape: (batch_size, num_jobs), True for completed jobs
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool,
                                                fill_value=False, device=self.device)
        # mask for machine, shape: (batch_size, num_mas), True for machines in process
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool,
                                                fill_value=False, device=self.device)
        self.mask_maintenance_ma_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool,
                                                    fill_value=False, device=self.device)

    def update_with_step(self, jobs, mas, ope_step_batch, end_ope_biases_batch, batch_idxes):
        self.mask_job_procing_batch[batch_idxes, jobs] = True
        self.mask_ma_procing_batch[batch_idxes, mas] = True
        self.mask_job_finish_batch = torch.where(ope_step_batch == end_ope_biases_batch + 1, True,
                                                 self.mask_job_finish_batch)

    def update_with_time(self, target_batch_idxes, job_idxes, mas_position, if_finish, num_opes=None):
        # 可能导致job无法释放？注意加上mask_main之后; 维护后还有一段工作未完，导致job提前释放！！！！！！！！！！！
        self.mask_ma_procing_batch[mas_position] = False
        self.mask_job_finish_batch = torch.where(if_finish, True, self.mask_job_finish_batch)
        if num_opes is not None:
            if ~(job_idxes - num_opes).any():
                pass
            else:
                self.mask_job_procing_batch[target_batch_idxes, job_idxes] = False

    def update_with_maintenance_time(self, current_time, maintenance_info, stuck_time):
        # 由于时间移动，已经可以释放mask_maintenance_ma
        if self.mask_maintenance_ma_batch.any():  # else不用改mask
            which_has_been_mask = torch.nonzero(self.mask_maintenance_ma_batch)  # shape: (num, [batch_idx, m_id])
            for main_info_num in range(len(maintenance_info)):  # 对于每一个维护窗
                m_id = maintenance_info[main_info_num][0]
                position_in_whbm_of_this_m = torch.nonzero(
                    which_has_been_mask[:, 1] == m_id)  # m_id在whbm的索引, 可能有多个， shape: (num, 1)
                for i in range(position_in_whbm_of_this_m.shape[0]):
                    batch_idx = which_has_been_mask[position_in_whbm_of_this_m[i]][-1, 0]
                    cur_time_for_this_batch = current_time[batch_idx]
                    if cur_time_for_this_batch >= maintenance_info[main_info_num][2] + stuck_time[
                        batch_idx, m_id]:  # else 没到时间，不做改变
                        self.mask_maintenance_ma_batch[batch_idx, m_id] = False
        pass


class schedule:
    time: Tensor

    def __init__(self, batch_size, num_opes, num_mas, num_ope_biases_batch, feat_opes_batch, mask_job_finish_batch, device='cpu'):
        """
        Partial Schedule (state) of jobs/operations, dynamic
            Status
            Allocated machines
            Start time
            End time
        Partial Schedule (state) of machines, dynamic
            idle
            available_time
            utilization_time
            id_ope
        now
            ope_step_batch
            batch_idxes
            time
            N
            makespan
            done
        maintenane
            stuck_job
            stuck_time
        """
        self.device = device
        self.batch_size, self.num_opes, self.num_mas = batch_size, num_opes, num_mas
        self.num_ope_biases_batch = num_ope_biases_batch
        self.feat_opes_batch, self.mask_job_finish_batch = feat_opes_batch, mask_job_finish_batch

        self.schedules_batch = torch.zeros(size=(batch_size, num_opes, 4), device=device)
        self.schedules_batch[:, :, 2] = feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = feat_opes_batch[:, 5, :] + feat_opes_batch[:, 2, :]

        self.machines_batch = torch.zeros(size=(batch_size, num_mas, 4), device=device)
        self.machines_batch[:, :, 0] = torch.ones(size=(batch_size, num_mas), device=device)

        self.ope_step_batch = copy.deepcopy(num_ope_biases_batch)  # shape: (batch_size, num_jobs)
        # the id of the current operation (be waiting to be processed) of each job

        # dynamic variable
        self.batch_idxes = torch.arange(batch_size, device=device)  # Uncompleted instances
        self.time = torch.zeros(batch_size, device=device)  # Current time of the environment
        self.N = torch.zeros(batch_size).int()  # Count scheduled operations
        self.makespan_batch = torch.max(feat_opes_batch[:, 4, :], dim=1)[0]  # shape: (batch_size)
        self.done_batch = mask_job_finish_batch.all(dim=1)  # shape: (batch_size)
        self.stuck_job = torch.ones(batch_size, num_mas, device=device) * -1
        self.stuck_time = torch.zeros(batch_size, num_mas, device=device)

        self.old_makespan = torch.max(feat_opes_batch[:, 4, :], dim=1)[0]

    def reset_self(self):
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4), device=self.device)
        self.schedules_batch[:, :, 2] = self.feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = self.feat_opes_batch[:, 5, :] + self.feat_opes_batch[:, 2, :]
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4), device=self.device)
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas), device=self.device)
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)

        self.time = torch.zeros(self.batch_size, device=self.device)
        self.N = torch.zeros(self.batch_size, device=self.device).int()
        self.makespan_batch = copy.deepcopy(self.old_makespan)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        self.stuck_job = torch.ones(self.batch_size, self.num_mas, device=self.device) * -1
        self.stuck_time = torch.zeros(self.batch_size, self.num_mas, device=self.device)
        self.batch_idxes = torch.arange(self.batch_size, device=self.device)

    def update_static(self, opes, mas, fob_st, fob_pt, proc_times, jobs, time_now):
        self.schedules_batch[self.batch_idxes, opes, :2] = torch.stack((torch.ones(self.batch_idxes.size(0), device=proc_times.device
                                                                                   ), mas),
                                                                       dim=1)
        self.schedules_batch[self.batch_idxes, :, 2] = fob_st
        self.schedules_batch[self.batch_idxes, :, 3] = fob_st + fob_pt
        self.machines_batch[self.batch_idxes, mas, 0] = torch.zeros(self.batch_idxes.size(0), device=proc_times.device)
        self.machines_batch[self.batch_idxes, mas, 1] = time_now + proc_times
        self.machines_batch[self.batch_idxes, mas, 2] += proc_times
        self.machines_batch[self.batch_idxes, mas, 3] = jobs.float()

        self.ope_step_batch[self.batch_idxes, jobs] += 1

    def update_info(self, done_batch, max_now):
        self.done_batch = done_batch
        self.makespan_batch = max_now


if __name__ == "__main__":
    ins = instance(['../Data/Mk01_v3.fjs', '../Data/Mk03.fjs'])  # default path
    feat = feature(ins)
    mask_at_this_time = mask(batch_size=ins.batch_size, num_jobs=ins.num_jobs, num_mas=ins.num_mas)
    schedule_result = schedule(batch_size=ins.batch_size, num_opes=ins.num_opes, num_mas=ins.num_mas,
                               num_ope_biases_batch=ins.num_ope_biases_batch, feat_opes_batch=feat.feat_opes_batch,
                               mask_job_finish_batch=mask_at_this_time.mask_job_finish_batch)
    print()
