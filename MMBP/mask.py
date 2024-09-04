import torch


class mask:
    def __init__(self, batch_size, num_jobs, num_mas):
        # Masks of current status, dynamic
        self.batch_size, self.num_jobs, self.num_mas = batch_size, num_jobs, num_mas
        self.mask_job_release_batch = torch.full(size=(batch_size, num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_job_procing_batch = torch.full(size=(batch_size, num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_job_finish_batch = torch.full(size=(batch_size, num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_ma_procing_batch = torch.full(size=(batch_size, num_mas), dtype=torch.bool, fill_value=False)
        self.mask_maintenance_ma_batch = torch.full(size=(batch_size, num_mas), dtype=torch.bool, fill_value=False)
        # job maintenance直接东job proc

    def reset_self(self):
        # release
        self.mask_job_release_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool,
                                                 fill_value=False)
        # mask for job, shape: (batch_size, num_jobs), True for jobs in process
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool,
                                                 fill_value=False)
        # mask for job, shape: (batch_size, num_jobs), True for completed jobs
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool,
                                                fill_value=False)
        # mask for machine, shape: (batch_size, num_mas), True for machines in process
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool,
                                                fill_value=False)
        self.mask_maintenance_ma_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool, fill_value=False)

    def update_with_step(self, jobs, mas, ope_step_batch, end_ope_biases_batch, batch_idxes):
        self.mask_job_procing_batch[batch_idxes, jobs] = True
        self.mask_ma_procing_batch[batch_idxes, mas] = True
        self.mask_job_finish_batch = torch.where(ope_step_batch == end_ope_biases_batch + 1, True,
                                                 self.mask_job_finish_batch)

    def update_with_mid_step_release(self):
        # step中途由于时间移动，新的order释放
        pass

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
        if self.mask_maintenance_ma_batch.any():    # else不用改mask
            which_has_been_mask = torch.nonzero(self.mask_maintenance_ma_batch) # shape: (num, [batch_idx, m_id])
            for main_info_num in range(len(maintenance_info)):  # 对于每一个维护窗
                m_id = maintenance_info[main_info_num][0]
                position_in_whbm_of_this_m = torch.nonzero(which_has_been_mask[:, 1] == m_id)  # m_id在whbm的索引, 可能有多个， shape: (num, 1)
                for i in range(position_in_whbm_of_this_m.shape[0]):
                    batch_idx = which_has_been_mask[position_in_whbm_of_this_m[i]][-1, 0]
                    cur_time_for_this_batch = current_time[batch_idx]
                    if cur_time_for_this_batch >= maintenance_info[main_info_num][2]+stuck_time[batch_idx,int(m_id)]:   # else 没到时间，不做改变
                        self.mask_maintenance_ma_batch[batch_idx, int(m_id)] = False
        pass