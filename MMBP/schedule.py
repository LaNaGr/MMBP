import torch
import copy


class schedule:
    def __init__(self, batch_size, num_opes, num_mas, num_ope_biases_batch, feat_opes_batch, mask_job_finish_batch,
                 device='cpu'):
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
            id_job! not ope!
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
        self.machines_batch[:, :, 0] = torch.ones(size=(batch_size, num_mas))
        self.machines_batch[:, :, 3] = torch.ones(size=(batch_size, num_mas)) * -1

        self.ope_step_batch = copy.deepcopy(num_ope_biases_batch)  # shape: (batch_size, num_jobs)
        # the id of the current operation (be waiting to be processed) of each job

        # dynamic variable
        self.batch_idxes = torch.arange(batch_size, device=device)  # Uncompleted instances
        self.time = torch.zeros(batch_size, device=device)  # Current time of the environment
        self.N = torch.zeros(batch_size, device=device).int()  # Count scheduled operations
        self.makespan_batch = torch.max(feat_opes_batch[:, 4, :], dim=1)[0]  # shape: (batch_size)
        self.done_batch = mask_job_finish_batch.all(dim=1)  # shape: (batch_size)
        self.stuck_job = (torch.ones(batch_size, num_mas, device=device) * -1).long()
        self.stuck_time = torch.zeros(batch_size, num_mas, device=device)

        self.old_makespan = torch.max(feat_opes_batch[:, 4, :], dim=1)[0]

    def reset_self(self):
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4), device=self.device)
        self.schedules_batch[:, :, 2] = self.feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = self.feat_opes_batch[:, 5, :] + self.feat_opes_batch[:, 2, :]
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4), device=self.device)
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas),device=self.device)
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)

        self.time = torch.zeros(self.batch_size, device=self.device)
        self.N = torch.zeros(self.batch_size, device=self.device).int()
        self.makespan_batch = copy.deepcopy(self.old_makespan)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        self.stuck_job = (torch.ones(self.batch_size, self.num_mas, device=self.device) * -1).long()
        self.stuck_time = torch.zeros(self.batch_size, self.num_mas, device=self.device)
        self.batch_idxes = torch.arange(self.batch_size, device=self.device)

    def update_static(self, opes, mas, fob_st, fob_pt, proc_times, jobs):
        self.schedules_batch[self.batch_idxes, opes, :2] = torch.stack((torch.ones(self.batch_idxes.size(0)), mas),
                                                                       dim=1)
        self.schedules_batch[self.batch_idxes, :, 2] = fob_st
        self.schedules_batch[self.batch_idxes, :, 3] = fob_st + fob_pt
        self.machines_batch[self.batch_idxes, mas, 0] = torch.zeros(self.batch_idxes.size(0))

        # self.time + porc_times 与 fob_st + fob_pt 不一样，可能是因为时间self.time更新的问题
        self.machines_batch[self.batch_idxes, mas, 1] = self.schedules_batch[self.batch_idxes, opes, 3]## self.time[self.batch_idxes] + proc_times
        self.machines_batch[self.batch_idxes, mas, 2] += proc_times
        self.machines_batch[self.batch_idxes, mas, 3] = jobs.float()

        self.ope_step_batch[self.batch_idxes, jobs] += 1

    def update_info(self, done_batch, max_now):
        self.done_batch = done_batch
        self.makespan_batch = max_now
