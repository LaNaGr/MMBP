import torch
import copy


class feature:
    def __init__(self, instances):
        """
        features, dynamic
            ope:
                Status
                Number of neighboring machines
                Processing time
                Number of unscheduled operations in the job
                Job completion time
                Start time
            ma:
                Number of neighboring operations
                Available time
                Utilization
        """
        # prepare data
        self.instance = instances
        self.feat_paras = {"ope_feat_dim": 6, "ma_feat_dim": 3}

        # read from instance
        batch_size = self.instance.batch_size
        num_opes = self.instance.num_opes
        num_mas = self.instance.num_mas
        ope_ma_adj_batch = self.instance.ope_ma_adj_batch
        proc_times_batch = self.instance.proc_times_batch
        nums_ope_batch = self.instance.nums_ope_batch
        cal_cumul_adj_batch = self.instance.cal_cumul_adj_batch
        end_ope_biases_batch = self.instance.end_ope_biases_batch
        opes_appertain_batch = self.instance.opes_appertain_batch

        # Generate raw feature vectors
        feat_opes_batch = torch.zeros(size=(batch_size, self.feat_paras["ope_feat_dim"], num_opes))
        feat_mas_batch = torch.zeros(size=(batch_size, self.feat_paras["ma_feat_dim"], num_mas))

        feat_opes_batch[:, 1, :] = torch.count_nonzero(ope_ma_adj_batch, dim=2)
        feat_opes_batch[:, 2, :] = torch.sum(proc_times_batch, dim=2).div(feat_opes_batch[:, 1, :] + 1e-9)
        feat_opes_batch[:, 3, :] = self.convert_feat_job_2_ope(nums_ope_batch, opes_appertain_batch)
        feat_opes_batch[:, 5, :] = torch.bmm(feat_opes_batch[:, 2, :].unsqueeze(1), cal_cumul_adj_batch).squeeze()
        end_time_batch = (feat_opes_batch[:, 5, :] +
                          feat_opes_batch[:, 2, :]).gather(1, end_ope_biases_batch)
        feat_opes_batch[:, 4, :] = self.convert_feat_job_2_ope(end_time_batch, opes_appertain_batch)
        feat_mas_batch[:, 0, :] = torch.count_nonzero(ope_ma_adj_batch, dim=1)

        # generate feat
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch

        # to reset
        self.old_feat_opes_batch = copy.deepcopy(self.feat_opes_batch)
        self.old_feat_mas_batch = copy.deepcopy(self.feat_mas_batch)

    def convert_feat_job_2_ope(self, feat_job_batch, opes_appertain_batch):
        """
        Convert job features into operation features (such as dimension)
        """
        return feat_job_batch.gather(1, opes_appertain_batch)

    def reset_self(self):
        self.feat_opes_batch = copy.deepcopy(self.old_feat_opes_batch)
        self.feat_mas_batch = copy.deepcopy(self.old_feat_mas_batch)

    def update_fob_0123(self, opes, proc_times, start_ope, end_ope, batch_idxes):
        # Update for some O-M arcs are removed, such as 'Status', 'Number of neighboring machines' and 'Processing time'
        self.feat_opes_batch[batch_idxes, :3, opes] = torch.stack(
            (torch.ones(batch_idxes.size(0), dtype=torch.float),
             torch.ones(batch_idxes.size(0), dtype=torch.float),
             proc_times), dim=1)
        # Update 'Number of unscheduled operations in the job'
        for i in range(batch_idxes.size(0)):
            self.feat_opes_batch[batch_idxes[i], 3, start_ope[i]:end_ope[i] + 1] -= 1

    def update_fob_45(self, time, opes, cca, eob, oa, batch_idxes):
        # Update 'Start time' and 'Job completion time'
        self.feat_opes_batch[batch_idxes, 5, opes] = time[batch_idxes]
        is_scheduled = self.feat_opes_batch[batch_idxes, 0, :]
        mean_proc_time = self.feat_opes_batch[batch_idxes, 2, :]
        start_times = self.feat_opes_batch[batch_idxes, 5, :] * is_scheduled  # real start time of scheduled ope
        un_scheduled = 1 - is_scheduled  # unscheduled opes
        # estimate start time of unscheduled opes, 前序O用时加起来
        estimate_times = torch.bmm((start_times + mean_proc_time).unsqueeze(1), cca).squeeze() * un_scheduled
        self.feat_opes_batch[batch_idxes, 5, :] = start_times + estimate_times
        end_time_batch = (self.feat_opes_batch[batch_idxes, 5, :] +
                          self.feat_opes_batch[batch_idxes, 2, :]).gather(1, eob)
        self.feat_opes_batch[batch_idxes, 4, :] = self.convert_feat_job_2_ope(end_time_batch, oa)

    def update_fmb(self, mas, ope_ma_adj_batch, proc_times, time, machines_batch, batch_idxes):
        # Update feature vectors of machines, when maintenance, only change the 1,2
        self.feat_mas_batch[batch_idxes, 0, :] = torch.count_nonzero(ope_ma_adj_batch[batch_idxes, :, :],
                                                                          dim=1).float()
        self.feat_mas_batch[batch_idxes, 1, mas] = time[batch_idxes] + proc_times
        utiliz = machines_batch[batch_idxes, :, 2]
        cur_time = time[batch_idxes, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(time[batch_idxes, None] + 1e-9)
        self.feat_mas_batch[batch_idxes, 2, :] = utiliz