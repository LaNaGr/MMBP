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


def load_fjs(lines, num_mas, num_opes):
    """
    Load the local FJSP instance.
    """
    flag = 0
    matrix_proc_time = torch.zeros(size=(num_opes, num_mas))
    matrix_pre_proc = torch.full(size=(num_opes, num_opes), dtype=torch.bool, fill_value=False)
    matrix_cal_cumul = torch.zeros(size=(num_opes, num_opes)).int()
    nums_ope = []  # A list of the number of operations for each job
    opes_appertain = np.array([])
    num_ope_biases = []  # The id of the first operation of each job
    # Parse data line by line
    for line in lines:
        # first line
        if flag == 0:
            flag += 1
        # last line
        elif line == "\n":
            break
        # other
        else:
            num_ope_bias = int(sum(nums_ope))  # The id of the first operation of this job
            num_ope_biases.append(num_ope_bias)
            # Detect information of this job and return the number of operations
            num_ope = edge_detec(line, num_ope_bias, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul)
            nums_ope.append(num_ope)
            # nums_option = np.concatenate((nums_option, num_option))
            opes_appertain = np.concatenate((opes_appertain, np.ones(num_ope) * (flag - 1)))
            flag += 1
    matrix_ope_ma_adj = torch.where(matrix_proc_time > 0, 1, 0)
    # Fill zero if the operations are insufficient (for parallel computation)
    opes_appertain = np.concatenate((opes_appertain, np.zeros(num_opes - opes_appertain.size)))
    return matrix_proc_time, matrix_ope_ma_adj, matrix_pre_proc, matrix_pre_proc.t(), \
           torch.tensor(opes_appertain).int(), torch.tensor(num_ope_biases).int(), \
           torch.tensor(nums_ope).int(), matrix_cal_cumul


def nums_detec(lines):
    """
    Count the number of jobs, machines and operations
    """
    num_opes = 0
    for i in range(1, len(lines)):
        num_opes += int(lines[i].strip().split()[0]) if lines[i] != "\n" else 0
    line_split = lines[0].strip().split()
    num_jobs = int(line_split[0])
    num_mas = int(line_split[1])
    return num_jobs, num_mas, num_opes


def edge_detec(line, num_ope_bias, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul):
    """
    Detect information of a job
    """
    line_split = line.split()
    flag = 0
    flag_time = 0
    flag_new_ope = 1
    idx_ope = -1
    num_ope = 0  # Store the number of operations of this job
    num_option = np.array([])  # Store the number of processable machines for each operation of this job
    mac = 0
    for i in line_split:
        x = int(float(i))
        # The first number indicates the number of operations of this job
        if flag == 0:
            num_ope = x
            flag += 1
        # new operation detected
        elif flag == flag_new_ope:
            idx_ope += 1
            flag_new_ope += x * 2 + 1
            num_option = np.append(num_option, x)
            if idx_ope != num_ope - 1:
                matrix_pre_proc[idx_ope + num_ope_bias][idx_ope + num_ope_bias + 1] = True
            if idx_ope != 0:
                vector = torch.zeros(matrix_cal_cumul.size(0))
                vector[idx_ope + num_ope_bias - 1] = 1
                matrix_cal_cumul[:, idx_ope + num_ope_bias] = matrix_cal_cumul[:, idx_ope + num_ope_bias - 1] + vector
            flag += 1
        # not proc_time (machine)
        elif flag_time == 0:
            mac = x - 1
            flag += 1
            flag_time = 1
        # proc_time
        else:
            matrix_proc_time[idx_ope + num_ope_bias][mac] = x
            flag += 1
            flag_time = 0
    return num_ope


def to_tensor(cases_path=None, batch=None):
    """
    tensors is a list, len(tensors) = num_data = 8
    these 8 tensor, len(tensor) = batch_size
        0: ops proc time !
        1: matrix O-M pair adjacent !
        2: pre-proc：是否前置工作
        3: pre-proc.T
        4: appertain O都属于哪个job, 0,1,... [0 0 0 1 1 1 2 2 2...]
        5: ope_biases, 每个job从哪里开始，是一个list, 长度等于job数量， [0, 6, 12, 16, 21, 25, 31, 35, 40, 45]
        6: num ops list, 每个job几个ope，长度等于job数量， [6, 6, 4, 5, 4, 6, 4, 5, 5, 4]
        7: matrix_cal_cumul: cumulative amount of each job;
            后续OPE，累积adjacent matrix, 比如O_11连接了O_12-15这个阵的第一行[0,1,1,1,1,0,0...0]"""
    if cases_path is None:
        cases_path = ['../Data/Mk01_v3.fjs', '../Data/Mk01_v3.fjs']  # 来自文件，将其重复size遍
    if batch is None:
        batch_size = len(cases_path)
    else:
        batch_size = batch
    lines = []
    # load instance
    num_data = 8  # The amount of data extracted from instance
    max_num_opes, max_num_mas = 0, 0  # longest opes
    tensors = [[] for _ in range(num_data)]
    if batch_size == 1:
        road = cases_path[0] if isinstance(cases_path, list) else cases_path
        with open(road) as file_object:
            line = file_object.readlines()
            lines.append(line)
        num_jobs, num_mas, num_opes = nums_detec(lines[0])
        max_num_opes = max(max_num_opes, num_opes)
        max_num_mas = max(max_num_mas, num_mas)  # default: same mas_num!
    else:
        for i in range(batch_size):
            road = cases_path[i]
            with open(road) as file_object:
                line = file_object.readlines()
                lines.append(line)
            num_jobs, num_mas, num_opes = nums_detec(lines[i])
            max_num_opes = max(max_num_opes, num_opes)
            max_num_mas = max(max_num_mas, num_mas)  # default: same mas_num!
    # load feats
    for i in range(batch_size):
        load_data = load_fjs(lines[i], max_num_mas, max_num_opes)
        for j in range(num_data):
            tensors[j].append(load_data[j])
    num = [num_jobs, max_num_mas, max_num_opes]
    return tensors, num


class instance:
    def __init__(self, paths=None, batch=None):
        """the information of instance:
        {file_path, batch_size, num_jobs, num_mas, num_opes, proc_times_batch, ope_ma_adj_batch, cal_cumul_adj_batch,
        ope_pre_adj_batch, ope_sub_adj_batch, opes_appertain_batch, num_ope_biases_batch, nums_ope_batch, end_ope_biases_batch, nums_opes}
        """
        # proc_times_batch                processing time O-M
        # ope_ma_adj_batch                adjacent matrix O-M
        # cal_cumul_adj_batch             cumulative amount of each job
        # ope_pre_adj_batch               predecessor O
        # ope_sub_adj_batch               successor O
        # opes_appertain_batch            which job the O belongs to
        # num_ope_biases_batch            the first O of each job
        # nums_ope_batch                  the number of O for each job
        # end_ope_biases_batch            the last O of each job
        # nums_opes                       total number of O - all batches
        if paths is None:
            paths = ['../Data/Mk01_v3.fjs', '../Data/Mk01_v3.fjs']
        self.file_path = paths
        self.batch_size = len(paths) if batch is None else batch
        tensors, num = to_tensor(paths, batch)
        self.num_jobs, self.num_mas, self.num_opes = num

        # dynamic feats
        self.proc_times_batch = torch.stack(tensors[0], dim=0)  # shape: (batch_size, num_opes, num_mas)
        self.ope_ma_adj_batch = torch.stack(tensors[1], dim=0).long()  # shape: (batch_size, num_opes, num_mas)
        # most important 2: ope_ma_adj_batch & pro_times_batch, 不变

        # for calculating the cumulative amount along the path of each job
        self.cal_cumul_adj_batch = torch.stack(tensors[7], dim=0).float()  # shape: (batch_size, num_opes, num_opes)

        # static feats
        self.ope_pre_adj_batch = torch.stack(tensors[2], dim=0)  # shape: (batch_size, num_opes, num_opes)
        self.ope_sub_adj_batch = torch.stack(tensors[3], dim=0)  # shape: (batch_size, num_opes, num_opes)

        # the mapping between operations and jobs
        self.opes_appertain_batch = torch.stack(tensors[4], dim=0).long()  # shape: (batch_size, num_opes)

        # the id of the first operation of each job
        self.num_ope_biases_batch = torch.stack(tensors[5], dim=0).long()  # shape: (batch_size, num_jobs)

        # the number of operations for each job
        self.nums_ope_batch = torch.stack(tensors[6], dim=0).long()  # shape: (batch_size, num_jobs)

        # the id of the last operation of each job
        self.end_ope_biases_batch = self.num_ope_biases_batch + self.nums_ope_batch - 1
        # shape: (batch_size, num_jobs)

        # the number of operations for each instance
        self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)  # shape: (batch_size)

        # to_reset
        self.old_proc_times_batch = copy.deepcopy(self.proc_times_batch)
        self.old_ope_ma_adj_batch = copy.deepcopy(self.ope_ma_adj_batch)
        self.old_cal_cumul_adj_batch = copy.deepcopy(self.cal_cumul_adj_batch)

    def reset_self(self):
        self.proc_times_batch = copy.deepcopy(self.old_proc_times_batch)
        self.ope_ma_adj_batch = copy.deepcopy(self.old_ope_ma_adj_batch)
        self.cal_cumul_adj_batch = copy.deepcopy(self.old_cal_cumul_adj_batch)

    def update_oma_pt_cca(self, mas, opes, last_opes, batch_idxes):
        # Removed unselected O-M arcs of the scheduled operations
        remain_ope_ma_adj = torch.zeros(size=(self.batch_size, self.num_mas), dtype=torch.int64)
        remain_ope_ma_adj[batch_idxes, mas] = 1
        self.ope_ma_adj_batch[batch_idxes, opes] = remain_ope_ma_adj[batch_idxes, :]
        self.proc_times_batch *= self.ope_ma_adj_batch
        self.cal_cumul_adj_batch[batch_idxes, last_opes, :] = 0


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


class mask:
    def __init__(self, batch_size, num_jobs, num_mas):
        # Masks of current status, dynamic
        self.batch_size, self.num_jobs, self.num_mas = batch_size, num_jobs, num_mas
        self.mask_job_procing_batch = torch.full(size=(batch_size, num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_job_finish_batch = torch.full(size=(batch_size, num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_ma_procing_batch = torch.full(size=(batch_size, num_mas), dtype=torch.bool, fill_value=False)
        self.mask_maintenance_ma_batch = torch.full(size=(batch_size, num_mas), dtype=torch.bool, fill_value=False)
        # job maintenance直接东job proc

    def reset_self(self):
        # mask for job, shape: (batch_size, num_jobs), True for jobs in process
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool,
                                                 fill_value=False)
        # mask for job, shape: (batch_size, num_jobs), True for completed jobs
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool,
                                                fill_value=False)
        # mask for machine, shape: (batch_size, num_mas), True for machines in process
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool,
                                                fill_value=False)
        self.mask_maintenance_ma_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool,
                                                    fill_value=False)

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

    def __init__(self, batch_size, num_opes, num_mas, num_ope_biases_batch, feat_opes_batch, mask_job_finish_batch):
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
        self.batch_size, self.num_opes, self.num_mas = batch_size, num_opes, num_mas
        self.num_ope_biases_batch = num_ope_biases_batch
        self.feat_opes_batch, self.mask_job_finish_batch = feat_opes_batch, mask_job_finish_batch

        self.schedules_batch = torch.zeros(size=(batch_size, num_opes, 4))
        self.schedules_batch[:, :, 2] = feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = feat_opes_batch[:, 5, :] + feat_opes_batch[:, 2, :]

        self.machines_batch = torch.zeros(size=(batch_size, num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(batch_size, num_mas))

        self.ope_step_batch = copy.deepcopy(num_ope_biases_batch)  # shape: (batch_size, num_jobs)
        # the id of the current operation (be waiting to be processed) of each job

        # dynamic variable
        self.batch_idxes = torch.arange(batch_size)  # Uncompleted instances
        self.time = torch.zeros(batch_size)  # Current time of the environment
        self.N = torch.zeros(batch_size).int()  # Count scheduled operations
        self.makespan_batch = torch.max(feat_opes_batch[:, 4, :], dim=1)[0]  # shape: (batch_size)
        self.done_batch = mask_job_finish_batch.all(dim=1)  # shape: (batch_size)
        self.stuck_job = torch.ones(batch_size, num_mas) * -1
        self.stuck_time = torch.zeros(batch_size, num_mas)

        self.old_makespan = torch.max(feat_opes_batch[:, 4, :], dim=1)[0]

    def reset_self(self):
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4))
        self.schedules_batch[:, :, 2] = self.feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = self.feat_opes_batch[:, 5, :] + self.feat_opes_batch[:, 2, :]
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)

        self.time = torch.zeros(self.batch_size)
        self.N = torch.zeros(self.batch_size)
        self.makespan_batch = copy.deepcopy(self.old_makespan)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        self.stuck_job = torch.ones(self.batch_size, self.num_mas) * -1
        self.stuck_time = torch.zeros(self.batch_size, self.num_mas)
        self.batch_idxes = torch.arange(self.batch_size)

    def update_static(self, opes, mas, fob_st, fob_pt, proc_times, jobs, time_now):
        self.schedules_batch[self.batch_idxes, opes, :2] = torch.stack((torch.ones(self.batch_idxes.size(0)), mas),
                                                                       dim=1)
        self.schedules_batch[self.batch_idxes, :, 2] = fob_st
        self.schedules_batch[self.batch_idxes, :, 3] = fob_st + fob_pt
        self.machines_batch[self.batch_idxes, mas, 0] = torch.zeros(self.batch_idxes.size(0))
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
