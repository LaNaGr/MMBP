import torch
import numpy as np
import copy


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
    matrix_proc_time = torch.zeros(size=(num_opes, num_mas), dtype=torch.float)
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
            num_ope = int(edge_detec(line, num_ope_bias, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul))  # 这里首先得到float
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
    '''
    Count the number of jobs, machines and operations
    '''
    num_opes = 0
    for i in range(1, len(lines)):
        num_opes += int(float(lines[i].strip().split()[0])) if lines[i] != "\n" else 0
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
        x = float(i)
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
            matrix_proc_time[idx_ope + num_ope_bias][int(mac)] = x      # mac 可能是float
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
        cases_path = ['../Data/Mk01_v3.fjs', '../Data/Mk01_v3.fjs'] # 来自文件，将其重复size遍
    if batch is None:
        batch_size = len(cases_path)
    else:
        batch_size = batch
    lines = []
    # load instance
    num_data = 8    # The amount of data extracted from instance
    max_num_opes, max_num_mas = 0, 0  # longest opes
    tensors = [[] for _ in range(num_data)]
    if batch_size == 1:
        road = cases_path[0] if isinstance(cases_path, list) else cases_path
        with open(road) as file_object:
            line = file_object.readlines()
            lines.append(line)
        num_jobs, num_mas, num_opes = nums_detec(lines[0])
        max_num_opes = max(max_num_opes, num_opes)
        max_num_mas = max(max_num_mas, num_mas)     # default: same mas_num!
    else:
        if isinstance(cases_path, str):
            cases_path = [cases_path]*batch_size
        else:
            cases_path = cases_path
        for i in range(batch_size):
            road = cases_path[i]
            with open(road) as file_object:
                line = file_object.readlines()
                lines.append(line)
            num_jobs, num_mas, num_opes = nums_detec(lines[i])
            max_num_opes = max(max_num_opes, num_opes)
            max_num_mas = max(max_num_mas, num_mas)     # default: same mas_num!
    # load feats
    for i in range(batch_size):
        load_data = load_fjs(lines[i], max_num_mas, max_num_opes)
        for j in range(num_data):
            tensors[j].append(load_data[j])
    num = [num_jobs, max_num_mas, max_num_opes]
    return tensors, num


class instance:
    def __init__(self, paths=['../Data/Mk01_v3.fjs', '../Data/Mk01_v3.fjs'], batch=None, device='cpu'):
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
        self.file_path = paths
        self.batch_size = len(paths) if batch is None else batch
        tensors, num = to_tensor(paths, batch)
        self.num_jobs, self.num_mas, self.num_opes = num

        # dynamic feats
        self.proc_times_batch = torch.stack(tensors[0], dim=0)          # shape: (batch_size, num_opes, num_mas)
        self.ope_ma_adj_batch = torch.stack(tensors[1], dim=0).long()   # shape: (batch_size, num_opes, num_mas)
        # most important 2: ope_ma_adj_batch & pro_times_batch, 不变

        # for calculating the cumulative amount along the path of each job
        self.cal_cumul_adj_batch = torch.stack(tensors[7], dim=0).float()   # shape: (batch_size, num_opes, num_opes)

        # static feats
        self.ope_pre_adj_batch = torch.stack(tensors[2], dim=0)     # shape: (batch_size, num_opes, num_opes)
        self.ope_sub_adj_batch = torch.stack(tensors[3], dim=0)     # shape: (batch_size, num_opes, num_opes)

        # the mapping between operations and jobs
        self.opes_appertain_batch = torch.stack(tensors[4], dim=0).long()       # shape: (batch_size, num_opes)

        # the id of the first operation of each job
        self.num_ope_biases_batch = torch.stack(tensors[5], dim=0).long()       # shape: (batch_size, num_jobs)

        # the number of operations for each job
        self.nums_ope_batch = torch.stack(tensors[6], dim=0).long()             # shape: (batch_size, num_jobs)

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


if __name__ == "__main__":

    ins = instance(['../Data/Mk01_v3.fjs','../Data/Mk03.fjs'])    # default path

    print()
