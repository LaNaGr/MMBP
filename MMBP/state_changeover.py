import copy

import torch
from dataclasses import dataclass


@dataclass
class EnvState:
    """
    Class for the state of the environment
    """
    # static
    opes_appertain_batch: torch.Tensor = None   # The operation belongs to which job
    ope_pre_adj_batch: torch.Tensor = None      # Predecessor operation adjacency matrix
    ope_sub_adj_batch: torch.Tensor = None      # Successor operation adjacency matrix
    end_ope_biases_batch: torch.Tensor = None   # The last operation of each job is which No.Operation in total problem
    nums_opes_batch: torch.Tensor = None        # Number of operations in each job

    # dynamic
    batch_idxes: torch.Tensor = None            # Index of the batch
    feat_opes_batch: torch.Tensor = None        # Operation feature
    feat_mas_batch: torch.Tensor = None         # Machine feature
    proc_times_batch: torch.Tensor = None       # Processing time of each operation
    ope_ma_adj_batch: torch.Tensor = None       # Machine adjacency matrix of each operation
    time_batch: torch.Tensor = None             # Current time

    mask_job_procing_batch: torch.Tensor = None # Mask of jobs that are being processed
    mask_job_finish_batch: torch.Tensor = None  # Mask of jobs that have been completed
    mask_ma_procing_batch: torch.Tensor = None  # Mask of machines that are being processed
    mask_maintenance_ma_batch: torch.Tensor = None  # Mask of machines that are in maintenance
    mask_job_release_batch: torch.Tensor = None # Mask of jobs that are released
    ope_step_batch: torch.Tensor = None         # The step of each operation

    def update(self, batch_idxes, feat_opes_batch, feat_mas_batch, proc_times_batch, ope_ma_adj_batch,
               mask_job_procing_batch, mask_job_finish_batch, mask_ma_procing_batch, ope_step_batch, time_batch,
               mask_maintenance_ma_batch, mask_job_release_batch):
        self.batch_idxes = batch_idxes
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch
        self.proc_times_batch = proc_times_batch
        self.ope_ma_adj_batch = ope_ma_adj_batch
        self.mask_job_procing_batch = mask_job_procing_batch
        self.mask_job_finish_batch = mask_job_finish_batch
        self.mask_ma_procing_batch = mask_ma_procing_batch
        self.mask_job_release_batch = mask_job_release_batch
        self.ope_step_batch = ope_step_batch
        self.time_batch = time_batch
        self.mask_maintenance_ma_batch = mask_maintenance_ma_batch

