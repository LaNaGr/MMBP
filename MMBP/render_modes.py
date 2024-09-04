import torch
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os
import numpy as np
import json

sys.path.append("./")


def write_json(data:dict, path:str):
    with open(path+".json", 'w', encoding='UTF-8') as fp:
        fp.write(json.dumps(data, indent=2, ensure_ascii=False))


class Instance_for_render:
    def __init__(self, batch_size, file_name, num_jobs, num_mas, num_opes, nums_opes,
                 schedules_batch, opes_appertain_batch, num_ope_biases_batch, maintenance_info=None, name=None,
                 color_type=plt.cm.tab20(np.linspace(0,1,50)), pic_settings=None, changeover_info=None):
        """
        :param batch_size: int, number of instances
        :param file_name: str, name of the instance file
        :param num_jobs: int, number of jobs
        :param num_mas: int, number of machines
        :param num_opes: int, number of operations
        :param nums_opes: list, number of operations for each instance
        :param schedules_batch: tensor, schedules for each instance
        :param opes_appertain_batch: list, the job index for each operation
        :param num_ope_biases_batch: list, the operation index for each job
        :param maintenance_info: list, maintenance information
        :param color_type: list, color for each job
        :param pic_settings: dict, picture settings, defalt is {'machine_name': 'Machine', 'job_name': 'Job', 'operation_name': 'Operation'}
        """
        self.batch_size = batch_size
        self.file_name = file_name
        self.num_jobs = num_jobs
        self.num_mas = num_mas
        self.num_opes = num_opes
        self.nums_opes = nums_opes
        self.schedules_batch = schedules_batch
        self.opes_appertain_batch = opes_appertain_batch
        self.num_ope_biases_batch = num_ope_biases_batch
        self.folder = self.make_folder()
        self.maintenance_info = maintenance_info
        self.color_type = color_type
        self.pic_settings = pic_settings if pic_settings is not None else {'machine_name': 'Machine', 'job_name': 'Job', 'operation_name': 'Operation'}
        self.changeover_info = changeover_info
        self.name = name

    def make_folder(self):
        # time_now = time.strftime('%Y%m%d_%H', time.localtime())
        if isinstance(self.file_name, str):
            path = "./table/" + os.path.basename(self.file_name)[0:-4]  # time_now
        else:
            path = "./table/" + os.path.basename(self.file_name[0])[0:-4]
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def get_idx(self, id_ope, batch_id):
        """
            Get job and operation (relative) index based on instance index and operation (absolute) index
            """
        # idx_job = max([idx for (idx, val) in enumerate(self.num_ope_biases_batch[batch_id]) if id_ope >= val])
        idx_job = self.opes_appertain_batch[batch_id][id_ope]
        idx_ope = id_ope - self.num_ope_biases_batch[batch_id][idx_job]
        return idx_job, idx_ope

    def print_table(self):
        for batch_id in range(self.batch_size):
            schedules = self.schedules_batch[batch_id]
            print(f"instance:{batch_id}")
            # df = pd.DataFrame(columns=["Machine", "Job", "Operation", "Start", "Finish"])
            id_ope = torch.tensor(range(self.num_opes))
            idx_job, idx_ope = self.get_idx(id_ope, batch_id)
            id_machine = schedules[:, 1]
            end_time = schedules[:, 3]
            start_time = schedules[:, 2]

            df = pd.DataFrame({self.pic_settings['machine_name']: id_machine.cpu().numpy().astype(int),
                               self.pic_settings['job_name']: idx_job.cpu().numpy(),
                               self.pic_settings['operation_name']: idx_ope.cpu().numpy(),
                               "Start": start_time.cpu().numpy(),
                               "Finish": end_time.cpu().numpy(),
                               "Duration": end_time.cpu().numpy() - start_time.cpu().numpy()
                               })
            if self.name is None:
                if self.batch_size > 1:
                    if isinstance(self.file_name, list):
                        df.to_csv(self.folder + '/' + self.file_name[batch_id][-6:-4] + f"b{batch_id}.csv", sep=",",
                                  index=False, header=True)
                    else:
                        df.to_csv(self.folder + '/' + self.file_name[-6:-4] + f"b{batch_id}.csv", sep=",", index=False,
                                  header=True)
                else:
                    df.to_csv(self.folder + '/' + self.file_name[-6:-4] + f"b{batch_id}.csv", sep=",", index=False,
                              header=True)
            else:
                if self.batch_size > 1:
                    if isinstance(self.file_name, list):
                        df.to_csv(self.folder + '/' + self.name + f"b{batch_id}_{self.name}.csv", sep=",",
                                  index=False, header=True)
                    else:
                        df.to_csv(self.folder + '/' + self.name + f"b{batch_id}_{self.name}.csv", sep=",", index=False,
                                  header=True)
                else:
                    df.to_csv(self.folder + '/' + self.name + f"b{batch_id}_{self.name}.csv", sep=",", index=False,
                                header=True)

    def draw(self):
        num_jobs = self.num_jobs
        num_mas = self.num_mas
        color = self.color_type
        gray_bar_len = (torch.min(self.schedules_batch[:,:,3]-self.schedules_batch[:,:,2]) * 0.05).item()
        # print(f"gray_bar_len:{gray_bar_len}")
        # color = read_json("../utils/color_config")["gantt_color"]
        larger_size = num_jobs//10 + 1

        font_size = 8+larger_size

        fig = plt.figure(figsize=(10*larger_size, 3*larger_size))
        if len(color) < num_jobs:
            num_append_color = num_jobs - len(color)
            color = color.tolist() if ~isinstance(color, list) else color
            color += ['#' + ''.join([random.choice("0123456789ABCDEF") for _ in range(6)]) for c in
                      range(num_append_color)]
            write_json({"gantt_color": color}, "./color_config")

        for batch_id in range(self.batch_size):
            # changeover?
            if self.changeover_info is not None:
                select_change_over_for_this_batch = self.changeover_info[self.changeover_info['batch'] == batch_id]
            else:
                select_change_over_for_this_batch = None

            # draw_fig_based_on_batch_id
            schedules = self.schedules_batch[batch_id].cpu()

            fig.canvas.manager.set_window_title(
                f"J{self.num_jobs}M{self.num_mas}")
            axes = fig.add_axes([0.1, 0.1, 0.85, 0.85])
            y_ticks = []
            y_ticks_loc = []
            ##### pic setting
            for i in range(num_mas):
                y_ticks.append(self.pic_settings['machine_name']+str(i + 1))
                y_ticks_loc.append(i)  # !!!!!!!!!!!!!!!!!!!!!!attention
            labels = [''] * num_jobs
            for j in range(num_jobs):
                labels[j] = self.pic_settings['job_name'] + str(j + 1)
            patches = ([mpatches.Patch(color=color[k], label="{:s}".format(labels[k])) for k in range(self.num_jobs)] +
                       [mpatches.Patch(edgecolor='black', facecolor='yellow', hatch='///', label='event', linewidth=1*larger_size, alpha=0.3)])
            axes.cla()
            # axes.set_title('FJSP Schedule')
            axes.grid(linestyle='-.', color='black', alpha=0.1)
            axes.set_xlabel('Time / h', fontsize=font_size+1)
            axes.set_ylabel(self.pic_settings['machine_name'], fontsize=font_size+1)
            axes.set_yticks(y_ticks_loc, y_ticks, fontsize=font_size)
            axes.legend(handles=patches, ncol=1, # bbox_to_anchor=(1.01, 1.0),
                        prop={'size': font_size})  # , fontsize=int(14 / pow(1, 0.3))
            # legend loc=1,2,3,4
            # axes.set_ybound(1 - 1 / num_mas, num_mas + 1 / num_mas)
            already_schedule = 0
            for i in range(int(self.nums_opes[batch_id])):
                if schedules[i, 0] == 1:
                    id_ope = i
                    already_schedule += 1
                    idx_job, idx_ope = self.get_idx(id_ope, batch_id)
                    id_machine = schedules[id_ope][1]
                    axes.barh(id_machine,
                              gray_bar_len,     # to cpu if needed
                              left=schedules[id_ope][2],
                              color='#b2b2b2',
                              height=0.5)
                    axes.barh(id_machine,
                              schedules[id_ope][3] - schedules[id_ope][2] - gray_bar_len,
                              left=schedules[id_ope][2] + gray_bar_len,
                              color=color[idx_job],
                              height=0.5)
                    axes.text(schedules[id_ope][2] + gray_bar_len, id_machine, str(idx_job.item()+1), color='black', fontsize=font_size)
                    ###################
                    # add text here
                    ###################

            if self.maintenance_info is not None:
                ###################
                # maintenance time
                ###################
                for jj in range(len(self.maintenance_info)):
                    m_id, start_main, end_main = self.maintenance_info[jj]
                    axes.barh(m_id, end_main - start_main, left=start_main, color='darkkhaki',
                              hatch='///', edgecolor='black',
                              alpha=0.3, height=0.5, align='center')
            if select_change_over_for_this_batch is not None:
                ###################
                # changeover time
                ###################
                axes.barh(select_change_over_for_this_batch['machine'],
                          select_change_over_for_this_batch['duration'],
                          left=select_change_over_for_this_batch['start'],
                          color='lightgray', hatch='x', edgecolor='black',
                          alpha=0.3, height=0.5, align='center')
            if self.name is None:
                if self.batch_size > 1:
                    if isinstance(self.file_name,list):
                        plt.savefig(self.folder + '/' + self.file_name[batch_id][-6:-4] + f"b{batch_id}_{already_schedule}.png")
                    else:
                        plt.savefig(self.folder + '/' + self.file_name[-6:-4] + f"b{batch_id}_{already_schedule}.png")
                else:
                    plt.savefig(self.folder + '/' + self.file_name[-6:-4] + f"b{batch_id}_{already_schedule}.png")
            else:
                if self.batch_size > 1:
                    if isinstance(self.file_name, list):
                        plt.savefig(self.folder + '/' + self.name + f"b{batch_id}_{self.name}_{already_schedule}.png")
                    else:
                        plt.savefig(self.folder + '/' + self.name + f"b{batch_id}_{self.name}_{already_schedule}.png")
                else:
                    plt.savefig(self.folder + '/' + self.name + f"b{batch_id}_{self.name}_{already_schedule}.png")
            plt.xlim(0)
            # plt.show()
            plt.clf()
        plt.close('all')
