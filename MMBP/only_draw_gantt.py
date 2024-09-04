import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import random

class DrawGantt:
    def __init__(self, schedules_batch, nums_opes, pic_settings, folder, file_name, batch_size, changeover_info=None, maintenance_info=None):
        self.schedules_batch = schedules_batch
        self.nums_opes = nums_opes
        self.pic_settings = pic_settings
        self.folder = folder
        self.file_name = file_name
        self.batch_size = batch_size
        self.changeover_info = changeover_info
        self.maintenance_info = maintenance_info
        self.num_jobs = schedules_batch.size(1)
        self.num_mas = schedules_batch.size(2)
        self.color_type = pic_settings['color']
        self.draw()

    def draw(self):
        num_jobs = self.num_jobs
        num_mas = self.num_mas
        color = self.color_type
        gray_bar_len = (torch.min(self.schedules_batch[:,:,3]-self.schedules_batch[:,:,2]) * 0.05).item()
        print(f"gray_bar_len:{gray_bar_len}")
        # color = read_json("../utils/color_config")["gantt_color"]
        larger_size = num_jobs//10 + 1

        font_size = 8+larger_size

        fig = plt.figure(figsize=(10*larger_size, 3*larger_size))
        if len(color) < num_jobs:
            num_append_color = num_jobs - len(color)
            color = color.tolist() if ~isinstance(color, list) else color
            color += ['#' + ''.join([random.choice("0123456789ABCDEF") for _ in range(6)]) for c in
                      range(num_append_color)]
            # write_json({"gantt_color": color}, "./color_config")

        for batch_id in range(self.batch_size):
            # changeover?
            if self.changeover_info is not None:
                select_change_over_for_this_batch = self.changeover_info[self.changeover_info['batch'] == batch_id]
            else:
                select_change_over_for_this_batch = None

            # draw_fig_based_on_batch_id
            schedules = self.schedules_batch[batch_id].cpu()        ############################ 设备迁移好像失败了

            fig.canvas.manager.set_window_title(
                f"J{self.num_jobs}M{self.num_mas}")  # 版本变动，加了manager
            axes = fig.add_axes([0.1, 0.1, 0.85, 0.85])
            y_ticks = []
            y_ticks_loc = []
            ##### 根据pic setting 决定图中的格式
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
                              gray_bar_len,     # 注意这个变量要转移到cpu
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
                    # 此处用来加文字
                    ###################

            if self.maintenance_info is not None:
                ###################
                # 此处用来画维护时间
                ###################
                for jj in range(len(self.maintenance_info)):
                    m_id, start_main, end_main = self.maintenance_info[jj]
                    axes.barh(m_id, end_main - start_main, left=start_main, color='darkkhaki',
                              hatch='///', edgecolor='black',
                              alpha=0.3, height=0.5, align='center')
            if select_change_over_for_this_batch is not None:
                ###################
                # 此处用来画换模时间
                ###################
                axes.barh(select_change_over_for_this_batch['machine'],
                          select_change_over_for_this_batch['duration'],
                          left=select_change_over_for_this_batch['start'],
                          color='lightgray', hatch='x', edgecolor='black',
                          alpha=0.3, height=0.5, align='center')

            if self.batch_size > 1:
                if isinstance(self.file_name,list):
                    plt.savefig(self.folder + '/' + self.file_name[batch_id][-6:-4] + f"b{batch_id}_{already_schedule}.png")
                else:
                    plt.savefig(self.folder + '/' + self.file_name[-6:-4] + f"b{batch_id}_{already_schedule}.png")
            else:
                plt.savefig(self.folder + '/' + self.file_name[-6:-4] + f"b{batch_id}_{already_schedule}.png")
            plt.xlim(0)
            # plt.show()
            plt.clf()
        plt.close('all')