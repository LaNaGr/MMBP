import os
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from itertools import product
import copy
from multiprocessing import Pool, cpu_count
import argparse


# 订单的单元分配和订单的顺序分别用不同的向量表示，属于单个个体
# 在每次迭代中，每个解集由单元分配向量和顺序向量组成。

# case example: case_studyA.fjs
'''5 17 1
6.0 2.0 1.0 1.0 2.0 1.0 3.0 3.0 1.45 4.0 1.45 5.0 2.331 3.0 6.0 1.815 7.0 2.245 8.0 2.245 1.0 11.0 0.642 3.0 12.0 0.25 13.0 0.42 14.0 0.42 3.0 15.0 2.202 16.0 2.202 17.0 0.629
5.0 2.0 1.0 1.0 2.0 1.0 2.0 3.0 1.45 4.0 1.45 3.0 6.0 1.0 7.0 1.0 8.0 1.0 1.0 11.0 0.766 3.0 12.0 0.25 13.0 0.42 14.0 0.42
6.0 2.0 1.0 1.0 2.0 1.0 2.0 3.0 1.45 4.0 1.45 3.0 6.0 1.0 7.0 1.0 8.0 1.0 2.0 9.0 0.371 10.0 0.371 3.0 12.0 0.75 13.0 1.26 14.0 1.26 3.0 15.0 0.364 16.0 0.364 17.0 0.104
6.0 2.0 1.0 1.0 2.0 1.0 2.0 3.0 1.45 4.0 1.45 3.0 6.0 1.0 7.0 1.0 8.0 1.0 1.0 11.0 0.648 3.0 12.0 0.5 13.0 0.84 14.0 0.84 3.0 15.0 2.222 16.0 2.222 17.0 0.635
6.0 2.0 1.0 1.0 2.0 1.0 2.0 3.0 1.45 4.0 1.45 3.0 6.0 1.0 7.0 1.0 8.0 1.0 2.0 9.0 0.247 10.0 0.247 3.0 12.0 0.25 13.0 0.42 14.0 0.42 3.0 15.0 0.242 16.0 0.242 17.0 0.069'''


def read_case_study(file_name):
    jobs = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
        # 第一行表示5个job, 17个machine, 忽略第三个数字
        header = lines[0].strip().split()
        num_jobs = int(header[0])
        num_machines = int(header[1])

        # 从第二行开始读取各个job的信息
        for line in lines[1:]:
            data = line.strip().split()
            if len(data) == 0:  # 检查空行
                continue
            job_operations = []
            i = 1
            num_operations = int(float(data[0]))
            while i < len(data):
                num_alternative_machines = int(float(data[i]))
                i += 1
                alternatives = []
                for _ in range(num_alternative_machines):
                    machine_id = int(float(data[i]))
                    processing_time = float(data[i + 1])
                    alternatives.append((machine_id, processing_time))
                    i += 2
                job_operations.append(alternatives)
            jobs.append(job_operations)
    return jobs


def perform_scheduling(jobs, unit_assignment_vectors, order_sequence_vectors, num_orders, num_stages,
                       out_schedule=False):
    # Perform the scheduling to get the Gantt chart
    machine_schedules = {}  # Record the task list for each machine
    job_completion_times = [0 for _ in range(num_orders)]  # Record the completion time for each order
    CT_ALL = np.zeros(shape=[num_orders, num_stages])  # Record the completion time for each operation
    machine_schedules_mas = []  # Record the task list for each machine

    # Schedule each stage
    for stage in range(num_stages):
        # Get the orders and their sequence variables for the current stage
        orders_stage = [(i, order_sequence_vectors[i][stage]) for i in range(num_orders) if
                        unit_assignment_vectors[i][stage] is not None]
        # Sort orders by sequence variable in descending order
        sorted_orders_stage = sorted(orders_stage, key=lambda x: x[1], reverse=True)

        for order_index, _ in sorted_orders_stage:
            assigned_machine = unit_assignment_vectors[order_index][stage]
            if assigned_machine is None:
                continue

            # Get the processing time of the current operation
            processing_time = next((alt[1] for alt in jobs[order_index][stage] if alt[0] == assigned_machine), 0)
            # Calculate start and end times
            machine_schedule = machine_schedules.get(assigned_machine, [])
            last_machine_end_time = machine_schedule[-1]['Finish'] if machine_schedule else 0
            start_time = max(last_machine_end_time, job_completion_times[order_index])
            end_time = start_time + processing_time

            # Update machine schedule and job completion time
            machine_schedules.setdefault(assigned_machine, []).append({
                'Order': order_index,
                'Stage': stage,
                'Start': round(start_time, 3),
                'Finish': round(end_time, 3),
                'Duration': end_time - start_time
            })
            job_completion_times[order_index] = end_time
            CT_ALL[order_index, stage] = end_time
            machine_schedules_mas.append([order_index, stage, start_time, end_time, processing_time, assigned_machine])
    if out_schedule:
        return machine_schedules_mas, CT_ALL
    else:
        return CT_ALL


class Individual:
    def __init__(self, jobs):
        self.jobs = jobs
        self.num_orders = len(jobs)
        self.num_stages = max(len(job) for job in jobs) if jobs else 0
        self.num_machines = max(max(machine[0] for stage in job for machine in stage) for job in jobs) if jobs else 0
        self.num_operations = sum(len(job) for job in jobs)
        self.unit_assignment_vectors = []  # 用于表示每个订单在每个阶段的单元分配
        self.order_sequence_vectors = []  # 用于表示每个订单在每个阶段的顺序变量
        self.initialize_individual()
        self.CTis = None
        self.machine_schedules_mas = None

    def initialize_individual(self):
        # 初始化单元分配向量和顺序向量
        self.unit_assignment_vectors = []  # 用于表示每个订单在每个阶段的单元分配
        self.order_sequence_vectors = []  # 用于表示每个订单在每个阶段的顺序变量
        for job in self.jobs:
            # 确保所有订单的单元分配和顺序向量长度一致
            num_stages = len(job)

            # 随机生成单元分配变量，每个订单在每个阶段分配到一个生产单元
            unit_assignment_vector = [random.choice([alt[0] for alt in stage]) for stage in job]
            self.unit_assignment_vectors.append(unit_assignment_vector + [None] * (self.num_stages - num_stages))

            # 随机生成顺序变量，取值在 (0, 1) 范围内（防止出现 全0序列）
            order_sequence_vector = [round(random.uniform(0, 1), 6) for _ in range(num_stages)]
            self.order_sequence_vectors.append(order_sequence_vector + [0] * (self.num_stages - num_stages))

    def decode(self):
        # Decode the Gantt chart by scheduling operations
        self.machine_schedules_mas, self.CTis = perform_scheduling(self.jobs, self.unit_assignment_vectors,
                                                                   self.order_sequence_vectors, self.num_orders,
                                                                   self.num_stages, out_schedule=True)

    def draw(self, batch_size=1, maintenance_info=None,
             pic_settings={'job_name': 'Order', 'operation_name': 'Operation'}, color_type=None,
             name=['GA', 'test'], folder='result_ga', file_name=None, format_p="svg"):
        schedules_batch = np.array(self.machine_schedules_mas)
        num_jobs = self.num_orders
        num_opes = self.num_operations
        # Calculate the length for the gray bar representing changeover
        gray_bar_len = (np.min(schedules_batch[:, 4]) * 0.05).item()
        color = plt.cm.rainbow(np.linspace(0, 1, num_jobs))
        larger_size = num_jobs // 10 + 1
        font_size = 8 + larger_size
        fig = plt.figure(figsize=(10 * larger_size, 3 * larger_size))
        fig.canvas.manager.set_window_title(f"J{num_jobs}M{self.num_machines}")
        axes = fig.add_axes([0.1, 0.1, 0.85, 0.85])
        y_ticks = [i + 1 for i in range(self.num_machines)]
        y_ticks_loc = [i for i in range(self.num_machines)]
        labels = [pic_settings['job_name'] + str(j + 1) for j in range(num_jobs)]
        patches = [mpatches.Patch(color=color[k], label=f"{labels[k]}") for k in range(num_jobs)]
        patches.append(mpatches.Patch(edgecolor='black', facecolor='yellow', hatch='///', label='event',
                                      linewidth=1 * larger_size, alpha=0.3))
        # Setting up axes
        axes.cla()
        axes.grid(linestyle='-.', color='black', alpha=0.1)
        axes.set_xlabel('Time / h', fontsize=font_size + 1)
        axes.set_ylabel('Unit', fontsize=font_size + 1)
        axes.set_yticks(y_ticks_loc, y_ticks, fontsize=font_size)
        axes.legend(handles=patches, ncol=1, prop={'size': font_size})
        for i in range(int(num_opes)):
            id_ope = i
            id_job = int(schedules_batch[id_ope][0])
            id_machine = int(schedules_batch[id_ope][5])
            axes.barh(id_machine, gray_bar_len, left=schedules_batch[id_ope][2], color='#b2b2b2', height=0.5)
            axes.barh(id_machine, schedules_batch[id_ope][4] - gray_bar_len,
                      left=schedules_batch[id_ope][2] + gray_bar_len, color=color[id_job], height=0.5)
            axes.text(schedules_batch[id_ope][2] + gray_bar_len, id_machine, str(id_job + 1), color='black',
                      fontsize=font_size)
        # Adding maintenance information
        if maintenance_info is not None:
            for jj in range(len(maintenance_info)):
                m_id, start_main, end_main = maintenance_info[jj]
                axes.barh(m_id, end_main - start_main, left=start_main, color='darkkhaki', hatch='///',
                          edgecolor='black', alpha=0.3, height=0.5, align='center')
        ms=self.CTis.max()
        print(f"makespan:{ms},name:{name[0]}_{name[1]}")
        axes.set_title(name[0]+"    MS="+str(round(ms, 3)))
        # Saving the figure
        if not os.path.exists(f"{folder}/{name[0]}/"):
            os.makedirs(f"{folder}/{name[0]}/")
        plt.savefig(f"{folder}/{name[0]}/{name[1]}.{format_p}", format=format_p)
        plt.close('all')
        del fig


############################################## HDDE ##############################################

# Example of population individual: {'order_sequence': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], 'unit_assignment': [[1, 2, 3], [4, 5, 6]]}

# Initialize population
def initialize_population(Ind_one, pop_size):
    population = []
    for _ in range(pop_size):
        Ind_one.initialize_individual()
        population.append({
            'order_sequence': copy.deepcopy(Ind_one.order_sequence_vectors),
            'unit_assignment': copy.deepcopy(Ind_one.unit_assignment_vectors)
        })
    return population


# Mutation operation
def mutation_method_1(parent, population, F1, F2):
    F1 = adaptive_parameter_setting(F1, 0.5, 0.1, 0.9)
    F2 = adaptive_parameter_setting(F2, 0.5, 0.1, 0.9)
    r2, r3 = random.sample(population, 2)
    r1 = parent
    mutated_order_sequence = []
    mutated_unit_assignment = []

    # HDDE Mutation Method 1: Current-to-rand mutation with probability-based decision
    for i in range(len(r1['order_sequence'])):
        order_sequence_stage = []
        unit_assignment_stage = []
        for j in range(len(r1['order_sequence'][i])):
            pt3 = pt_finder(i, j, r3['unit_assignment'][i][j])
            pt2 = pt_finder(i, j, r2['unit_assignment'][i][j])
            if pt3 is not None and pt2 is not None:
                P_1 = pt3 / (pt2+pt3)
            else:
                P_1 = 0.5
            if random.random() < P_1:
                new_value_u = r2['unit_assignment'][i][j]
                new_value_v = r2['order_sequence'][i][j]
            else:
                new_value_u = r3['unit_assignment'][i][j]
                new_value_v = r3['order_sequence'][i][j]

            if j == 0:
                new_value_v = r1['order_sequence'][i][j] + F1 * (new_value_v - r1['order_sequence'][i][j])
            else:
                previous_value_v = order_sequence_stage[-1]
                previous_value_u = unit_assignment_stage[-1]
                m_previous = previous_value_u
                PT_previous = pt_finder(i, j-1, m_previous)
                new_value_v = r1['order_sequence'][i][j] + F2 * (
                            previous_value_v - PT_previous / wf - r1['order_sequence'][i][j])
            order_sequence_stage.append(new_value_v)
            unit_assignment_stage.append(new_value_u)
        mutated_order_sequence.append(order_sequence_stage)
        mutated_unit_assignment.append(unit_assignment_stage)

    return {
        'order_sequence': mutated_order_sequence,
        'unit_assignment': mutated_unit_assignment
    }, F1, F2

def mutation_method_2(parent, population, F1, F2):
    F1 = adaptive_parameter_setting(F1, 0.5, 0.1, 0.9)
    F2 = adaptive_parameter_setting(F2, 0.5, 0.1, 0.9)
    r2, r3 = random.sample(population, 2)
    r1 = parent
    mutated_order_sequence = []
    mutated_unit_assignment = []

    # HDDE Mutation Method 2
    for i in range(len(r1['order_sequence'])):
        order_sequence_stage = []
        unit_assignment_stage = []
        for j in range(len(r1['order_sequence'][i])):
            pt3 = pt_finder(i, j, r3['unit_assignment'][i][j])
            pt2 = pt_finder(i, j, r2['unit_assignment'][i][j])
            if pt3 is not None and pt2 is not None:
                P_1 = pt3 / (pt2 + pt3)
            else:
                P_1 = 0.5
            if random.random() < P_1:
                new_value_u = r2['unit_assignment'][i][j]
                new_value_v = r2['order_sequence'][i][j]
            else:
                new_value_u = r3['unit_assignment'][i][j]
                new_value_v = r3['order_sequence'][i][j]

            if j == 0:
                new_value_v = r1['order_sequence'][i][j] + F1 * (new_value_v - r1['order_sequence'][i][j])
            else:
                previous_value_v = order_sequence_stage[-1]
                previous_value_u = unit_assignment_stage[-1]
                m_previous = previous_value_u
                PT_previous = pt_finder(i, j-1, m_previous)
                new_value_v = r1['order_sequence'][i][j] + F2 * (
                            previous_value_v + PT_previous / wf - r1['order_sequence'][i][j])
            order_sequence_stage.append(new_value_v)
            unit_assignment_stage.append(new_value_u)
        mutated_order_sequence.append(order_sequence_stage)
        mutated_unit_assignment.append(unit_assignment_stage)

    return {
        'order_sequence': mutated_order_sequence,
        'unit_assignment': mutated_unit_assignment
    }, F1, F2


def mutation(parent, population, F1, F2):
    if random.random() < 0.5:
        mutant, F1_new, F2_new = mutation_method_2(parent, population, F1, F2)
    else:
        mutant, F1_new, F2_new = mutation_method_1(parent, population, F1, F2)
    return mutant, F1_new, F2_new


# Crossover operation
def crossover(parent, mutant, Ci_s_parent, Cr):
    Cr = adaptive_parameter_setting(Cr, 0.5, 0.2, 0.8)
    trial = {
        'order_sequence': [],
        'unit_assignment': []
    }
    for i in range(len(parent['order_sequence'])):
        trial_order_sequence = []
        trial_unit_assignment = []
        for j in range(len(parent['order_sequence'][i])):
            if random.random() < Cr:
                trial_order_sequence.append(mutant['order_sequence'][i][j])
            else:
                trial_order_sequence.append(parent['order_sequence'][i][j])
            if random.random() < Cr:
                trial_unit_assignment.append(mutant['unit_assignment'][i][j])
            else:
                trial_unit_assignment.append(parent['unit_assignment'][i][j])
        trial['order_sequence'].append(trial_order_sequence)
        trial['unit_assignment'].append(trial_unit_assignment)

    # 计算试验个体的适应度
    Ci_s_trial = perform_scheduling(jobs, trial['unit_assignment'], trial['order_sequence'], len(jobs), len(jobs[0]))
    if np.max(Ci_s_trial) < np.max(Ci_s_parent):
        return trial
    else:
        return {
            'order_sequence': copy.deepcopy(parent['order_sequence']),
            'unit_assignment': copy.deepcopy(parent['unit_assignment'])
        }


# Permutation operation
def permutation_method_1(individual, Ci_s):
    order_sq = np.array(individual['order_sequence'])
    unit_ass = np.array(individual['unit_assignment'])
    init_sq = order_sq.copy()
    shape_sq = order_sq.shape
    for i in range(shape_sq[-1]):  # in stage i
        Ip = np.argsort(order_sq[:, i])  # argmin可能打乱order_sq
        order_sq = copy.deepcopy(init_sq)
        if len(Ip) > 0:
            m = random.randint(round(len(Ip) * 0.4), len(Ip) - 1)
            Ip_prime = Ip[:m].tolist()

            ms_Ip = []
            tries = []
            pt_Ip = []
            valid_orders = []
            for ind, order in enumerate(Ip_prime):
                if len(pt[order]) - 1 < i:
                    # skip last stage
                    continue
                else:
                    available_machines = []
                    pt_Ip_one = m_time.copy()
                    for alt in pt[order][i]:
                        available_machines.append(alt[0])
                        pt_Ip_one[alt[0] - 1] = alt[1]  # machine pt 存在 机器号-1 的位置上
                    pt_Ip.append(pt_Ip_one)
                    ms_Ip.append([order, available_machines])
                    tries.append(len(available_machines))
                    valid_orders.append(order)
            Ip_prime = valid_orders  # 更新 Ip_prime，只保留有效的订单
            if len(Ip_prime) == 0:
                continue  # 如果没有有效的订单，跳过当前阶段

            if i == 0:
                prev_stage_CT = np.zeros(len(Ip_prime))
            else:
                prev_stage_CT = Ci_s[:, i - 1][Ip_prime]

            iter_unit_possibilities = explore_all_possibilities(tries, ms_Ip)
            LF = []
            for unit_assignment in iter_unit_possibilities:
                cis_calcu_list = []
                m_time_calcu = m_time.copy()
                for pos, unit_calcu in enumerate(unit_assignment):
                    cis_prev = max(prev_stage_CT[pos], m_time_calcu[unit_calcu - 1])
                    cis_calcu = cis_prev + pt_Ip[pos][unit_calcu - 1]
                    m_time_calcu[unit_calcu - 1] = cis_calcu
                    cis_calcu_list.append(cis_calcu)
                LF.append(sum(cis_calcu_list))
            best_unit_assignment_in_stage_i = iter_unit_possibilities[np.argmin(LF)]
            # update unit_ass in stage i
            for poss, order_ip in enumerate(Ip_prime):
                unit_ass[order_ip, i] = best_unit_assignment_in_stage_i[poss]
            individual['unit_assignment'] = unit_ass.tolist()
        Ci_s = perform_scheduling(jobs, individual['unit_assignment'], individual['order_sequence'], len(jobs),
                                  len(jobs[0]))
    return individual

def permutation_method_1_1(individual, Ci_s):
    # Apply local search to improve based on Ci_s
    # This is a dummy example to adjust sequence variables based on Ci_s
    m_time_local = [0 for _ in range(Ind_one.num_machines)]
    order_sq = np.array(individual['order_sequence'])
    unit_ass = np.array(individual['unit_assignment'])
    # print(unit_ass)
    init_sq = order_sq.copy()
    shape_sq = order_sq.shape
    for i in range(shape_sq[-1]):  # in stage i
        Ip = np.argsort(order_sq[:, i])  # argmin可能打乱order_sq
        order_sq = copy.deepcopy(init_sq)
        # print(Ip)
        if len(Ip) > 0:
            # 第4步：随机设置变量 m
            m = random.randint(round(len(Ip) * 0.4, 0), len(Ip) - 1)

            # 第5步：选择 Ip 中前 m 个订单得到子集 Ip'
            Ip_prime = Ip[:m].tolist()

            # 第6步：通过最小化 LF 选择最佳单元分配 pu
            # 只改变unit assignment 不改变order sequence
            ms_Ip = []
            tries = []
            pt_Ip = []
            pop_skip_index = None
            for ind, order in enumerate(Ip_prime):
                available_machines = []
                pt_Ip_one = m_time_local.copy()
                if len(pt[order]) - 1 < i:
                    # skip last stage
                    pop_skip_index = ind
                    continue
                else:
                    for alt in pt[order][i]:
                        available_machines.append(alt[0])
                        pt_Ip_one[alt[0] - 1] = alt[1]  # machine pt 存在 机器号-1 的位置上
                    pt_Ip.append(pt_Ip_one)
                    ms_Ip.append([order, available_machines])
                    tries.append(len(available_machines))
            if pop_skip_index is not None:
                Ip_prime.pop(pop_skip_index)
            if i == 0:
                prev_stage_CT = np.zeros(len(Ip_prime))
            else:
                prev_stage_CT = Ci_s[:, i - 1][Ip_prime]
                # prev_stage_CT = np.max(Ci_s[Ip_prime, :i-1], axis=1)

            # 遍历unit_assignment
            iter_unit_possibilities = explore_all_possibilities(tries, ms_Ip)
            LF = []
            for unit_assignment in iter_unit_possibilities:
                cis_calcu_list = []
                m_time_calcu = m_time_local.copy()
                for pos, unit_calcu in enumerate(unit_assignment):
                    # cis
                    cis_prev = max(prev_stage_CT[pos], m_time_calcu[unit_calcu - 1])
                    cis_calcu = cis_prev + pt_Ip[pos][unit_calcu - 1]  # machine pt 存在 机器号-1 的位置上
                    m_time_calcu[unit_calcu - 1] = cis_calcu
                    # print(f'Order {Ip_prime[pos]}: {unit_calcu}, till {cis_calcu}')
                    cis_calcu_list.append(cis_calcu)
                LF.append(sum(cis_calcu_list))
            best_unit_assignment_in_stage_i = iter_unit_possibilities[np.argmin(LF)]
            # update unit_ass in stage i
            poss = 0
            for order_ip in Ip_prime:
                # 被选中要局部优化的order
                # print(f'{order_ip} former assign:{unit_ass[order_ip, i]} to {best_unit_assignment_in_stage_i[poss]}')
                unit_ass[order_ip, i] = best_unit_assignment_in_stage_i[poss]
                poss += 1
            individual['unit_assignment'] = unit_ass.tolist()
        Ci_s = perform_scheduling(jobs, individual['unit_assignment'], individual['order_sequence'], len(jobs),
                                  len(jobs[0]))
    return individual


def explore_all_possibilities_1(tries, pt_Ip):
    # for permutation, 遍历order可选的所有unit
    all_possibilities = list(product(*[range(t) for t in tries]))
    results = []
    for possibility in all_possibilities:
        result = []
        for i, choice in enumerate(possibility):
            order, available_units = pt_Ip[i]
            selected_unit = available_units[choice]
            result.append(selected_unit)
        results.append(result)

    return results


def explore_all_possibilities(tries, pt_Ip, max_combinations=1000):
    # 如果可能的组合数量小于 max_combinations，就生成所有组合
    num_combinations = 1
    for t in tries:
        num_combinations *= t

    if num_combinations <= max_combinations:
        all_possibilities = list(product(*[range(t) for t in tries]))
    else:
        # 否则，随机生成 max_combinations 个组合
        all_possibilities = []
        for _ in range(max_combinations):
            possibility = [random.randrange(t) for t in tries]
            all_possibilities.append(possibility)

    results = []
    for possibility in all_possibilities:
        result = []
        for i, choice in enumerate(possibility):
            order, available_units = pt_Ip[i]
            selected_unit = available_units[choice]
            result.append(selected_unit)
        results.append(result)

    return results


def permutation_method_2(individual, Ci_s):
    m_time_local = [0 for _ in range(Ind_one.num_machines)]
    order_sq = np.array(individual['order_sequence'])
    unit_ass = np.array(individual['unit_assignment'])
    init_sq = order_sq.copy()
    shape_sq = order_sq.shape

    for i in range(shape_sq[-1]):  # in stage i
        Ip = np.argsort(order_sq[:, i])  # sorting orders based on some criteria
        order_sq = copy.deepcopy(init_sq)

        if len(Ip) > 0:
            # Step 4: Randomly set variable n (from 1 to 3 as described)
            n = random.randint(1, 3)

            for order in Ip:
                available_time_intervals = []
                pt_Ip = m_time_local.copy()

                if len(pt[order]) - 1 < i:
                    continue

                for alt in pt[order][i]:
                    available_time_intervals.append(alt[0])
                    pt_Ip[alt[0] - 1] = alt[1]  # updating machine processing times

                if i == 0:
                    prev_stage_CT = np.zeros(len(Ip))
                else:
                    prev_stage_CT = Ci_s[:, i - 1][order]

                # Step 5: Choose n time intervals for each order
                iter_time_possibilities = explore_time_intervals(n, available_time_intervals)

                # Step 6: Choose best permutation by minimizing max completion time for all units
                LF = []
                for time_assignment in iter_time_possibilities:
                    cis_calcu_list = []
                    m_time_calcu = m_time_local.copy()

                    for unit_calcu in time_assignment:
                        cis_prev = max(prev_stage_CT, m_time_calcu[unit_calcu - 1])
                        cis_calcu = cis_prev + pt_Ip[unit_calcu - 1]
                        m_time_calcu[unit_calcu - 1] = cis_calcu
                        cis_calcu_list.append(cis_calcu)

                    LF.append(max(cis_calcu_list))

                best_time_assignment = iter_time_possibilities[np.argmin(LF)]

                # Step 7: Update unit assignments with best time intervals
                for unit_assigned, order_ip in zip(best_time_assignment, Ip):
                    unit_ass[order_ip, i] = unit_assigned

            individual['unit_assignment'] = unit_ass.tolist()

        # Recalculate Ci_s with updated unit assignments
        Ci_s = perform_scheduling(jobs, individual['unit_assignment'], individual['order_sequence'], len(jobs),
                                  len(jobs[0]))

    return individual


def explore_time_intervals(n, available_time_intervals):
    # Generate all possible time interval permutations
    time_possibilities = []
    for _ in range(n):
        time_possibilities.append(random.sample(available_time_intervals, n))

    return time_possibilities


def permutation(individual, Ci_s):
    individual_copy = {
        'order_sequence': copy.deepcopy(individual['order_sequence']),
        'unit_assignment': copy.deepcopy(individual['unit_assignment'])
    }
    if random.random() < 0.5:
        if random.random() < 0.5:
            return permutation_method_1(individual_copy, Ci_s)
        else:
            return permutation_method_1(individual_copy, Ci_s)
    return individual_copy
    

# Update sequence variables pvi,s after permutation

def update_sequence_variables(individual, Ci_s):
    # Example update equation (Eq. 11): pvi,s = f(Ci,s, other_parameters)
    # Here we apply a simple transformation for demonstration purposes
    # WF = max(Ci_s)  # Set WF as the maximum completion time
    for i in range(len(individual['order_sequence'])):
        for j in range(len(individual['order_sequence'][i])):
            m_this_ope = individual['unit_assignment'][i][j]
            PTij = pt_finder(i,j,m_this_ope)
            if PTij != None:
                rand_alpha = random.uniform(0.0, 0.1)
                individual['order_sequence'][i][j] = 1 - (Ci_s[i][j] - PTij + rand_alpha) / wf
            else:
                continue
    return individual


def pt_finder(order, stage, machine):
    if len(pt[order])-1 < stage:
        return None
    else:
        pt_tuples = pt[order][stage]
        for tuple_pt in pt_tuples:
            if tuple_pt[0] == machine:
                return tuple_pt[1]
    return None  # 如果没有找到匹配的机器，返回 None

# Parameters
def adaptive_parameter_setting(factor, tau, lower_bound, upper_bound):
    if random.random() < tau:
        return lower_bound + random.random() * (upper_bound - lower_bound)
    else:
        return factor


# Selection based on fitness
# For simplicity, use negative makespan as fitness (minimize makespan)
def fitness(Cis):
    return - np.max(Cis)
    

def process_individual(args):
    idx, individual, population, jobs, jobs_len, num_stages, base_seed, gen = args
    random.seed(base_seed + random.randint(1, 10000))
    F1, F2, Cr = 0.5, 0.5, 0.5
    initial_temperature = 1.0  # Set an initial temperature for simulated annealing

    # Compute parent's scheduling and fitness
    Ci_s_parent = perform_scheduling(jobs, individual['unit_assignment'], individual['order_sequence'], jobs_len, num_stages)
    CT_parent = np.max(Ci_s_parent)

    # Mutation, Crossover, Permutation, and Update
    mutant, F1_new, F2_new = mutation(individual, population, F1, F2)
    trial = crossover(individual, mutant, Ci_s_parent, Cr)
    Ci_s_cro = perform_scheduling(jobs, trial['unit_assignment'], trial['order_sequence'], jobs_len, num_stages)
    permu = permutation(trial, Ci_s_cro)
    Ci_s_per = perform_scheduling(jobs, permu['unit_assignment'], permu['order_sequence'], jobs_len, num_stages)
    trial_updated = update_sequence_variables(permu, Ci_s_per)

    # Compute trial individual's fitness
    Ci_s_trial_updated = perform_scheduling(jobs, trial_updated['unit_assignment'], trial_updated['order_sequence'], jobs_len, num_stages)
    CT_trial = np.max(Ci_s_trial_updated)

    # Acceptance criterion using simulated annealing
    delta_CT = CT_trial - CT_parent
    T = initial_temperature / (1 + gen)
    accept_prob = np.exp(-delta_CT / T) if delta_CT > 0 else 1.0
    if random.random() < accept_prob:
        return (idx, trial_updated, CT_trial)
    else:
        return (idx, individual, CT_parent)



def process_individual_1(args):
    idx, individual, population, jobs, jobs_len, num_stages, base_seed, gen = args
    
    random.seed(base_seed + random.randint(1, 10000))
    F1, F2, Cr = 0.5, 0.5, 0.5
    initial_temperature = 1.0  # Initial temperature for simulated annealing

    # Compute parent's scheduling and fitness
    Ci_s_parent = perform_scheduling(jobs, individual['unit_assignment'], individual['order_sequence'], jobs_len, num_stages)
    CT_parent = np.max(Ci_s_parent)

    # Mutation, Crossover, Permutation, and Update
    mutant, F1_new, F2_new = mutation(individual, population, F1, F2)
    trial = crossover(individual, mutant, Ci_s_parent, Cr)
    Ci_s_cro = perform_scheduling(jobs, trial['unit_assignment'], trial['order_sequence'], jobs_len, num_stages)
    permu = permutation(trial, Ci_s_cro)
    Ci_s_per = perform_scheduling(jobs, permu['unit_assignment'], permu['order_sequence'], jobs_len, num_stages)
    trial_updated = update_sequence_variables(permu, Ci_s_per)

    # Compute trial individual's fitness
    Ci_s_trial_updated = perform_scheduling(jobs, trial_updated['unit_assignment'], trial_updated['order_sequence'], jobs_len, num_stages)
    CT_trial = np.max(Ci_s_trial_updated)

    # Acceptance criterion using simulated annealing
    delta_CT = CT_trial - CT_parent
    T = initial_temperature / (1 + gen)
    accept_prob = np.exp(-delta_CT / T) if delta_CT > 0 else 1.0
    if random.random() < accept_prob:
        return (idx, trial_updated, CT_trial)
    else:
        return (idx, individual, CT_parent)
        

def hdde(Ind_one, pop_size, iterations, cpu_core_num=None, seed=2):
    population = initialize_population(Ind_one, pop_size)
    step_fit = []
    best_fit = np.inf
    jobs_len = len(jobs)
    num_stages = len(jobs[0])
    results_CT_list=[]
    # 使用给定的基础种子生成每个个体的随机种子
    random.seed(seed)
    seeds = [random.randint(1, 100000) for _ in range(pop_size)]

    for gen in range(1, iterations + 1):
        new_population = []
        fit = []

        # 准备多处理池的参数
        args_list = [(idx, individual, population, jobs, jobs_len, num_stages, seeds[idx],gen)
                     for idx, individual in enumerate(population)]

        # 使用指定的 CPU 核心数量
        with Pool(processes=cpu_core_num) as pool:
            results = pool.map(process_individual, args_list)

        # 收集结果
        for idx, individual_new, fitness_value in results:
            new_population.append(individual_new)
            fit.append(fitness_value)

        # 更新最佳个体
        # Collect fitness values
        fit_array = np.array(fit)
        CT_min = np.min(fit_array)
        fit_mean = np.mean(fit_array)

        if CT_min < best_fit:
            index_min = np.argmin(fit_array)
            best_fit = CT_min
            best_individual = copy.deepcopy(new_population[index_min])

        # Keep the best individual (elitism)
        population = [best_individual]

        # Use tournament selection to fill up the rest of the population
        tournament_size = 3
        brand_new_pop_size = int(pop_size * 0.05)
        num_selected = pop_size - 1 - brand_new_pop_size
        selected_individuals = tournament_selection(new_population, fit_array, tournament_size, num_selected)
        population.extend(selected_individuals)

        # Introduce new random individuals
        brand_new_pop = initialize_population(Ind_one, brand_new_pop_size)
        population.extend(brand_new_pop)

        # Ensure population size remains constant
        population = population[:pop_size]

        step_fit.append(CT_min)
        print(f'# {gen:2d} {fit_mean:.3f} {best_fit:.3f}', end='\t')
        if gen % 5 == 0:
            print()
        if gen % 10 == 0:
            brand_new_pop = initialize_population(Ind_one, int(pop_size * 0.2))
            population.extend(brand_new_pop)
            population = population[:pop_size]  # Trim population to original size
            
        results_CT_list.append(round(best_fit,3))
        if gen>20:
            if best_fit == results_CT_list[-10]:
                brand_new_pop2 = initialize_population(Ind_one, int(pop_size*0.8))
                current_best_pop = [copy.deepcopy(best_individual)] * int(pop_size*0.2)
                # 检测到结果停滞超过指定的代数（如 10 代），重新初始化部分个体。
                population = brand_new_pop2 + current_best_pop
            if best_fit == results_CT_list[-20]:
                return best_individual, results_CT_list

    return best_individual, results_CT_list
    
def tournament_selection(population, fitnesses, tournament_size, num_selected):
    selected = []
    pop_size = len(population)
    for _ in range(num_selected):
        participants = random.sample(range(pop_size), tournament_size)
        best_idx = participants[0]
        best_fitness = fitnesses[participants[0]]
        for idx in participants[1:]:
            if fitnesses[idx] < best_fitness:
                best_fitness = fitnesses[idx]
                best_idx = idx
        selected.append(copy.deepcopy(population[best_idx]))
    return selected


def fitness(Cis, individual, population):
    makespan = np.max(Cis)
    diversity = calculate_diversity(individual, population)
    alpha = 0.5  # Weighting factor between makespan and diversity
    return - (alpha * makespan + (1 - alpha) * diversity)
    
    
# 示例使用
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='HDDE Algorithm for Scheduling')
    parser.add_argument('--population_size', type=int, default=100, help='Population size')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--seed', type=int, default=2, help='Random seed')
    parser.add_argument('--cpu_core_num', type=int, default=None, help='Number of CPU cores to use')
    parser.add_argument('--file_name', type=str, default='case_studyB.fjs', help='Input file name')

    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)  # 设置随机数种子
    file_name = args.file_name  # 输入文件名
    print(file_name)
    # HDDE 算法，使用的参数
    POPULATION_SIZE = args.population_size
    ITERATIONS = args.iterations
    CPU_CORE_NUM = args.cpu_core_num
    if CPU_CORE_NUM is None:
        CPU_CORE_NUM = cpu_count()

    global F1, F2, Cr
    F1, F2, Cr = 0.5, 0.5, 0.5
    global jobs
    jobs = read_case_study(file_name)
    global wf
    wf = 150
    # 初始化一个个体
    import time
    
    t1 = time.time()
    global Ind_one
    Ind_one = Individual(jobs)
    global pt
    pt = Ind_one.jobs
    global m_time
    m_time = [0 for _ in range(Ind_one.num_machines)]
    Ind_one.decode()
    Ind_one.draw(name=['GA_init', f'{file_name[-5]}_{POPULATION_SIZE}_gen{ITERATIONS}_sd{seed}'], folder='result_ga', format_p="svg")
    # run
    print(f'Run HDDE:{POPULATION_SIZE},{ITERATIONS},{CPU_CORE_NUM}') 
    best_solution, ct_list = hdde(Ind_one, POPULATION_SIZE, ITERATIONS, CPU_CORE_NUM, random.randint(1,10000))
    calcu_time = time.time() - t1
    print(f'Time calculation: {calcu_time}')
    Ind_one.order_sequence_vectors = best_solution['order_sequence']
    Ind_one.unit_assignment_vectors = best_solution['unit_assignment']
    Ind_one.decode()
    cis = perform_scheduling(jobs, best_solution['unit_assignment'], best_solution['order_sequence'], Ind_one.num_orders,
                             Ind_one.num_stages)
    print(cis)
    Ind_one.draw(name=['HDDE', f'{file_name[-5]}_{POPULATION_SIZE}_gen{ITERATIONS}_sd{seed}_t{calcu_time:.1f}'], folder='result_ga', format_p="svg")
    # print("Best Solution:", best_solution)
    ct_file = f'result_ga/{file_name[-5]}_{POPULATION_SIZE}_g{ITERATIONS}_sd{seed}_t{calcu_time:.1f}.txt'
    with open(ct_file, 'w') as f:
        f.write(','.join(map(str,ct_list)))
    individual_file = f'result_ga/{file_name[-5]}_{POPULATION_SIZE}_g{ITERATIONS}_sd{seed}_individual.txt'
    with open(individual_file, 'w') as ff:
        ff.write(str(best_solution))

