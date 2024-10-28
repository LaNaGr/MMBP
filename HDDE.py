import os

import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import matplotlib.patches as mpatches
import pandas as pd

# 新的编码方案的实现
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


def perform_scheduling(jobs, unit_assignment_vectors, order_sequence_vectors, num_orders, num_stages):
    # Perform the scheduling to get the Gantt chart
    machine_schedules = {}  # Record the task list for each machine
    job_completion_times = [0] * num_orders  # Record the completion time for each order
    machine_schedules_mas = []  # Record the task list for each machine

    # Schedule each stage
    for stage in range(num_stages):
        # Get the orders and their sequence variables for the current stage
        orders_stage = [(i, order_sequence_vectors[i][stage]) for i in range(num_orders) if unit_assignment_vectors[i][stage] is not None]
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
            machine_schedules_mas.append([order_index, stage, start_time, end_time, processing_time, assigned_machine])

    return machine_schedules_mas, job_completion_times


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
        self.order_sequence_vectors = [] # 用于表示每个订单在每个阶段的顺序变量
        for job in self.jobs:
            # 确保所有订单的单元分配和顺序向量长度一致
            num_stages = len(job)

            # 随机生成单元分配变量，每个订单在每个阶段分配到一个生产单元
            unit_assignment_vector = [random.choice([alt[0] for alt in stage]) for stage in job]
            self.unit_assignment_vectors.append(unit_assignment_vector + [None] * (self.num_stages - num_stages))

            # 随机生成顺序变量，取值在 (0, 1) 范围内
            order_sequence_vector = [round(random.uniform(0, 1), 6) for _ in range(num_stages)]
            self.order_sequence_vectors.append(order_sequence_vector + [0] * (self.num_stages - num_stages))

    def decode(self):
        # Decode the Gantt chart by scheduling operations
        self.machine_schedules_mas, self.CTis = perform_scheduling(self.jobs, self.unit_assignment_vectors,
                                                                 self.order_sequence_vectors, self.num_orders,
                                                                 self.num_stages)

    def draw(self, batch_size=1, maintenance_info=None,
             pic_settings={'job_name': 'Order', 'operation_name': 'Operation'}, color_type=None,
             name=['GA','test'], folder='result_ga', file_name=None, format_p="png"):
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
        # Saving the figure
        if not os.path.exists(f"{folder}/{name[0]}/"):
            os.makedirs(f"{folder}/{name[0]}/")
        plt.savefig(f"{folder}/{name[0]}/{name[1]}.{format_p}", format=format_p)
        plt.close('all')
        del fig


############################################## HDDE ##############################################
# Initialize parameters
POPULATION_SIZE = 100
ITERATIONS = 1000
F = 0.5  # Scaling factor for mutation
CR = 0.9  # Crossover rate

# Initialize population
def initialize_population(Ind_one, pop_size):
    population = []
    for _ in range(pop_size):
        Ind_one.initialize_individual()
        population.append({
            'order_sequence': Ind_one.order_sequence_vectors,
            'unit_assignment': Ind_one.unit_assignment_vectors
        })
    return population

# Mutation operation
# Mutation operation
# Mutation operation

def mutation_method_1(parent, population):
    r1, r2, r3 = random.sample(population, 3)
    mutated_order_sequence = []
    mutated_unit_assignment = []

    # HDDE Mutation Method 1: Current-to-rand mutation with probability-based decision
    for i in range(len(parent['order_sequence'])):
        order_sequence_stage = []
        unit_assignment_stage = []
        for j in range(len(parent['order_sequence'][i])):
            if random.random() < 0.5:
                # Current-to-rand mutation
                new_value = parent['order_sequence'][i][j] + F * (r1['order_sequence'][i][j] - r2['order_sequence'][i][j])
            else:
                # Standard DE mutation
                new_value = parent['order_sequence'][i][j] + F * (r1['order_sequence'][i][j] - r2['order_sequence'][i][j]) + F * (r3['order_sequence'][i][j] - parent['order_sequence'][i][j])
            order_sequence_stage.append(new_value)
            # Unit assignment mutation: choose from r1 or r2 if not None, else keep parent
            if parent['unit_assignment'][i][j] is not None:
                unit_assignment_stage.append(parent['unit_assignment'][i][j])
            else:
                unit_assignment_stage.append(random.choice([r1['unit_assignment'][i][j], r2['unit_assignment'][i][j]]))
        mutated_order_sequence.append(order_sequence_stage)
        mutated_unit_assignment.append(unit_assignment_stage)

    return {
        'order_sequence': mutated_order_sequence,
        'unit_assignment': mutated_unit_assignment
    }


def mutation_method_2(parent, population):
    # Mutation Method 2: Start from the last stage to the first stage
    r1, r2, r3 = random.sample(population, 3)
    mutated_order_sequence = []
    mutated_unit_assignment = []

    for i in reversed(range(len(parent['order_sequence']))):
        order_sequence_stage = []
        unit_assignment_stage = []
        for j in range(len(parent['order_sequence'][i])):
            new_value = parent['order_sequence'][i][j] + F * (r1['order_sequence'][i][j] - r2['order_sequence'][i][j])
            order_sequence_stage.append(new_value)
            # Unit assignment mutation: choose from r1 or r2 if not None, else keep parent
            if parent['unit_assignment'][i][j] is not None:
                unit_assignment_stage.append(parent['unit_assignment'][i][j])
            else:
                unit_assignment_stage.append(random.choice([r1['unit_assignment'][i][j], r2['unit_assignment'][i][j]]))
        mutated_order_sequence.insert(0, order_sequence_stage)
        mutated_unit_assignment.insert(0, unit_assignment_stage)

    return {
        'order_sequence': mutated_order_sequence,
        'unit_assignment': mutated_unit_assignment
    }


def mutation(parent, population):
    # Choose between Mutation Method 1 and Mutation Method 2
    if random.random() < 0.5:
        return mutation_method_1(parent, population)
    else:
        return mutation_method_2(parent, population)


# Crossover operation
def crossover(parent, mutant, Ci_s_parent):
    trial = {
        'order_sequence': [],
        'unit_assignment': []
    }
    for i in range(len(parent['order_sequence'])):
        trial_order_sequence = []
        trial_unit_assignment = []
        for j in range(len(parent['order_sequence'][i])):
            if random.random() < CR:
                trial_order_sequence.append(mutant['order_sequence'][i][j])
            else:
                trial_order_sequence.append(parent['order_sequence'][i][j])
            trial_unit_assignment.append(
                parent['unit_assignment'][i][j]) if random.random() >= CR else trial_unit_assignment.append(
                mutant['unit_assignment'][i][j])
        trial['order_sequence'].append(trial_order_sequence)
        trial['unit_assignment'].append(trial_unit_assignment)

    # Calculate fitness for parent and trial
    _, Ci_s_trial = perform_scheduling(jobs, trial['unit_assignment'], trial['order_sequence'], len(jobs), len(jobs[0]))
    if fitness(Ci_s_trial) > fitness(Ci_s_parent):
        return trial
    else:
        return parent


# Permutation operation
def permutation_method_1(individual, Ci_s):
    # Apply local search to improve based on Ci_s
    # This is a dummy example to adjust sequence variables based on Ci_s
    for i in range(len(individual['order_sequence'])):
        for j in range(len(individual['order_sequence'][i])):
            individual['order_sequence'][i][j] = round(individual['order_sequence'][i][j] * (1 + 0.1 * (Ci_s[i] / 100)), 6)
    return individual


def permutation_method_2(individual, Ci_s):
    # Apply local improvement by focusing on reducing idle time
    for i in range(len(individual['order_sequence'])):
        for j in range(len(individual['order_sequence'][i])):
            adjustment_factor = random.uniform(0.9, 1.1)  # Random adjustment to reduce idle time
            individual['order_sequence'][i][j] = round(individual['order_sequence'][i][j] * adjustment_factor, 6)
    return individual


def permutation(individual, Ci_s):
    # Decide whether to apply permutation
    if random.random() < 0.5:
        # Choose between Permutation Method 1 and Permutation Method 2
        if random.random() < 0.5:
            return permutation_method_1(individual, Ci_s)
        else:
            return permutation_method_2(individual, Ci_s)
    return individual

# Update sequence variables pvi,s after permutation

def update_sequence_variables(individual, Ci_s):
    # Example update equation (Eq. 11): pvi,s = f(Ci,s, other_parameters)
    # Here we apply a simple transformation for demonstration purposes
    WF = max(Ci_s)  # Set WF as the maximum completion time
    for i in range(len(individual['order_sequence'])):
        for j in range(len(individual['order_sequence'][i])):
            PTij = individual['order_sequence'][i][j]  # Placeholder for processing time
            individual['order_sequence'][i][j] = round((PTij / WF) * (1 + 0.1 * (Ci_s[i] / 100)), 6)
    return individual


# Selection based on fitness
# For simplicity, use negative makespan as fitness (minimize makespan)
def fitness(Cis):
    return -max(Cis)


def selection(parent, trial, Ci_s_parent, Ci_s_trial):
    if fitness(Ci_s_trial) > fitness(Ci_s_parent):
        return trial
    else:
        return parent


def hdde(Ind_one, pop_size, iterations):
    population = initialize_population(Ind_one, pop_size)
    for _ in range(iterations):
        new_population = []
        for individual in population:
            _, Ci_s_parent = perform_scheduling(jobs, individual['unit_assignment'], individual['order_sequence'], len(jobs), len(jobs[0]))
            mutant = mutation(individual, population)
            _, Ci_s_mut = perform_scheduling(jobs, mutant['unit_assignment'], mutant['order_sequence'], len(jobs), len(jobs[0]))
            cross= crossover(individual, mutant, Ci_s_mut)
            _, Ci_s_cro = perform_scheduling(jobs, cross['unit_assignment'], cross['order_sequence'], len(jobs), len(jobs[0]))
            permu = permutation(cross, Ci_s_cro)  # Apply permutation after crossover
            _, Ci_s_per = perform_scheduling(jobs, permu['unit_assignment'], permu['order_sequence'], len(jobs), len(jobs[0]))
            trial = update_sequence_variables(permu, Ci_s_per)  # Update sequence variables after permutation
            _, Ci_s_trial_updated = perform_scheduling(jobs, trial['unit_assignment'], trial['order_sequence'], len(jobs), len(jobs[0]))
            if fitness(Ci_s_trial_updated) > fitness(Ci_s_parent):
                new_population.append(trial)
            else:
                new_population.append(individual)
        population = new_population
    # Return the best individual based on fitness
    best_individual = max(population, key=lambda ind: fitness(perform_scheduling(jobs, ind['unit_assignment'], ind['order_sequence'], len(jobs), len(jobs[0]))))
    return best_individual


# 示例使用
if __name__ == "__main__":
    # 读取数据和初始化调度
    file_name = 'case_studyA.fjs'
    jobs = read_case_study(file_name)

    # 初始化一个个体
    Ind_one = Individual(jobs)

    # Run HDDE
    best_solution = hdde(Ind_one, POPULATION_SIZE, ITERATIONS)
    print("Best Solution:", best_solution)


    # 解码并绘制甘特图
    '''individual.decode()
    individual.draw()'''
