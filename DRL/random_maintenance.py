import random
import numpy as np
from math import ceil

percentage_of_machine, percentage_of_CT = 50/100, 25/100
times_for_iter = 10


def random_maintenance(CT, MACHINES, ratio_machine=percentage_of_machine, ratio_time=percentage_of_CT):
    """get initial makespan, and set 10% makespan(SIGMA=0.5MU) in 20% machines for dynamic events"""
    number_of_events = ceil(ratio_machine * MACHINES)
    choose_m = random.sample([i for i in range(MACHINES)], number_of_events)
    mu = (ratio_time * CT)
    sigma = 0.5*mu
    time_length = abs(np.random.normal(mu, sigma, number_of_events))
    mt = []
    if number_of_events==1:
        st = random.randint(0, int(CT))
        if time_length > 0:
            mt.append([choose_m[0], st, st + int(time_length)])
        else:
            mt.append([choose_m[0], st, st + 1])
    else:
        for mains in range(number_of_events):
            st = random.randint(0, int(CT))
            if int(time_length[mains]) >= 1:
                mt.append([choose_m[mains], st, st+int(time_length[mains])])
            else:
                mt.append([choose_m[mains], st, st+1])
    return mt

def random_r_n_d(num_job, file_path='./data_dev/'):
    rnd = np.random.randint(1,100, size=(num_job,2))
    rnd[:,1] = rnd[:,0] + rnd[:,1]
    np.savetxt(file_path+str(num_job)+'.csv', rnd, delimiter=",")
    return rnd

if __name__ == "__main__":
    r1 = random_r_n_d(30)