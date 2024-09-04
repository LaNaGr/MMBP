import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from MMBP_v1_1 import MMBPEnv
import torch
import cv2
import io
from typing import Union, Dict
import matplotlib as mpl
import pandas as pd


class draw_graph:
    """Node Graph of instances in MMBPEnv, using NetworkX
    Heterogeneous Graph"""

    def __init__(self, env: MMBPEnv, dpi=150, width=15, height=10,
                 scheduled_color="#DAF7A6", not_scheduled_color="#FFC300", color_job_edge="tab:gray",
                 node_drawing_kwargs=None,
                 edge_drawing_kwargs=None,
                 critical_path_drawing_kwargs=None):
        self.env = env
        self.width = width
        self.height = height
        self.dpi = dpi
        self.color_scheduled = scheduled_color
        self.color_not_scheduled = not_scheduled_color
        self.color_job_edge = color_job_edge
        self.node_drawing_kwargs = {
            "node_size": 800,
            "linewidths": 5
        } if node_drawing_kwargs is None else node_drawing_kwargs
        self.edge_drawing_kwargs = {
            "arrowsize": 30
        } if edge_drawing_kwargs is None else edge_drawing_kwargs
        self.critical_path_drawing_kwargs = {
            "edge_color": 'r',
            "width": 20,
            "alpha": 0.1,
        } if critical_path_drawing_kwargs is None else critical_path_drawing_kwargs
        self.G = None
        self.machine_colors, self.gantt_colors = None, None
        self.scheduled_array = pd.DataFrame(data=None, columns=["Task", "Start", "Finish", "Resource"])

    def graph_set(self, env):
        arr = np.linspace(0, 1, env.instance.num_mas)
        machine_colors = {m_id: np.array(c_map(val))[0:-1].reshape(1, -1) for m_id, val in enumerate(arr)}
        gantt_colors = {m_id: c_map(val)[0:-1] for m_id, val in enumerate(arr)}
        # machine_colors = {m_id: plt.get_cmap('tab20')[m_id] for m_id in range(env.instance.num_mas)}
        self.machine_colors = machine_colors
        self.gantt_colors = gantt_colors
        machine_routes = {m_id: np.array([], dtype=int) for m_id in range(env.instance.num_mas)}
        dummy_task_machine = -1
        dummy_task_job = -1
        dummy_task_color = "tab:gray"
        n_jobs = env.instance.num_jobs
        n_machines = env.instance.num_mas
        total_task = env.instance.num_opes
        src_task = 'src'
        sink_task = 'sink'
        sink_pos = torch.max(env.instance.nums_ope_batch)

        # build a graph
        G = nx.DiGraph()
        # source
        G.add_node(
            src_task, node_shape='o',
            pos=(-1, int(-n_jobs * 0.5)),
            duration=0,
            machine=dummy_task_machine,
            scheduled=True,
            color=dummy_task_color,
            job=dummy_task_job,
            start_time=0,
            finish_time=0,
            label='0'
        )
        # add sink task at the end to avoid permutation in the adj matrix.
        # the rows and cols correspond to the order the nodes were added not the index/value of the node.
        G.add_node(
            sink_task, node_shape='o',
            pos=(sink_pos, int(-n_jobs * 0.5)),
            color=dummy_task_color,
            duration=0,
            machine=dummy_task_machine,
            job=dummy_task_job,
            scheduled=True,
            start_time=None,
            finish_time=None,
            label='*'
        )
        # add machine nodes
        pos_interval = (sink_pos+0.5) / n_machines
        for m in range(n_machines):
            G.add_node(
                'M' + str(m),
                pos=(pos_interval * m-0.5, 2),
                color=machine_colors[m],
                duration=0,
                machine=None,
                job=None,
                scheduled=False,
                start_time=None,
                finish_time=None,
                node_shape='s',
                label='m'+str(m+1)
            )

        # add nodes in grid format
        task_id = -1  # original: 0, change it here
        # machine_order = env.instance.
        # processing_times = env.instance.proc_times_batch

        for i in range(env.instance.num_jobs):
            num_o_of_job = env.instance.nums_ope_batch[0, i]
            for j in range(num_o_of_job.item()):
                task_id += 1  # start from task id 0, -1 is dummy starting task
                m_id_list = env.instance.ope_ma_adj_batch[0, task_id]  # machine id
                m_id = torch.nonzero(m_id_list).T[0].tolist()
                print(m_id)
                # 内置问题，只有一个machine时无法输出list甚至ope-ma对不正确
                dur = env.instance.proc_times_batch[0, task_id][m_id_list.bool()]  # duration of the task
                mean_dur = round(torch.mean(dur).item(), 1)

                G.add_node(
                    task_id,
                    pos=(j, -i),
                    color=dummy_task_color,  # machine_colors[m_id]
                    duration=mean_dur,  # jsp: duration, fjsp: mean duration
                    scheduled=False,
                    machine=m_id,
                    job=i,
                    start_time=None,
                    finish_time=None,
                    node_shape='o',
                    label='O'+f'$_{i+1}$'+'$_,$'+f'$_{j}$',
                )

                if j == 0:  # first task in a job
                    G.add_edge(
                        src_task, task_id,
                        job_edge=True, machine_arc=-1,
                        # weight=G.nodes[src_task]['duration'],
                        nweight=-G.nodes[src_task]['duration'],
                    )
                elif j == num_o_of_job - 1:  # last task of a job, remember to link it to the sink
                    G.add_edge(
                        task_id - 1, task_id,
                        job_edge=True, machine_arc=-1,
                        # weight=G.nodes[task_id - 1]['duration'],
                        nweight=-G.nodes[task_id - 1]['duration'],
                    )
                    G.add_edge(
                        task_id, sink_task,
                        job_edge=True, machine_arc=-1,
                        # weight=G.nodes[task_id]['duration']
                    )
                else:
                    G.add_edge(
                        task_id - 1, task_id,
                        job_edge=True, machine_arc=-1,
                        # weight=G.nodes[task_id - 1]['duration'],
                        nweight=-G.nodes[task_id - 1]['duration']
                    )

                # add machine edge
                if isinstance(m_id, int):
                    G.add_edge(
                        task_id, 'M' + str(m_id),
                        job_edge=False,
                        machine_arc=1,
                        weight=round(dur.item(),2)
                    )
                else:
                    for arc_to_m in m_id:
                        weight_arc = round(env.instance.proc_times_batch[0, task_id, arc_to_m].item(),2)
                        G.add_edge(
                            'M' + str(arc_to_m), task_id,
                            weight=weight_arc,
                            job_edge=False,
                            machine_arc=0,
                        )
        self.G = G

    def graph_rgb_array(self) -> np.ndarray:
        """
        #####################
        ##### Main Step #####
        #####################

        Wrapper for `nx` drawing operations. Draw networkx

        :param G:   the `nx.DiGraph` of an `DisjunctiveGraphJssEnv` instance
        :return:    a plot of the provided graph as rgb array.
        """

        plt.figure(dpi=self.dpi)
        plt.axis("off")
        plt.tight_layout()

        pos: Dict = nx.get_node_attributes(self.G, 'pos')  # node positions
        fig = mpl.pyplot.gcf()
        fig.set_size_inches(self.width, self.height)

        # draw nodes
        label_node = {}
        for task, data in self.G.nodes(data=True):
            nx.draw_networkx_nodes(self.G, pos,
                                   nodelist=[task],
                                   edgecolors=data["color"],
                                   node_color=data["color"] if data["scheduled"] else 'w',
                                   node_shape=data["node_shape"],
                                   **self.node_drawing_kwargs
                                   )
            label_node[task] = data['label']
        # draw node labels
        # nx.draw_networkx_labels(self.G, pos)
        nx.draw_networkx_labels(self.G, pos, label_node)

        # draw edges
        for from_, to_, data_dict in self.G.edges(data=True):
            if data_dict["job_edge"]:
                nx.draw_networkx_edges(self.G, pos,
                                       edgelist=[(from_, to_)],
                                       alpha=0.5,
                                       edge_color='gray',
                                       **self.edge_drawing_kwargs
                                       )
            elif data_dict["machine_arc"] == 0:
                nx.draw_networkx_edges(self.G, pos,
                                       edgelist=[(from_, to_)],
                                       alpha=0.5, edge_color=self.G.nodes[from_]["color"],
                                       arrowsize=1,
                                       )
            elif data_dict["machine_arc"] == 1:
                nx.draw_networkx_edges(self.G, pos,
                                       edgelist=[(from_, to_)],
                                       edge_color=self.G.nodes[from_]["color"],
                                       arrowsize=2
                                       )

        # draw edge labels
        labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels, font_size=9, label_pos=0.6)   # 可选alpha=0.5
        # change the label position to make it visible

        """# longest path corresponds to makespan of the jsp
        longest_path = nx.dag_longest_path(G)
        longest_path_edges = list(zip(longest_path, longest_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=longest_path_edges)"""

        # convert canvas to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        try:
            # with scientific mode enabled in pycharm this code works
            # no idea why enabling scientific mode in pycharm changes anything at all :o
            # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            w, h = fig.canvas.get_width_height()
            img = img.reshape((h, w, 3))
        except ValueError:
            w, h = fig.canvas.get_width_height()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=self.dpi)
            buf.seek(0)
            img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            buf.close()
            img = cv2.imdecode(img_arr, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.reshape((h, w, 3))
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # clear current frame
        plt.clf()
        plt.close('all')
        return img

    @staticmethod
    def render_rgb_array(vis, *, path, number, window_title: str = "Flexible Job Shop Scheduling",
                         wait: int = 1) -> None:
        """
        renders a rgb-array in an `cv2` window.
        the window will remain open for `:param wait:` ms or till the user presses any key.

        :param vis:             the rgb-array to render.
        :param window_title:    the title of the `cv2`-window
        :param path:            the fold to save jpg
        :param number:          the number of pic
        :param wait:            time in ms that the `cv2`-window is open.
                                if `None`, then the window will remain open till a keyboard occurs.

        :return:
        """
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_title, vis)
        # https://stackoverflow.com/questions/64061721/opencv-to-close-the-window-on-a-specific-key
        cv2.imwrite(path + str(number) + '.jpg', vis)
        k = cv2.waitKey(wait) & 0xFF
        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()

    def graph_widow_primary(self, wait=100, path='./default_fold_pic/'):
        number = 0
        self.graph_set(self.env)  # MAKE networkx graph
        vis = self.graph_rgb_array()  # TO RGB
        self.render_rgb_array(vis, path=path, number=0, window_title='gross', wait=1)

    def graph_window_middle(self, action, path='./default_fold_pic/', number=1, wait=1):
        # self.graph_set(self.env)    # MAKE networkx graph
        self.step(action)
        vis2 = self.graph_rgb_array()  # TO RGB
        self.render_rgb_array(vis2, path=path, number=number, window_title='SECOND_plot', wait=wait)

    def step(self, action_1_step):
        """each step will change G"""
        # basic step info
        unflatten = action_1_step.tolist()
        Ope, Machine, Job = [item for l in unflatten for item in l]
        Start, End = self.env.schedule.schedules_batch[0, Ope, [2, 3]].tolist()

        # node update
        node = self.G.nodes[Ope]
        old_ms = node['machine']  # 登记已有边
        node['machine'], node['scheduled'], node['color'] = Machine, True, self.machine_colors[Machine]
        node['start_time'], node['end_time'] = Start, End

        for old_m in old_ms:
            if old_m != Machine:
                self.G.remove_edge('M' + str(old_m), Ope)  # delete edge O-M PAIR

        # update scheduled_array as dataframe
        append_array = pd.DataFrame([{"Task": Job, "Resource": Machine, "Start": Start, "Finish": End}])
        self.scheduled_array = pd.concat([self.scheduled_array, append_array], ignore_index=True)
        print(self.scheduled_array)
        pass

    @staticmethod
    def draw_env_nodes(env, path, method, dpi=100, width=10, height=6, wait=1):
        """画图"""
        # 1. 原始环境
        GG = draw_graph(env, dpi=dpi, width=width, height=height)
        GG.graph_widow_primary(path=path, wait=wait)
        # 2. 每一步
        for i in range(1, env.instance.num_opes + 1):
            new_env, action = example_random(env, method)
            GG.env = new_env
            # 在graph window middle 执行action
            GG.graph_window_middle(action, path=path, number=i, wait=wait)

        env.render(changeable_mode='p_d')


def example_random(env, method, step_output=1):
    state = env.state
    dones = env.schedule.done_batch
    done = dones.all()  # Unfinished at the beginning
    i = 0
    while ~done:
        i += 1
        action = method.go(env)
        # time
        time_move = env.feature.feat_mas_batch[0, 1, action[1, 0]]
        if time_move >= env.schedule.time:
            env.time_move_to_t(torch.tensor([time_move]))
        print(f'step {i}, action {action}')
        state, rewards, dones, _, _ = env.step(action)
        done = dones.all()
        if step_output <= i:
            break
    return env, action


if __name__ == "__main__":
    # 测试效果
    need_save = False
    import os
    import time

    local_time = time.strftime("%Y%m%d_%H_%M")

    if need_save:
        fold_path = "./" + local_time + './'
    else:
        fold_path = './default_fold_pic/'

    if not os.path.exists(fold_path):
        os.mkdir(fold_path)

    # prepare color and env
    color = plt.cm.tab20(np.linspace(0, 1, 20))
    c_map = plt.cm.get_cmap('tab20')

    default_path = 'case_studyA.fjs'

    # important! set the env
    env = MMBPEnv(ins_file=default_path, batch=1, device='cpu', render_mode='p_d', release_and_due='case_studyA.csv',
                  time_slot=0.1, color_type=plt.cm.rainbow(np.linspace(0, 1,5)),
                  maintenance=[[10,4,6]])

    from MMBP.Heuristics import Heuristic
    Heu_method = Heuristic('MWRM_EET')

    # default set: class: draw_graph
    """GG = draw_graph(env, dpi=100, width=10, height=6)
    GG.graph_widow_primary(path=fold_path, wait=1)

    # 开始求解


    for i in range(1, env.instance.num_opes + 1):
        new_env, action = example_random(env)
        GG.env = new_env
        # 在graph window middle 执行action
        GG.graph_window_middle(action, path=fold_path, number=i, wait=1)"""

    draw_graph.draw_env_nodes(env, fold_path, Heu_method)
    # draw_graph.draw_env_gantt(env, fold_path, example_random)
