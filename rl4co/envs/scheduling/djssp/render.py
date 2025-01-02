from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.colors import ListedColormap
from tensordict.tensordict import TensorDict

from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def render(td: TensorDict, batch_no: int):

    td = td.detach().cpu()


    # fetch the batch (scheduling instance)
    inst = td[batch_no]
    # number of jobs
    num_jobs = inst["job_ops_adj"].size(0)

    # create colormap and assign each job a color
    colors = plt.cm.tab10(np.linspace(0, 1, num_jobs))
    cmap = ListedColormap(colors)

    # extract machine-operation matrix
    assign = inst["ma_assignment"].nonzero()


    schedule = defaultdict(list)

    # iterate over machine-operation assignments
    for val in assign:
        # (machine,operation)
        machine = val[0].item()
        op = val[1].item()
        # start time of the operation
        start = inst["start_times"][val[1]]
        # ending time of the operation
        end = inst["finish_times"][val[1]]
        # save in schedule dictionary ["machine_no":(operation_id,start_time, end_time)]
        schedule[machine].append((op, start, end))

    _, ax = plt.subplots()

    #for job arrival lines
    #TODO: here td[] or inst[] ?
    arrival_times = td["job_arrival_times"][batch_no].tolist()
    start_op = td["start_op_per_job"].squeeze(0)


    # Plot horizontal bars for each task
    for ma, ops in schedule.items():
        for op, start, end in ops:

            # job of the operation
            job = inst["job_ops_adj"][:, op].nonzero().item()
            # x-axis(horizontal) bar
            ax.barh(
                ma, # y-axis of operation (machine)
                end - start, # operation duration
                left=start, # starting position
                height=0.6,
                color=cmap(job),
                edgecolor="black",
                linewidth=1,
                )
            # write op (operation_no) in the center of the bar
            ax.text(
                start + (end - start) / 2, ma, op, ha="center", va="center", color="white"
            )

            # Draw the job arrival line
            if torch.isin(op, start_op):
                for job_id, arrival_time in enumerate(arrival_times):
                    if job_id == job:
                        ax.plot(
                            [arrival_time, arrival_time],  # x-axis
                            [ma - 0.6, ma + 0.6],  # y-axis ; in range of machine section
                            color="red",
                            linestyle="--",
                            linewidth=1,
                        )
                        # Annotate the dashed line below the x-axis
                        ax.text(
                                arrival_time,
                                ma - 0.6,
                                f"Job {job_id} ",
                                rotation=90,
                                va="top",  # Align the text to the top of the annotation point
                                ha="center",
                                color="red",
                                fontsize=8,
                        )
            # mark all the machine breakdowns
            for index in range(int((inst["machine_breakdowns"][ma].size(0)) / 2)):
                breakdown_time = inst["machine_breakdowns"][ma, index * 2]
                breakdown_duration = inst["machine_breakdowns"][ma, index * 2 + 1]
                ax.barh(
                    ma,  # y-axis of operation (machine)
                    breakdown_duration,  # operation duration
                    left=breakdown_time,  # starting position
                    height=1,  #  0.6 idi
                    color="red",
                    edgecolor="red",
                    linewidth=1,
                )


            # mark the breakdowns only when a machine is operating on a job
            # shape [number_of_machines , num_max_breakdowns]
            # inst["machine_breakdowns"]
            # iterate over each breakdown-duration pairs
            # for index  in range(int((inst["machine_breakdowns"][ma].size(0))/2)):
            #     breakdown_time = inst["machine_breakdowns"][ma,index*2]
            #     if (start < breakdown_time < end):
            #         breakdown_duration = inst["machine_breakdowns"][ma, index * 2 +1 ]
            #         ax.barh(
            #             ma,  # y-axis of operation (machine)
            #             breakdown_duration,  # operation duration
            #             left=breakdown_time,  # starting position
            #             height=1, # bvuradi 0.6 idi
            #             color="red",
            #             edgecolor="red",
            #             linewidth=1,
            #         )








        # Determine the minimum and maximum times
        min_time = min(val[1].item() for val in assign)
        # 9999.0 -> operation is not scheduled
        max_time = max(time for time in inst["finish_times"] if time != 9999.0) + 100

        # Set the x-axis range
        ax.set_xlim(min_time, max_time)




    # Set labels and title
    ax.set_yticks(range(len(schedule)))
    ax.set_yticklabels([f"Machine {i}" for i in range(len(schedule))])
    ax.set_xlabel("Time")
    ax.set_title("Gantt Chart")

    # Add a legend for class labels
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(num_jobs)]
    ax.legend(
        handles,
        [f"Job {label}" for label in range(num_jobs)],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    plt.tight_layout()
    # Show the Gantt chart
    plt.show()



