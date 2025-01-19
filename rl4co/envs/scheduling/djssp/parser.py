from pathlib import Path
from typing import List, Tuple, Union

import torch

from tensordict import TensorDict


ProcessingData = List[Tuple[int, int]]


def parse_job_line(line: Tuple[int]) -> Tuple[ProcessingData]:
    """
    Parses a DJSSP job data line of the following form:

        <num operations> * (<machine> <processing time>)

    In words, a line consist of n_ops pairs of values, where the first value is the
    machine identifier and the second value is the processing time of the corresponding
    operation-machine combination

    Note that the machine indices start from 1, so we subtract 1 to make them
    zero-based.
    """

    operations = []
    i = 0

    while i < len(line):
        machine = int(line[i])
        duration = int(line[i + 1])
        operations.append((machine, duration))
        i += 2

    return operations


def get_n_ops_of_instance(file):
    lines = file2lines(file)
    jobs = [parse_job_line(line) for line in lines[1:]]
    n_ope_per_job = torch.Tensor([len(x) for x in jobs]).unsqueeze(0)
    total_ops = int(n_ope_per_job.sum())
    return total_ops


def get_max_ops_from_files(files):
    return max(map(get_n_ops_of_instance, files))


def read(loc: Path, max_ops=None):
    """
    Reads an DJSSP instance.

    Args:
        loc: location of instance file
        max_ops: optionally specify the maximum number of total operations (will be filled by padding)

    Returns:
        instance: the parsed instance
    """
    lines = file2lines(loc)

    # First line contains metadata.
    num_jobs, num_machines = lines[0][0], lines[0][1]

    # The remaining lines contain the job-operation data, where each line
    # represents a job and its operations.
    jobs = [parse_job_line(line) for line in lines[1:]]
    n_ope_per_job = torch.Tensor([len(x) for x in jobs]).unsqueeze(0)
    total_ops = int(n_ope_per_job.sum())
    if max_ops is not None:
        assert total_ops <= max_ops, "got more operations then specified through max_ops"
    max_ops = max_ops or total_ops
    max_ops_per_job = int(n_ope_per_job.max())

    end_op_per_job = n_ope_per_job.cumsum(1) - 1
    start_op_per_job = torch.cat((torch.zeros((1, 1)), end_op_per_job[:, :-1] + 1), dim=1)

    pad_mask = torch.arange(max_ops)
    pad_mask = pad_mask.ge(total_ops).unsqueeze(0)

    proc_times = torch.zeros((num_machines, max_ops))
    op_cnt = 0
    for job in jobs:
        for ma, dur in job:
            # subtract one to let indices start from zero
            proc_times[ma - 1, op_cnt] = dur
            op_cnt += 1
    proc_times = proc_times.unsqueeze(0)
    # stochastic processing times
    actual_processing_times = _simulate_actual_processing_times(proc_times)
    # job arrival times
    job_arrival_times = _generate_random_job_arrivals(start_op_per_job.size(0), start_op_per_job.size(1),20)
    td = TensorDict(
        {
            "start_op_per_job": start_op_per_job,
            "end_op_per_job": end_op_per_job,
            "proc_times": actual_processing_times,
            "pad_mask": pad_mask,
            "job_arrival_times": job_arrival_times,
        },
        batch_size=[1],
    )


    td["job_arrival_times"] =  _generate_random_job_arrivals(td["start_op_per_job"].size(0), td["start_op_per_job"].size(1),20)
    return td, num_jobs, num_machines, max_ops_per_job


def file2lines(loc: Union[Path, str]) -> List[List[int]]:
    with open(loc, "r") as fh:
        lines = [line for line in fh.readlines() if line.strip()]

    def parse_num(word: str):
        return int(word) if "." not in word else int(float(word))

    return [[parse_num(x) for x in line.split()] for line in lines]


####################################################################
import numpy as np
def _simulate_actual_processing_times(td) -> torch.Tensor:
    """
    generates actual processing times for operations on machines (stochastic processing time)
        td = TensorDict of estimated processing times
             -> (from _simulate_estimated_processing_times() )
    NOTES:
          1. variance can be adjusted
          2. normal distribution can be changed
    """
    variance = 0.1
    for i in range(td.shape[0]):
        for j in range(td.shape[1]):
            for z in range(td.shape[2]):
                # here: Normal Distribution
                stochastic_noise = np.random.normal(0, variance * td[i, j, z])
                actual_time = max(0, td[i, j, z] + stochastic_noise)
                td[i, j, z] = actual_time
    return td

# generate random arrival times for files
# DYNAMIC JOB ARRIVAL
def _generate_random_job_arrivals(bs, number_of_jobs, E_new):
    """
    From Smart scheduling of dynamic job shop based on discrete event simulation and deep reinforcement learning:
    ` The arrival of subsequent new jobs follows a Poisson distribution, hence the interarrival time between
    two successive new job is subjected to exponential distribution with average value Enew
    Average value of interarrival time Enew is 20. `
    """
    # Create exponential distribution for interarrival times
    exponential_distribution = torch.distributions.Exponential(1 / E_new)
    # Sample interarrival times for all batches and jobs at once
    # Shape: (bs, number_of_jobs)
    interarrival_times = exponential_distribution.sample((bs, number_of_jobs))
    # Compute cumulative arrival times for all jobs in all batches
    # Shape: (bs, number_of_jobs)
    job_arrival_times = torch.cumsum(interarrival_times, dim=-1)
    # rounded
    return torch.round(job_arrival_times , decimals= 3)


def generate_machine_breakdowns(batch_size, num_machines, num_breakdowns, lambda_mtbf, lambda_mttr, max_time=10000):
    """
    Generates a tensor representing machine breakdowns.
    Args:
        batch_size (int): Number of batches.
        num_machines (int): Number of machines per batch.
        num_breakdowns (int): Maximum number of breakdowns per machine.
        lambda_mtbf (float): Mean time between failures.
        lambda_mttr (float): Mean time to repair.
        max_time (int): Maximum simulation time. (upper bound)
    Returns:
        torch.Tensor: Shape (batch_size, num_machines, num_breakdowns * 2).
    """
    #TODO: In some papers seed are used to ensure reproducibility
    #torch.manual_seed(90)
    # Exponential distributions for MTBF and MTTR
    mtbf_distribution = torch.distributions.Exponential(1 / lambda_mtbf)
    mttr_distribution = torch.distributions.Exponential(1 / lambda_mttr)
    # Initialize cumulative times and breakdown tensor
    cumulative_times = torch.zeros(batch_size, num_machines)  # Tracks the current time per machine
    breakdowns = torch.zeros(batch_size, num_machines, num_breakdowns,
                             2)  # To store [occurrence time, repair duration]
    for i in range(num_breakdowns):
        # Sample failure occurrence times and repair durations
        failure_occ_times = mtbf_distribution.sample((batch_size, num_machines))
        repair_durations = mttr_distribution.sample((batch_size, num_machines))
        # Advance cumulative time by failure occurrence times
        cumulative_times += failure_occ_times
        # Mask to check if the breakdown occurs within max_time
        valid_mask = cumulative_times < max_time
        # Record breakdown times and durations only for valid breakdowns
        breakdowns[:, :, i, 0] = torch.where(valid_mask, cumulative_times, torch.tensor(0.0))
        breakdowns[:, :, i, 1] = torch.where(valid_mask, repair_durations, torch.tensor(0.0))
        # Advance cumulative time by repair durations (only for valid breakdowns)
        cumulative_times += torch.where(valid_mask, repair_durations, torch.tensor(0.0))
    # Reshape breakdowns tensor to match expected output (batch_size, num_machines, num_breakdowns * 2)
    breakdowns = breakdowns.reshape(batch_size, num_machines, num_breakdowns * 2)
    # round the machine breakdowns  this can be adjusted https://pytorch.org/docs/main/generated/torch.round.html
    return torch.round(breakdowns)