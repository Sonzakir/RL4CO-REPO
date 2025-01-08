import os

from functools import partial
from typing import List

import numpy as np
import torch

from tensordict.tensordict import TensorDict
from torch.nn.functional import one_hot


from rl4co.envs.common.utils import Generator
from rl4co.envs.scheduling.djssp.parser import get_max_ops_from_files, read
from rl4co.utils.pylogger import get_pylogger


log = get_pylogger(__name__)


class DJSSPGenerator(Generator):
    """
    Input Data Generator for the Dynamic Job Shop Scheduling Problem
    GOAL: Create DJSP Instances that can be used to train RL Algorithms

    Args:
        num_machine: number of machines
        num_job: number of jobs
        min_processing_time: minimum time required for a machine to process an operation #TODO: burada minimum lowerbound mi demis
        max_processing_time: maximum time required for a machine to process an operation
        one2one_ma_map: whether each machine should have exactly one operation per job

        DYNAMIC ARGUMENTS
        #TODO: MTBF AND MTTR as a parameter?
        MACHINE RELATED:
            mtbf: mean time between failures (for machine breakdowns)
            mttr: mean time to repair  (for machine breakdowns)

    Returns:
          A TensorDict with the following key:
            start_op_per_job [batch_size, num_jobs]: first operation of each job
            end_op_per_job [batch_size, num_jobs]: last operation of each job
            proc_times [batch_size, num_machines , total_n_ops]: estimated processing time of operations on machines
            pad_mask [batch_size, total_n_ops]: padded mask of operations on machines

            DYNAMIC ATTRIBUTES:
            actual_proc_times [batch_size, num_machines , total_n_ops]: actual processing time of operations on machines
            machine_breakdowns [Array of length batch_size-> index = machine_idx
                                                            TIME : timestamp when the breakdown occurred
                                                            DURATION : duration of the breakdown
                                                            ] : machine breakdown of each machine
            job arrival times (batch_size , num_jobs): Each entry indicates the arrival time of the specific job
    """

    def __init__(
        self,
        num_jobs: int = 6,
        num_machines: int = 6,
        min_ops_per_job: int = None,
        max_ops_per_job: int = None,
        min_processing_time: int = 1,
        max_processing_time: int = 99,
        one2one_ma_map: bool = True,
        mtbf: int = 20,
        mttr: int = 3,
        **unused_kwargs: object,
    ) -> object:
        super().__init__(**unused_kwargs)
        self.num_jobs = num_jobs
        self.num_mas = num_machines
        self.min_ops_per_job = min_ops_per_job or self.num_mas
        self.max_ops_per_job = max_ops_per_job or self.num_mas
        self.min_processing_time = min_processing_time
        self.max_processing_time = max_processing_time
        self.one2one_ma_map = one2one_ma_map

        # DYNAMIC ATTRIBUTES
        self.mtbf = mtbf
        self.mttr = mttr

        #TODO: delete this

        if self.one2one_ma_map:
            assert self.min_ops_per_job == self.max_ops_per_job == self.num_mas
        # fixed number of total operations == maximum number of operation per job * number of jobs
        self.n_ops_max = self.max_ops_per_job * self.num_jobs

        # Argument chekcer
        if len(unused_kwargs) > 0:
            log.error(f"This kwargs is not yet implemented{unused_kwargs}")

    def _simulate_estimated_processing_times(self, bs, n_ops_max) -> torch.Tensor:
        """
        generates random processing times for operations on machines (estimated processing time)
        eq. to _simulate_processing_times from the jssp\generator.py
            bs = batch size
            n_ops_max = maximum number of operations per job
        """
        if self.one2one_ma_map:
            ops_machine_ids = (
                torch.rand((*bs, self.num_jobs, self.num_mas))
                .argsort(dim=-1)
                .flatten(1, 2)
            )
        else:
            ops_machine_ids = torch.randint(
                low=0,
                high=self.num_mas,
                size=(*bs, n_ops_max),
            )
        ops_machine_adj = one_hot(ops_machine_ids, num_classes=self.num_mas)

        # (bs, max_ops, machines)
        proc_times = torch.ones((*bs, n_ops_max, self.num_mas))
        proc_times = torch.randint(
            self.min_processing_time,
            self.max_processing_time + 1,
            size=(*bs, self.num_mas, n_ops_max),
        )

        # remove proc_times for which there is no corresponding ma-ops connection
        proc_times = proc_times * ops_machine_adj.transpose(1, 2)
        # in JSSP there is only one machine capable to process an operation
        assert (proc_times > 0).sum(1).eq(1).all()
        return proc_times.to(torch.float32)

    # TODO: input can be different
    #problem : this cuases increased processing times when we try to train the model
    # def _simulate_actual_processing_times(self, td) -> torch.Tensor:
    #     """
    #     generates actual processing times for operations on machines (stochastic processing time)
    #         td = TensorDict of estimated processing times
    #              -> (from _simulate_estimated_processing_times() )
    #     NOTES:
    #           1. variance can be adjusted
    #           2. normal distribution can be changed
    #     """
    #     variance = 0.1
    #     for i in range(td.shape[0]):
    #         for j in range(td.shape[1]):
    #             for z in range(td.shape[2]):
    #                 # here: Normal Distribution
    #                 stochastic_noise = np.random.normal(0, variance * td[i, j, z])
    #                 actual_time = max(0, td[i, j, z] + stochastic_noise)
    #                 td[i, j, z] = actual_time
    #     return td
    #################################################################################################
    def _simulate_actual_processing_times(self, bs, n_ops_max) -> torch.Tensor:
        """
        generates actual processing times for operations on machines (stochastic processing time)
            td = TensorDict of estimated processing times
                 -> (from _simulate_estimated_processing_times() )
        NOTES:
              1. variance can be adjusted
              2. normal distribution can be changed
        """
        """
                generates random processing times for operations on machines (estimated processing time)
                eq. to _simulate_processing_times from the jssp\generator.py
                    bs = batch size
                    n_ops_max = maximum number of operations per job
                """
        if self.one2one_ma_map:
            ops_machine_ids = (
                torch.rand((*bs, self.num_jobs, self.num_mas))
                .argsort(dim=-1)
                .flatten(1, 2)
            )
        else:
            ops_machine_ids = torch.randint(
                low=0,
                high=self.num_mas,
                size=(*bs, n_ops_max),
            )
        ops_machine_adj = one_hot(ops_machine_ids, num_classes=self.num_mas)

        # (bs, max_ops, machines)
        proc_times = torch.ones((*bs, n_ops_max, self.num_mas))
        proc_times = torch.clip(
            torch.normal(
                mean=(self.min_processing_time + self.max_processing_time) / 2,
                std=(self.max_processing_time - self.min_processing_time) / 6,
                size=(*bs, self.num_mas, n_ops_max),
            ),
            min=self.min_processing_time,
            max=self.max_processing_time,
        )

        # remove proc_times for which there is no corresponding ma-ops connection
        proc_times = proc_times * ops_machine_adj.transpose(1, 2)
        # in JSSP there is only one machine capable to process an operation
        assert (proc_times > 0).sum(1).eq(1).all()
        return proc_times.to(torch.float32)
####################################################################################################
    def generate_machine_breakdowns(self,batch_size, num_machines, num_breakdowns, lambda_mtbf, lambda_mttr, max_time=10000):
        """
        Generates a tensor representing machine breakdowns.

        Args:
            batch_size (int): Number of batches.
            num_machines (int): Number of machines per batch.
            num_breakdowns (int): Maximum number of breakdowns per machine.
            lambda_mtbf (float): Mean time between failures.
            lambda_mttr (float): Mean time to repair.
            max_time (int): Maximum simulation time.

        Returns:
            torch.Tensor: Shape (batch_size, num_machines, num_breakdowns * 2).
        """
        torch.manual_seed(77)

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

        return breakdowns

###########################################################################################################################
        # torch.manual_seed(77)
        # # Exponential distributions for MTBF and MTTR
        # mtbf_distribution = torch.distributions.Exponential(1 / lambda_mtbf)
        # mttr_distribution = torch.distributions.Exponential(1 / lambda_mttr)
        #
        # # Sample occurrence times and durations
        # breakdown_occurrence_times = mtbf_distribution.sample((batch_size, num_machines, num_breakdowns))
        # breakdown_durations = mttr_distribution.sample((batch_size, num_machines, num_breakdowns))
        #
        # # Convert occurrence times into cumulative times
        # cumulative_times = torch.cumsum(breakdown_occurrence_times, dim=-1)
        #
        # # Ensure cumulative times stay within max_time
        # cumulative_times = torch.where(cumulative_times < max_time, cumulative_times, max_time)
        #
        # # Randomly decide how many breakdowns occur for each machine
        # num_actual_breakdowns = torch.randint(
        #     0, num_breakdowns + 1, (batch_size, num_machines)
        # )  # Random integer [0, num_breakdowns]
        #
        # # Create the final breakdown tensor
        # breakdowns = torch.zeros(batch_size, num_machines, num_breakdowns * 2)
        #
        # for i in range(num_breakdowns):
        #     # Mask to include only machines that still need more breakdowns
        #     mask = num_actual_breakdowns > i
        #
        #     # Fill occurrence times at even indices
        #     breakdowns[:, :, i * 2] = torch.where(
        #         mask, cumulative_times[:, :, i], torch.zeros_like(cumulative_times[:, :, i])
        #     )
        #
        #     # Fill durations at odd indices
        #     breakdowns[:, :, i * 2 + 1] = torch.where(
        #         mask, breakdown_durations[:, :, i], torch.zeros_like(breakdown_durations[:, :, i])
        #     )
        #
        # return breakdowns

    ####################################################################################################

    # this is the last added simulate_machine_breakdowns
    # to ensure consistency with the environment
    def _simulate_machine_breakdowns_(self, td, lambda_mtbf, lambda_mttr):

        # The mean time between failure and mean time off line subject to exponential distribution
        # (from E.S) assumin that MTBF-MTTR obey the exponential distribution
        # TODO: or we can use torch.Tensor.exponential_ in here to
        # TODO: what is the really difference ???
        mtbf_distribtuion = torch.distributions.Exponential(1 / lambda_mtbf)
        mttr_distribution = torch.distributions.Exponential(1 / lambda_mttr)
        # In some papers seed are used to ensure reproducibility
        torch.manual_seed(seed=77)

        # In two paper Machine Failure Time Percentage is used but i dont understand the purpose of it
        MFTp = lambda_mttr / (lambda_mttr + lambda_mtbf)


        # [batch_number]  [breakdowns of the machine in the batch]
        batch_breakdowns = []
        for _ in range(td.size(0)):
            # machine_idx_sorted version
            # [machine_idx, occurence_time , duration]
            breakdowns = {}

            for machine_idx in range(0, self.num_mas):

                current_time = 0

                machine_idx_breakdowns = []
                # TODO: harcoded maximal processing time !!!
                while current_time < 10000:

                    # machine failure occurence time
                    failure_occ_time = mtbf_distribtuion.sample().item() + current_time
                    failure_occ_time = mtbf_distribtuion.sample().item() + current_time
                    # advance time
                    current_time += failure_occ_time
                    # the machine repair time
                    machine_repair_time = mttr_distribution.sample().item()
                    # machine cannot break again, while being repaired -> therefore advance the time
                    current_time += machine_repair_time

                    # still, current time must be less than max_processing time
                    if 10000 >= current_time:
                        machine_idx_breakdowns.append(
                            {"TIME": failure_occ_time, "DURATION": machine_repair_time}
                        )
                    breakdowns[machine_idx] = machine_idx_breakdowns

            batch_breakdowns.append(breakdowns)
        return batch_breakdowns


    # For simualting machine breakdowns without considering MTBF-MTOL
    # currently  NOT USED
    def _simulate_machine_breakdowns(self):
        # seeds -> currently random numbers
        time_seed = 8484
        machine_seed = 1245
        # RNGenerators
        from random import random
        rng_time = random.Random(time_seed)
        rng_machine = random.Random(machine_seed)

        # TODO- do we really need lower-upper bounds
        earliest = 0  # earliest time machine-breakdown can occur
        # TODO: upperbound can be adjusted with the information from the td["time"]
        # or it can be taken from the jssp instance if the one with the lower-upperbound is given
        latest = 10000  # latest time machine-breakdown can occur

        # TODO: this variables can be adjusted or replaced with MTBF-MTOL
        lower_duration_bound = 2
        upper_duration_bound = 2000
        machine_breakdowns = {}
        # machine id's starting from the zero since in ma_assignments adjacency matrix
        # the machine_id's starting from the zero
        for machine_idx in range(0, self.num_mas + 1):
            # number of breakdowns for current machine
            # currently between 0-3 but can be adjusted
            num_breakdown = rng_machine.randint(0, 3)
            # array to save the breakdowns of the machine with ID= machine_idx
            mac_idx_breakdown = []

            for _ in range(num_breakdown):
                # occurence time of the breakdown
                time = rng_time.randint(earliest, latest)
                # duration of the breakdown
                duration = rng_machine.randint(lower_duration_bound, upper_duration_bound)

                # save the breakdown
                mac_idx_breakdown.append({"TIME": time, "DURATION": duration})
            # Save the breakdowns of the machine in the machine_breakdowns dict
            machine_breakdowns[machine_idx] = mac_idx_breakdown

        return machine_breakdowns

    #  as in many papers, i am going to implement machine breakdowns using the MTBF-MTOL
    # here MTBF and MTTR can be optionally attribute of the environment
    # non-chronological : machine sorted version
    # from the real time scheduling in job manufacturing paper
    def _simulate_machine_breakdowns_with_mtbf_mttr(self, bs, lambda_mtbf, lambda_mttr):
        assert (
            self.max_processing_time >= lambda_mtbf
        ), "MTBF cannot be greater than maximum processing time"
        assert (
            self.max_processing_time >= lambda_mttr
        ), "MTTR cannot be greater than maximum processing time"
        # The mean time between failure and mean time off line subject to exponential distribution
        # (from E.S) assumin that MTBF-MTTR obey the exponential distribution
        # or we can use torch.Tensor.exponential_ in here to
        # TODO: what is the really difference ???
        mtbf_distribtuion = torch.distributions.Exponential(1 / lambda_mtbf)
        mttr_distribution = torch.distributions.Exponential(1 / lambda_mttr)
        # In some papers seed are used to ensure reproducibility
        torch.manual_seed(seed=77)

        # In two paper Machine Failure Time Percentage is used but i dont understand the purpose of it
        MFTp = lambda_mttr / (lambda_mttr + lambda_mtbf)

        # [batch_number]  [breakdowns of the machine in the batch]
        batch_breakdowns = []
        for _ in range(bs[0]):
            # machine_idx_sorted version
            # [machine_idx, occurence_time , duration]
            breakdowns = {}

            for machine_idx in range(0, self.num_mas):

                current_time = 0

                machine_idx_breakdowns = []
                while current_time < self.max_processing_time:

                    # machine failure occurence time
                    failure_occ_time = mtbf_distribtuion.sample().item() + current_time
                    failure_occ_time = mtbf_distribtuion.sample().item() + current_time
                    # advance time
                    current_time += failure_occ_time
                    # the machine repair time
                    machine_repair_time = mttr_distribution.sample().item()
                    # machine cannot break again, while being repaired -> therefore advance the time
                    current_time += machine_repair_time

                    # still, current time must be less than max_processing time
                    if self.max_processing_time >= current_time:
                        machine_idx_breakdowns.append(
                            {"TIME": failure_occ_time, "DURATION": machine_repair_time}
                        )
                    breakdowns[machine_idx] = machine_idx_breakdowns

            batch_breakdowns.append(breakdowns)
        return batch_breakdowns

    # DYNAMIC JOB ARRIVAL
    def _generate_random_job_arrivals(self, bs, number_of_jobs, E_new):
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

        ################### ORIGINAL ONE ###########################################
        # job_arrival_times = []
        # for _ in range(bs[0]):
        #     exponential_distribution = torch.distributions.Exponential(1 / E_new)
        #     interarrival_times_between_operations = exponential_distribution.sample(
        #         (number_of_jobs,)
        #     )
        #
        #     # cumulative sum to calculate the arrival times
        #     # TODO: here i assummed that jobs are coming one after another
        #     # but i can code them as totally independent too
        #     arrival_time = torch.cumsum(interarrival_times_between_operations, 0)
        #     job_arrival_times.append(arrival_time)
        # job_arrival_times = torch.stack(job_arrival_times)
        #######################################################################
        return job_arrival_times

    def _generate(self, batch_size) -> TensorDict:
        # simulate how many operations each job has
        n_ope_per_job = torch.randint(
            self.min_ops_per_job,
            self.max_ops_per_job + 1,
            size=(*batch_size, self.num_jobs),
        )

        # determine the total number of operations per batch instance (which may differ)
        n_ops_batch = n_ope_per_job.sum(1)  # (bs)
        # determine the maximum total number of operations over all batch instances
        n_ops_max = self.n_ops_max or n_ops_batch.max()

        # generate a mask, specifying which operations are padded
        pad_mask = torch.arange(n_ops_max).unsqueeze(0).expand(*batch_size, -1)
        pad_mask = pad_mask.ge(n_ops_batch[:, None].expand_as(pad_mask))

        # determine the id of the end operation for each job
        end_op_per_job = n_ope_per_job.cumsum(1) - 1

        # determine the id of the starting operation for each job
        # (bs, num_jobs)
        start_op_per_job = torch.cat(
            (
                torch.zeros((*batch_size, 1)).to(end_op_per_job),
                end_op_per_job[:, :-1] + 1,
            ),
            dim=1,
        )


        # DYNAMIC ATTRIBUTES

        # stochastic processing time
        # simulate actual processing times for machine-operation pairs
        # (bs, num_mas, n_ops_max)
        actual_proc_times = self._simulate_actual_processing_times(batch_size, n_ops_max)

        # since in dynamic job shop scheduling problem we are using stochastic processing times with noises
        # we do not use the processing times that is generated randomly (as in JSSP)
        # however if you want to use the processing times as in JSSP you can get the processing times like this :
        # simulate processing times for machine-operation pairs
        # (bs, num_mas, n_ops_max)
        #proc_times = self._simulate_estimated_processing_times(batch_size, n_ops_max)




        # dynamic job arrival_times :torch Tensor
        # shape : ( batch_size ,num_jobs)
        arrival_times = self._generate_random_job_arrivals(batch_size[0],
                                                           number_of_jobs = self.num_jobs ,
                                                           E_new = 20
                                                           )





        number_of_macimum_breakdowns = 17
        tensor_shape = (*batch_size, self.num_mas, number_of_macimum_breakdowns *2 )
        machine_breakdown_tensor = torch.zeros(tensor_shape)
        #TODO: NEWLY ADDED MNACHINE BREAKDOWN WITH THE GENERATE MACHINE BREAKDOWNS!
        machine_breakdown_tensor = self.generate_machine_breakdowns(batch_size= batch_size[0],
                                                                    num_machines= self.num_mas,
                                                                    num_breakdowns= number_of_macimum_breakdowns,
                                                                    lambda_mtbf= self.mtbf,
                                                                    lambda_mttr=self.mttr,
                                                                    max_time=10000
                                                                    )


        ############################# old machine breakdown#######################################################################
        # ma_breakdowns = self._simulate_machine_breakdowns_(td,self.mtbf,self.mttr)
        # print("this is machine_breakdown_tensor", machine_breakdown_tensor)
        # for batch_no in range(*batch_size):
        #     breakdowns_in_batch =ma_breakdowns[batch_no]
        #     for machine_no in range(0, len(breakdowns_in_batch)):
        #         current_machine = breakdowns_in_batch[machine_no]
        #         for breakdown_no in range(len(current_machine)):
        #             # Note: I set the third dimension to 16, so I used 16 here. If I change that value later,
        #             # I will need to update this accordingly
        #             # Even-numbered indices represent breakdown times
        #             machine_breakdown_tensor[batch_no, machine_no, (breakdown_no * 2)] = current_machine[breakdown_no]["TIME"]
        #             # Odd-numbered indices represent breakdown durations
        #             machine_breakdown_tensor[batch_no, machine_no, (breakdown_no * 2 + 1)] = current_machine[breakdown_no]["DURATION"]
        # td["machine_breakdowns"] = machine_breakdown_tensor
        ###############################################################################################################################



        td = TensorDict(
            {
                "start_op_per_job": start_op_per_job,
                "end_op_per_job": end_op_per_job,
                "proc_times": actual_proc_times,                 #  estimated -> actual_proc_times
                "pad_mask": pad_mask,
                "machine_breakdowns" : machine_breakdown_tensor,        #  machine breakdowns
                "job_arrival_times" : arrival_times              #  job arrival times
            },
            batch_size=batch_size,
        )


        return td







class DJSSPFileGenerator(Generator):
    """Data generator for the Job-Shop Scheduling Problem (JSSP) using instance files

    Args:
        path: path to files

    Returns:
        A TensorDict with the following key:
            start_op_per_job [batch_size, num_jobs]: first operation of each job
            end_op_per_job [batch_size, num_jobs]: last operation of each job
            proc_times [batch_size, num_machines, total_n_ops]: processing time of ops on machines
            pad_mask [batch_size, total_n_ops]: not all instances have the same number of ops, so padding is used

    """

    def __init__(self, file_path: str, n_ops_max: int = None, **unused_kwargs):

        self.files = (
            [file_path] if os.path.isfile(file_path) else self.list_files(file_path)
        )
        self.num_samples = len(self.files)

        if len(unused_kwargs) > 0:
            log.error(f"Found {len(unused_kwargs)} unused kwargs: {unused_kwargs}")

        if len(self.files) > 1:
            n_ops_max = get_max_ops_from_files(self.files)

        ret = map(partial(read, max_ops=n_ops_max), self.files)

        td_list, num_jobs, num_machines, max_ops_per_job = list(zip(*list(ret)))
        num_jobs, num_machines = map(lambda x: x[0], (num_jobs, num_machines))
        max_ops_per_job = max(max_ops_per_job)

        self.td = torch.cat(td_list, dim=0)
        self.num_mas = num_machines
        self.num_jobs = num_jobs
        self.max_ops_per_job = max_ops_per_job
        self.start_idx = 0

    def _generate(self, batch_size: List[int]) -> TensorDict:
        batch_size = np.prod(batch_size)
        if batch_size > self.num_samples:
            log.warning(
                f"Only found {self.num_samples} instance files, but specified dataset size is {batch_size}"
            )
        end_idx = self.start_idx + batch_size
        td = self.td[self.start_idx : end_idx]
        self.start_idx += batch_size
        if self.start_idx >= self.num_samples:
            self.start_idx = 0

        ## Add dynamic attributes
        #E_new = 20  # Average inter-arrival time
        #job_arrival_times = self._generate_random_job_arrivals(batch_size=(batch_size,),
        #                                                       number_of_jobs=self.num_jobs,
        #                                                       E_new=E_new)
        #machine_breakdowns = self._simulate_machine_breakdowns_with_mtbf_mttr(
        #    bs=(batch_size,), lambda_mtbf=20, lambda_mttr=3
        #)
#
        #    # Update TensorDict with dynamic keys
        ## Add the dynamic keys
        #td = td.update({
        #    "job_arrival_times": job_arrival_times,
        #    "machine_breakdowns": machine_breakdowns
        #})

        return td

    @staticmethod
    def list_files(path):
        files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
        assert len(files) > 0, "No files found in the specified path"
        return files
