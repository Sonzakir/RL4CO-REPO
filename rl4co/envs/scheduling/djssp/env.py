import collections
import os

from ortools.sat.python import cp_model
from torchrl.data import Composite, Unbounded, Bounded

from rl4co.envs.scheduling.djssp.generator import DJSSPGenerator, DJSSPFileGenerator
from rl4co.envs.scheduling.fjsp import INIT_FINISH, NO_OP_ID
from rl4co.envs.scheduling.djssp.render import render
from rl4co.envs.scheduling.fjsp.utils import calc_lower_bound
from rl4co.envs.scheduling.jssp.env import JSSPEnv

from rl4co.utils.ops import gather_by_index, sample_n_random_actions
from einops import einsum, reduce
import torch
from tensordict.tensordict import TensorDict
from torch._tensor import Tensor



class DJSSPEnv(JSSPEnv):
    """Dynamic Job-Shop Scheduling Problem (DJSSP) environment
    At each step, the agent chooses a job. The operation to be processed next for the selected job is
    then executed on the associated machine. The reward is 0 unless the agent scheduled all operations of all jobs.
    In that case, the reward is (-)makespan of the schedule: maximizing the reward is equivalent to minimizing the makespan.
    NOTE: The DJSSP is a special case of the FJSP, when the number of eligible machines per operation is equal to one for all
    operations. Therefore, this environment is a subclass of the FJSP environment.
    Observations:
        - time: current time
        - next_op: next operation per job
        - proc_times: processing time of operation-machine pairs
        - pad_mask: specifies padded operations
        - start_op_per_job: id of first operation per job
        - end_op_per_job: id of last operation per job
        - start_times: start time of operation (defaults to 0 if not scheduled)
        - finish_times: finish time of operation (defaults to INIT_FINISH if not scheduled)
        - job_ops_adj: adjacency matrix specifying job-operation affiliation
        - ops_job_map: same as above but using ids of jobs to indicate affiliation
        - ops_sequence_order: specifies the order in which operations have to be processed
        - ma_assignment: specifies which operation has been scheduled on which machine
        - busy_until: specifies until when the machine will be busy
        - num_eligible: number of machines that can process an operation
        - job_in_process: whether job is currently being processed
        - job_done: whether the job is done

    Constrains:
        the agent may not select:
        - jobs that are done already
        - jobs that are currently processed

    Finish condition:
        - the agent has scheduled all operations of all jobs

    Reward:
        - the negative makespan of the final schedule

    Args:
        generator: JSSPGenerator instance as the data generator
        generator_params: parameters for the generator
        mask_no_ops: if True, agent may not select waiting operation (unless instance is done)
    """

    name = "djsp"

    def __init__(
        self,
        generator: DJSSPGenerator = None,
        generator_params: dict = {},
        mask_no_ops: bool = True,
        **kwargs,
    ):
        if generator is None:
            if generator_params.get("file_path", None) is not None:
                generator = DJSSPFileGenerator(**generator_params)
            else:
                generator = DJSSPGenerator(**generator_params)

        self.generator = generator
        super().__init__(generator, generator_params, mask_no_ops, **kwargs)





    def _reset(self, td: TensorDict = None, batch_size=None) -> TensorDict:
        self.set_instance_params(td)

        td_reset = td.clone()

        td_reset, n_ops_max = self._decode_graph_structure(td_reset)

        # schedule
        # starting operation of each job
        start_op_per_job = td_reset["start_op_per_job"]
        start_times = torch.zeros((*batch_size, n_ops_max))
        finish_times = torch.full((*batch_size, n_ops_max), INIT_FINISH)
        ma_assignment = torch.zeros((*batch_size, self.num_mas, n_ops_max))

        # reset feature space
        busy_until = torch.zeros((*batch_size, self.num_mas))
        # (bs, ma, ops)
        ops_ma_adj = (td_reset["proc_times"] > 0).to(torch.float32)
        # (bs, ops)
        num_eligible = torch.sum(ops_ma_adj, dim=1)



        td_reset = td_reset.update(
            {
                "start_times": start_times,
                "finish_times": finish_times,
                "ma_assignment": ma_assignment,
                "busy_until": busy_until,
                "num_eligible": num_eligible,
                "next_op": start_op_per_job.clone().to(torch.int64),
                "ops_ma_adj": ops_ma_adj,
                "op_scheduled": torch.full((*batch_size, n_ops_max), False),
                "job_in_process": torch.full((*batch_size, self.num_jobs), False),
                "reward": torch.zeros((*batch_size,), dtype=torch.float32),
                "time": torch.min(td_reset["job_arrival_times"] , dim=1).values,  # -> advance the starting time to the earliest time a job arrives
                "job_done": torch.full((*batch_size, self.num_jobs), False),
                "done": torch.full((*batch_size, 1), False),
            },
            # changes: "time": torch.zeros((*batch_size,)) -> torch.min(td["job_arrival_times"], dim=1).values
            # "time": torch.min(td["job_arrival_times"],dim=1).values,  # -> advance the starting time to the earliest time a job arrives

        )


        # check machine breakdowns and
        # update td["busy_until"] if necessary
        #td_reset = self._check_machine_breakdowns(td_reset)

        # mask the infeasible actions
        td_reset.set("action_mask", self.get_action_mask(td_reset))

        # add additional features to tensordict
        td_reset["lbs"] = calc_lower_bound(td_reset)
        td_reset = self._get_features(td_reset)



        return td_reset


    # #TODO COPY PASTE FROM GITHUB
    # # WARNING: Maybe here we have to clone the tensordict
    def _check_machine_breakdowns(self, td: TensorDict ):
    #     """
    #         Method to check for machine breakdowns in the environment.
    #         - td["machine_breakdowns"] is a tensor with the shape [batch_size, number of machines, number of maximum breakdowns].
    #         - The number of maximum breakdowns is currently set to 33 in the generator/environment.
    #         - Here, the times at which breakdowns occur correspond to entries where the number of maximum breakdowns is
    #         an even number, and the duration of that breakdown is given by the subsequent entry.
    #             - Occurence time of the first breakdown in batch x , machine y -> td["machine_breakdowns"][ x , y , 0 ]
    #             - The duration of the  first breakdown in batch x , machine y -> td["machine_breakdowns"][ x , y , 1 ]
    #         - Meaning
    #             - Breakdown occurence times td["machine_breakdowns"][batch_idx , machine_idx , x ] , where x= 0,2,4,6,8,....32
    #             - Breakdown duration times td["machine_breakdowns"][batch_idx , machine_idx , y ] , where x= 1,3,5,7,9,.....33
    #
    #
    #         Args:
    #             td: The state of the environment at the current time step (td["time"]).
    #
    #         Returns:
    #             The updated td with the modified td["busy_until"] entry.
    #             If a machine has a breakdown at the current time step (td["time"])
    #             -> the "busy_until" entry for that machine is adjusted to the time step when the machine is repaired.
    #     """
    #
    #     # shape : (batch_size, num_mas ,  number_of_max_breakdowns *2 )
    #     machine_breakdowns = td["machine_breakdowns"]
    #     # get the current time of each batch
    #     # add dimension at the last index for brodcasting
    #     # before unsqueeze (batch_size,) -> after: (batch_size,1)
    #     current_time = td["time"].unsqueeze(-1)
    #     # overwrite busy_until entry to indicate machine breakdown
    #     # shape : (batch_size, num_mas)
    #     busy_until = td["busy_until"]
    #
    #     # Get the machine breakdown occurence times and duration times
    #
    #     # even indices represesting the breakdown occurence times
    #     # therefore: select all element from the first dim(bs) and second dim (num_mas) and
    #     # select only even indices from the third dim (machine_breakdowns)
    #     # shape : (batch_size , num_mas , number_of_max_breakdowns)
    #     tensor_breakdown_occ_times = machine_breakdowns[:, :, 0::2]
    #
    #     # same here: odd indices representing the breakdown duration times
    #     # shape : (batch_size , num_mas , number_of_max_breakdowns)
    #     tensor_brekdown_durations = machine_breakdowns[:, :, 1::2]
    #
    #     # boolean tensor to check if a breakdown occurs at the current time
    #     # shape : (batch_size , num_mas , number_of_max_breakdowns)
    #     # bool_breakdown[x,y,z]= True -> in batch x ; machine y; z.breakdown occurs at the current decision time td["time"][x]
    #     # we need to add one dimension to the current time to match the shapes (e.g. to make both 3D tensors)
    #     bool_breakdown = (tensor_breakdown_occ_times == current_time.unsqueeze(-1))
    #
    #     # filter/ mask the inactive machine breakdown durations
    #     # and helps to reduce the computing time
    #     # if breakdown occurs get the duration from the breakdown durations tensor
    #     # otherwise  fill it with 0
    #     # shape : (batch_size , num_mas , number_of_max_breakdowns)
    #     curr_brekadown_durations = torch.where(bool_breakdown, tensor_brekdown_durations, torch.zeros_like(tensor_brekdown_durations))
    #
    #     # Find the first active breakdown for each machine
    #     # calculate the sum of the breakdown duration for each machine (total breakdown time)
    #     # shape : (batch_size , num_mas )
    #     sum_duration_time = curr_brekadown_durations.sum(dim=-1)
    #
    #     # expand current time to have the same shape as busy_until (batch_size, num_machines)
    #     expanded_current_time = current_time.expand(-1, busy_until.size(1))
    #
    #     # update the busy_until
    #     # if there is a breakdown : machine is busy(not available) until -> " current_time + duration of the breakdown"
    #     # othwerise do not busy_until entry remains unchanged
    #     td["busy_until"] = torch.where(
    #         sum_duration_time > 0,  # If a breakdown occurs
    #         expanded_current_time + sum_duration_time,  # Add active breakdown durations
    #         busy_until,  # Retain the original value if no breakdown occurs
    #     )
    #
    #
    #    ##########################################################################################################

       # old version (performance issues with attention model)
       # (bs , num_mas ,  n_max_breakdowns)
       #machine_breakdowns = td["machine_breakdowns"]

       #for batch_id in range(machine_breakdowns.size(0)):
       #   for machine_idx in range(machine_breakdowns.size(1)):
       #       # breakdowns of the machine in the current batch
       #       machine_idx_breakdowns = machine_breakdowns[batch_id,machine_idx]
       #       for breakdown_no in range(int(machine_breakdowns.size(2)/2)):
       #           # 0-2-4-6-8...-> breakdown occurence time
       #           # 1-3-5-7-9...-> breakdown duration
       #           # if current time == machine breakdown time
       #           # if machine_idx_breakdowns[breakdown_no]["TIME"] == td["time"][batch_id]:
       #           breakdown_occ_time = machine_idx_breakdowns[(breakdown_no)*2]
       #           breakdown_end_time = machine_idx_breakdowns[(breakdown_no)*2 +1] +  breakdown_occ_time
       #           if breakdown_occ_time <= td["time"][batch_id] <= breakdown_end_time:
       #               if td["busy_until"][batch_id][machine_idx] < td["time"][batch_id]:
       #                   print(
       #                       f"we mask this BATCH {batch_id} MACHINE {machine_idx} CURRENT TIME {td['time'][batch_id]}"
       #                       f" BREAKDOWN START TIME = {breakdown_occ_time}  BREAKDOWN END TIME = {breakdown_end_time}")
       #                   # machine is busy(not available) until -> " current_time + duration of the breakdown"
       #                   td["busy_until"][batch_id][machine_idx] = breakdown_end_time
       #                   print("####################################################################")
        return td









    def _get_features(self, td):
        td = super()._get_features(td)
        # get the id of the machine that executes an operation:
        # (bs, ops, ma)
        ops_ma_adj = td["ops_ma_adj"].transpose(1, 2)
        # (bs, jobs, ma)
        ma_of_next_op = gather_by_index(ops_ma_adj, td["next_op"], dim=1)
        # (bs, jobs)
        td["next_ma"] = ma_of_next_op.argmax(-1)

        # adjacency matrix specifying neighbors of an operation, including its
        # predecessor and successor operations and operations on the same machine
        ops_on_same_ma_adj = einsum(
            td["ops_ma_adj"], td["ops_ma_adj"], "b m o1, b m o2 -> b o1 o2 "
        )
        # concat pred, succ and ops on same machine
        adj = torch.cat((td["ops_adj"], ops_on_same_ma_adj.unsqueeze(-1)), dim=-1).sum(-1)
        # mask padded operations and those scheduled
        mask = td["pad_mask"] + td["op_scheduled"]
        adj.masked_fill_(mask.unsqueeze(1), 0)
        td["adjacency"] = adj

        return td

    def get_action_mask(self, td: TensorDict) -> Tensor:
        action_mask = self._get_job_machine_availability(td)
        if self.mask_no_ops:
            # masking is only allowed if instance is finished
            no_op_mask = td["done"]
        else:
            # if no job is currently processed and instance is not finished yet, waiting is not allowed
            no_op_mask = (
                td["job_in_process"].any(1, keepdims=True) & (~td["done"])
            ) | td["done"]
        # reduce action mask to correspond with logit shape
        action_mask = reduce(action_mask, "bs j m -> bs j", reduction="all")
        # NOTE: 1 means feasible action, 0 means infeasible action
        # (bs, 1 + n_j)
        mask = torch.cat((no_op_mask, ~action_mask), dim=1)

        return mask




    def _translate_action(self, td):
        job = td["action"]
        op = gather_by_index(td["next_op"], job, dim=1)
        # get the machine that corresponds to the selected operation
        ma = gather_by_index(td["ops_ma_adj"], op.unsqueeze(1), dim=2).nonzero()[:, 1]
        return job, op, ma




    def _step(self, td: TensorDict):
        
        # cloning required to avoid inplace operation which avoids gradient backtracking
        td = td.clone()

        # 1- Retrieve the input keys
        td["action"].subtract_(1)

        # (bs)
        dones = td["done"].squeeze(1)

        # specify which batch instances require which operation
        # batch instance that do not require action
        no_op = td["action"].eq(NO_OP_ID)
        no_op = no_op & ~dones
        req_op = ~no_op & ~dones

        # transition to next time for no op instances
        if no_op.any():
            td, dones = self._transit_to_next_time(no_op, td)

        # select only instances that perform a scheduling action
        td_op = td.masked_select(req_op)

        # 2- Execute Simulation (Write new observations)
        td_op = self._make_step(td_op)
        
        # 3- Write new observations
        # update the tensordict
        td[req_op] = td_op

        # set the action mask for the state of EACH batch AFTER make_step
        #TODO: masking machine breakdowns should be implemented in here (HOWEVER LOGIC PROBLEM IS HARD)
        # action mask
        td.set("action_mask", self.get_action_mask(td))

        step_complete = self._check_step_complete(td, dones)
        while step_complete.any():
            td, dones = self._transit_to_next_time(step_complete, td)
            td.set("action_mask", self.get_action_mask(td))
            step_complete = self._check_step_complete(td, dones)
        if self.check_mask:
            assert reduce(td["action_mask"], "bs ... -> bs", "any").all()

        if self.stepwise_reward:
            # if we require a stepwise reward, the change in the calculated lower bounds could serve as such
            lbs = calc_lower_bound(td)
            td["reward"] = -(lbs.max(1).values - td["lbs"].max(1).values)
            td["lbs"] = lbs
        else:
            td["lbs"] = calc_lower_bound(td)

        # add additional features to tensordict
        td = self._get_features(td)

        # we can render the environment
        # if td["done"].all():
        #     render(td, 0)

        return td

    def _make_step(self, td: TensorDict) -> TensorDict:

        """
        Environment transition function.
        In this make_step method, machine breakdowns are modeled as a tensor.
        If you want to model machine breakdowns using a NumPy ndarray,
            delete this method and rename the nd_array_make_step(self, td: TensorDict) method to make_step().
        """

        batch_idx = torch.arange(td.size(0))

        # 3*(#req_op)
        selected_job, selected_op, selected_machine = self._translate_action(td)

        # mark job as being processed
        td["job_in_process"][batch_idx, selected_job] = 1

        # mark op as schedules
        td["op_scheduled"][batch_idx, selected_op] = True

        # update machine state
        proc_time_of_action = td["proc_times"][batch_idx, selected_machine, selected_op]
        # we may not select a machine that is busy
        assert torch.all(td["busy_until"][batch_idx, selected_machine] <= td["time"])

        # update schedule
        td["start_times"][batch_idx, selected_op] = td["time"]
        td["finish_times"][batch_idx, selected_op] = td["time"] + proc_time_of_action

        td["ma_assignment"][batch_idx, selected_machine, selected_op] = 1
        # update the state of the selected machine
        td["busy_until"][batch_idx, selected_machine] = td["time"] + proc_time_of_action

        """
            Job Interrupt Check
            If there is a machine breakdown in time interval [td["start_times"] - td["finish_times"]]
                Then td["finish_times"] = td["finish_times"] + machine repair time
        """
        machine_breakdowns = td["machine_breakdowns"]  # [batch_size, num_machines, num_breakdowns * 2]
        start_times = td["start_times"]  # [batch_size, num_operations]
        finish_times = td["finish_times"]  # [batch_size, num_operations]
        busy_until = td["busy_until"]  # [batch_size, num_machines]

        batch_indices = torch.arange(machine_breakdowns.size(0), device=machine_breakdowns.device)

        # Gather breakdowns for the selected machines
        selected_machine_breakdowns = machine_breakdowns[
            batch_indices, selected_machine]  # [batch_size, num_breakdowns * 2]
        breakdown_times = selected_machine_breakdowns[:, ::2]  # [batch_size, num_breakdowns]
        breakdown_durations = selected_machine_breakdowns[:, 1::2]  # [batch_size, num_breakdowns]

        # Gather operation start and finish times for the selected operations
        starting_times = start_times[batch_indices, selected_op]  # [batch_size]
        finishing_times = finish_times[batch_indices, selected_op]  # [batch_size]

        # Create a copy of finishing_times to update iteratively
        updated_finishing_times = finishing_times.clone()


        num_breakdowns = breakdown_times.size(1)
        for breakdown_idx in range(num_breakdowns):
            # occurrence time
            occ_time = breakdown_times[:, breakdown_idx]
            # breakdown duration
            brk_dur = breakdown_durations[:, breakdown_idx]


            ## # machine is being repaired
            mask_ttr = torch.le(occ_time,starting_times)  & \
                       torch.lt(starting_times , (occ_time+brk_dur)) & \
                   torch.le(updated_finishing_times , 9999.0)

            # wait until machine is repaired
            updated_finishing_times = torch.where(
                mask_ttr,
              updated_finishing_times + (occ_time + brk_dur - starting_times) ,
                updated_finishing_times
            )


            # job interrupt due to machine breakdown during processing
            mask = (torch.lt(starting_times , occ_time)) & \
                   torch.lt(occ_time , updated_finishing_times) & \
                   torch.lt(updated_finishing_times , 9999.0)


            # Update finishing times where the mask is true
            updated_finishing_times = torch.where(
                mask,
                updated_finishing_times + brk_dur,
                updated_finishing_times
            )


        # Write back the updated finish times to the tensor
        finish_times[batch_indices, selected_op] = updated_finishing_times

        td["finish_times"][batch_indices, selected_op] = updated_finishing_times


        # Update busy_until for the selected machines
        busy_until[batch_indices, selected_machine] = updated_finishing_times

        # update adjacency matrices (remove edges)
        td["proc_times"] = td["proc_times"].scatter(
            2,
            selected_op[:, None, None].expand(-1, self.num_mas, 1),
            torch.zeros_like(td["proc_times"]),
        )
        td["ops_ma_adj"] = td["proc_times"].contiguous().gt(0).to(torch.float32)
        td["num_eligible"] = torch.sum(td["ops_ma_adj"], dim=1)
        # update the positions of an operation in the job (subtract 1 from each operation of the selected job)
        td["ops_sequence_order"] = (
            td["ops_sequence_order"] - gather_by_index(td["job_ops_adj"], selected_job, 1)
        ).clip(0)

        # some checks currently commented out
        # assert torch.allclose(
        #     td["proc_times"].sum(1).gt(0).sum(1),  # num ops with eligible machine
        #     (~(td["op_scheduled"] + td["pad_mask"])).sum(1),  # num unscheduled ops
        # )


        # Alternatively, these two lines can be used to render the environment at each step
        # #clear_output()
        # render(td,6)
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



    def _transit_to_next_time(self, step_complete, td: TensorDict) -> TensorDict:
        """
        Method to transit to the next time where either a machine becomes idle or a new job arrives
        We transition to the next time step where step_complete is true and we update the given tensordict
        with the new values
        Args:
            step_complete: boolean tensor indicating whether a batch is ready to transition
            td: TensorDict containing information about the environment

        Returns: updated tensordict and td["done"] indicating whether all operations are done

        """
        # we need a transition to a next time step if either
        # 1.) all machines are busy
        # 2.) all operations are already currently in process (can only happen if num_jobs < num_machines)
        # 3.) idle machines can not process any of the not yet scheduled operations
        # 4.) no_op is choosen
        # 5.) a job is arrived


        #AssertionError: infeasible action selected
        #td = self._check_machine_breakdowns(td)

        #available_time_ma = td["busy_until"]
        # calculate the earliest time when a machine becomes IDLE again
        # we want to transition to the next time step where a machine becomes idle again. This time step must be
        # in the future, therefore we mask all machine idle times lying in the past / present
        available_time = (
            torch.where(
                td["busy_until"] > td["time"][:, None], td["busy_until"], torch.inf
            ).min(1).values
        )
        # check if a new job arrived at/before td["time"]
        # if a job is not arrived set the value as infinity
        # if a job has arrived set the values as the job arrival time
        # take the earliest time
        next_job_arrival = td["job_arrival_times"].masked_fill(
            td["job_arrival_times"] <= td["time"].unsqueeze(1), torch.inf
        ).min(1).values

        end_op_per_job = td["end_op_per_job"]


        # determine the earliest next time step.
        # depending on the situation either transit to the earliest machine idle time
        # or earliest next job arrival time
        next_time = torch.min(available_time, next_job_arrival)
        #assert not torch.any(next_time[step_complete].isinf())
        assert not torch.any(torch.isinf(next_time) & step_complete)
        # advance to the next time step where the only steps have completed
        td["time"] = torch.where(step_complete, next_time, td["time"])

        # this may only be set when the operation is finished, not when it is scheduled
        # operation of job is finished, set next operation and flag job as being idle
        curr_ops_end = td["finish_times"].gather(1, td["next_op"])
        op_finished = td["job_in_process"] & (curr_ops_end <= td["time"][:, None])
        # check whether a job is finished, which is the case when the last operation of the job is finished
        job_finished = op_finished & (td["next_op"] == end_op_per_job)
        # determine the next operation for a job that is not done, but whose latest operation is finished
        td["next_op"] = torch.where(
            op_finished & ~job_finished,
            td["next_op"] + 1,
            td["next_op"],
        )
        td["job_in_process"][op_finished] = False

        td["job_done"] = td["job_done"] + job_finished
        #td["done"] = td["job_done"].all(1, keepdim=True)
        # alternative checking if done -> then the current time must be greater than equal
        # latest job arrival time in each batch & (td["time"] >= td["job_arrival_times"].max(1).values).unsqueeze(1)
        td["done"] = td["job_done"].all(1, keepdim=True) & (td["time"] >= td["job_arrival_times"].max(1).values).unsqueeze(1)
        return td, td["done"].squeeze(1)

    #######################################################################################################################




    @staticmethod
    def load_data(fpath, batch_size=[]):
        g = DJSSPFileGenerator(fpath)
        return g(batch_size=batch_size)



    def _get_job_machine_availability(self, td: TensorDict):
        '''
        False(0) -> action is feasible True(1)-> action is not feasible
        Args:
            td: TensorDict representing the current state of the environment.
        Returns:
            action_mask of size (batch_size, number of jobs, number of machines)
            If an entry [x][y][z] = False (Include)
                then in batch x the job y on machine z is available(feasible) and can be scheduled.
            If an entry [x][y][z] = True  (Exclude)
                then in batch x the job y on machine z is not available(feasible) and  therefore can't be dispatched
        1 indicates machine or job is unavailable at current time step


        '''
        batch_size = td.size(0)
        # TODO: CHECK_MACHINE_BREAKDOWNS_GET_JOB_MACHINE_AVAILABILITY

        ######################################################################################################
        # AssertionError: infeasible action selected
        #td = self._check_machine_breakdowns(td)
        ###################################################################################
        #machine_breakdowns = td["machine_breakdowns"]
        # checking machine breakdowns with a for loop
        # for batch_id in range(machine_breakdowns.size(0)):
        #   for machine_idx in range(machine_breakdowns.size(1)):
        #       # breakdowns of the machine in the current batch
        #       machine_idx_breakdowns = machine_breakdowns[batch_id, machine_idx]
        #       for breakdown_no in range(int(machine_breakdowns.size(2) / 2)):
        #           # 0-2-4-6-8...-> breakdown occurence time
        #           # 1-3-5-7-9...-> breakdown duration
        #           # if current time == machine breakdown time
        #           # if machine_idx_breakdowns[breakdown_no]["TIME"] == td["time"][batch_id]:
        #
        #           breakdown_occ_time = machine_idx_breakdowns[(breakdown_no) * 2]
        #           breakdown_end_time = machine_idx_breakdowns[(breakdown_no) * 2 + 1] + breakdown_occ_time
        #
        #           if breakdown_occ_time <= td["time"][batch_id] < breakdown_end_time:
        #                #  sikinti bence buradan kaynakli oluyor, eger bir tane job dispatchlenmissse ve calisiyorsa
        #                #  bir sonraki steplerde onun td["time"]'i her türlü daha kücük oluyor.
        #                # ama daha sonra bir step yapildiginda ve biz tekrar kontrol ettigimizde hali hazirda
        #                #  busy until olan yer assert oluyoR
        #               if td["busy_until"][batch_id,machine_idx] < td["time"][batch_id]:
        #                   td["busy_until"][batch_id,machine_idx] = breakdown_end_time

        ###################################################################################

        #(bs, jobs, machines)
        action_mask = torch.full((batch_size, self.num_jobs, self.num_mas), False).to(
            td.device
        )


        ###############MASK JOBS THAT ARE NOT ARRIVED YET######################################
        # mask jobs that are not arrived yet
        # (bs, jobs) True -> job has not arrived yet ; False -> job has arrived
        job_arrivals = td["time"].unsqueeze(1)<td["job_arrival_times"]
        # expand the job arrivals to shape of action_mask and add_ .
        # This ensures that, if job has not arrived yet,
        # it is not available across all machines
        action_mask.add_(job_arrivals.unsqueeze(2))

        # mask jobs that are done already
        action_mask.add_(td["job_done"].unsqueeze(2))
        # as well as jobs that are currently processed
        action_mask.add_(td["job_in_process"].unsqueeze(2))
        # mask machines that are currently busy
        action_mask.add_(td["busy_until"].gt(td["time"].unsqueeze(1)).unsqueeze(1))
        # exclude job-machine combinations, where the machine cannot process the next op of the job
        next_ops_proc_times = gather_by_index(
            td["proc_times"], td["next_op"].unsqueeze(1), dim=2, squeeze=False
        ).transpose(1, 2)
        action_mask.add_(next_ops_proc_times == 0)

        return action_mask


    @staticmethod
    def render(td, idx):
        return render(td, idx)

    def select_start_nodes(self, td: TensorDict, num_starts: int):
        return sample_n_random_actions(td, num_starts)

    def get_op_ma_proctime(self,td):
        """
        helper function to get operation-machine-processing time attributes for each operation
        which will be used in google OR-TOOLS
        NOTE: This method extracts only the information from the first batch (batch 0)
        Args:
            td: tensorDict (observation after reset)

        Returns:
            array fin of size number of operation
                where each index of the fin is a sub array with the values
                [operation ID , machineID , processing time of operation]
        """
        # (num_mas , num_op)
        data = torch.tensor(td["proc_times"][0])
        column_id , row_id , process_time = [] , [] , []

        # iterate through each column in data
        # where each column represents an operation
        for column_no in range(data.size(1)):
            # op_column = all rows of the current column
            op_column = data[:,column_no]

            # non-zero row in column = machine id
            machine_id = torch.nonzero(op_column, as_tuple=True)[0].item()
            proc_time_of_operation = op_column[machine_id].item()
            column_id.append(column_no)
            row_id.append(machine_id)
            process_time.append(proc_time_of_operation)

        fin = []
        for op , machine , proc_time in zip (column_id , row_id , process_time):
            fin.append([op,machine,proc_time])

        return  fin

    import collections
    from ortools.sat.python import cp_model

    def OR_TOOLS(self,td):
        # job arrival times
        arrival_times = td["job_arrival_times"][0]

        # [[operation_id, machine_id, proc_time]]
        fin = self.get_op_ma_proctime(td)

        # create the jobs_data array
        # [ job [machine_no , proc_time]]
        jobs_data = []
        num_jobs = td["start_op_per_job"].size(1)

        # add empty array for each job in jobs_data
        for i in range(num_jobs):
            jobs_data.append([])

        for x in range((num_jobs * num_jobs + 1) - 1):
            # her job esit sayida operation'a sahip
            job_no = x // num_jobs
            # task (machine_id , processing_time)
            task = (fin[x][1], fin[x][2])
            jobs_data[job_no].append(task)

        # horizon = torch.sum(td["proc_times"][0]).item()
        horizon = sum(op[2] for op in fin)

        # declare the model
        model = cp_model.CpModel()

        # define the variables
        # create a named tuple to store information about created varibles
        task_type = collections.namedtuple("task_type", "start end interval")

        # create a named tuple to manipulate solution information
        assigned_task_type = collections.namedtuple("assigned_task_type", "start job index duration")

        # create job intervals and add to the corresponding MACHINE LIST
        all_task = {}
        machine_to_intervals = collections.defaultdict(list)

        all_machines = range(self.num_mas)

        # OR-Tools does not support float type numbers therefore we reound all of them

        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                # get the machine id and the duration from the task
                machine, duration = task
                # AttributeError: 'float' object has no attribute 'get_integer_var_value_map'
                duration = int(duration) + 1
                suffix = f"_{job_id}_{task_id}"
                # Create an integer variable with domain [lb, ub]. [0,horizon] "start time of the specific task"
                # TypeError: __init__(): incompatible constructor arguments. The following argument types are supported:
                # 1. ortools.util.python.sorted_interval_list.Domain(arg0: int, arg1: int)
                # Invoked with: 0, 1929.2211356163025
                horizon = int(horizon) + 1
                # start time of the task is created here
                job_arrival_time = int(arrival_times[job_id].item()) + 1

                # start_var = model.new_int_var(0 , horizon , "start"+ suffix)
                start_var = model.new_int_var(job_arrival_time, horizon, "start" + suffix)

                # create ending time of the specific task using constraint programming model (end_0_0...)
                # final/end time of the specific task is created here
                # end_var = model.new_int_var(0, horizon , "end"+ suffix)
                end_var = model.new_int_var(job_arrival_time, horizon, "end" + suffix)
                #  create interval variable from start_var duration end_var (interval_0_2.....)
                interval_var = model.new_interval_var(
                    start_var, duration, end_var, "interval" + suffix
                )

                # TODO: machine breakdowns
                # create updated_end variable to use if there is a machine breakdown
                updated_end = model.new_int_var(0, horizon, "adjusted_end" + suffix)
                # extract the breakdowns of the machine
                breakdowns = td["machine_breakdowns"][0, machine]
                # extract the breakdown occurrence times
                occurrences = breakdowns[::2]  # to get the only even indices (occurrence times)
                # extract the breakdown durations
                durations = breakdowns[1::2]  # to get the only odd indices (durations)



                #######################################################################################################
                # # we need to add constraint on breakdown
                for breakdwn_no in range(occurrences.size(0) - 1):
                    # occurence time
                    occ_time = int(occurrences[breakdwn_no].item())
                    # duration
                    breakdown_duration = int(durations[breakdwn_no].item()) + 1
                    # i have padded the tensor with values 0 if there is no breakdown, therefore ignore these instances
                    if occ_time == 0 and duration == 0:
                        continue
                    # check if breakdown happens when an operation is being processed on machine
                    # create condition variable
                    breakdown_condition = model.new_bool_var(f"breakdown_{suffix}_{breakdwn_no}")
                    # same logic as in makestep
                    model.add(start_var>=occ_time).only_enforce_if(breakdown_condition)
                    # operation starts before breakdown ends
                    #model.add(start_var < (occ_time + breakdown_duration)).only_enforce_if(breakdown_condition)
                    # operation ends after breakdowns starts
                    #model.add((start_var + duration) > occ_time).only_enforce_if(breakdown_condition)

                    # if there is breakdown add duration to the updated end
                    model.add(updated_end == end_var + breakdown_duration).only_enforce_if(breakdown_condition)


                # if there is no breakdown during operaion then updated_end  = end_var
                no_breakdown = model.new_bool_var(f"no_breakdown_{suffix}")
                model.add(updated_end == end_var).only_enforce_if(no_breakdown)
                model.add_bool_or([no_breakdown] + [model.new_bool_var(f"breakdown_{suffix}_{idx}") for idx in
                                                    range(occurrences.size(0))])

                #######################################################################################################

                # add all the task's with start,interval,end informations in all_task dict
                all_task[job_id, task_id] = task_type(
                    start=start_var,
                    end=updated_end,     #            end = updated_end,
                    interval=interval_var
                )


                # add at each machine index the operations/tasks interval where it containes start, end, duration
                machine_to_intervals[machine].append(interval_var)

        # DEFINE THE CONSTRAINTS

        # create and add disjunctive constraints
        for machine in all_machines:
            # use add_no_overlap method to create no overlap constrains
            # to prevent tasks for the same machine from overlapping time
            model.add_no_overlap(machine_to_intervals[machine])

        # precedences inside a job
        for job_id, job in enumerate(jobs_data):
            for task_id in range(len(job) - 1):
                model.add(
                    all_task[job_id, task_id + 1].start >= all_task[job_id, task_id].end
                )

        # Makespan objective
        # create a new integer variable for the makespan (obj_var is the makespan)
        obj_var = model.new_int_var(0, horizon, "makespan")  # makespan(0..21)

        # add constraint to make sthe makespan to the last task of all jobs
        # obj_var(makespan) is equal to latest end time of all task
        # obj_var == max (ebd times of all tasks)
        model.add_max_equality(
            obj_var, [all_task[job_id, len(job) - 1].end for job_id, job in enumerate(jobs_data)],
        )

        # set objective to minimize the makespan
        model.minimize(obj_var)

        solver = cp_model.CpSolver()
        status = solver.solve(model)
        # Check if solution was found
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print("Solution:")
            # Create one list of assigned tasks per machine
            # keys = mcahine IDs (0,1,2.---)
            # values = list of tasks assigned to each machine
            assigned_jobs = collections.defaultdict(list)
            # iterate through all jobs
            for job_id, job in enumerate(jobs_data):
                # job_id = ID of the job 0,1,2..
                # job = list of tasks for that job [(0, 3), (1, 2), (2, 2)]

                # iterate over tasks for that job
                for task_id, task in enumerate(job):
                    # task_id = index of the task in job  0,1,2..
                    # task tuple (0, 3) : (machine_id, proc_time)

                    machine = task[0]
                    # add tasks details to the machines list
                    assigned_jobs[machine].append(
                        assigned_task_type(
                            start=solver.value(all_task[job_id, task_id].start),
                            job=job_id,
                            index=task_id,
                            duration=task[1],
                        )
                    )
            # Create per machine output lines.
            output = ""
            for machine in all_machines:
                # Sort by starting time.
                assigned_jobs[machine].sort()
                sol_line_tasks = "Machine " + str(machine) + ": "
                sol_line = "              "

                for assigned_task in assigned_jobs[machine]:
                    name = f"job_{assigned_task.job}_task_{assigned_task.index}       "
                    # add spaces to output to align columns.
                    sol_line_tasks += f"{name:15}"
                    # TODO: !!!!!!!!
                    start = assigned_task.start
                    print("this is assigned task" , assigned_task)
                    print("this is machine", machine)
                    print("this is type of the machine" , type(machine))
                    print("this is start" , start)
                    print("this is type of the start" , type(start))
                    duration = assigned_task.duration
                    print("this is duration ", duration)
                    print("this is type of the duration" , type(duration))
                    sol_tmp = f"[{start},{start + duration}]"
                    # add spaces to output to align columns.
                    sol_line += f"{sol_tmp:15}"

                sol_line += "\n"
                sol_line_tasks += "\n"
                output += sol_line_tasks
                output += sol_line

            # Finally print the solution found.
            print(f"Optimal Schedule Length: {solver.objective_value}")
            print(output)
        else:
            print("No solution found.")

        # TODO: add job arrival_times here

    def _make_spec(self, generator: DJSSPGenerator):
        self.observation_spec = Composite(
            time=Unbounded(
                shape=(1,),
                dtype=torch.int64,
            ),
            next_op=Unbounded(
                shape=(self.num_jobs,),
                dtype=torch.int64,
            ),
            proc_times=Unbounded(
                shape=(self.num_mas, self.n_ops_max),
                dtype=torch.float32,
            ),
            pad_mask=Unbounded(
                shape=(self.num_mas, self.n_ops_max),
                dtype=torch.bool,
            ),
            start_op_per_job=Unbounded(
                shape=(self.num_jobs,),
                dtype=torch.bool,
            ),
            end_op_per_job=Unbounded(
                shape=(self.num_jobs,),
                dtype=torch.bool,
            ),
            start_times=Unbounded(
                shape=(self.n_ops_max,),
                dtype=torch.int64,
            ),
            finish_times=Unbounded(
                shape=(self.n_ops_max,),
                dtype=torch.int64,
            ),
            job_ops_adj=Unbounded(
                shape=(self.num_jobs, self.n_ops_max),
                dtype=torch.int64,
            ),
            ops_job_map=Unbounded(
                shape=(self.n_ops_max),
                dtype=torch.int64,
            ),
            ops_sequence_order=Unbounded(
                shape=(self.n_ops_max),
                dtype=torch.int64,
            ),
            ma_assignment=Unbounded(
                shape=(self.num_mas, self.n_ops_max),
                dtype=torch.int64,
            ),
            busy_until=Unbounded(
                shape=(self.num_mas,),
                dtype=torch.int64,
            ),
            num_eligible=Unbounded(
                shape=(self.n_ops_max,),
                dtype=torch.int64,
            ),
            job_in_process=Unbounded(
                shape=(self.num_jobs,),
                dtype=torch.bool,
            ),
            job_done=Unbounded(
                shape=(self.num_jobs,),
                dtype=torch.bool,
            ),
            machine_breakdowns=Unbounded(
                shape=(),  # for now ()
            ),
            shape=(),
        )
        self.action_spec = Bounded(
            shape=(1,),
            dtype=torch.int64,
            low=-1,
            high=self.n_ops_max,
        )
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)


    def _simulate_machine_breakdowns_(self, td, lambda_mtbf, lambda_mttr):

        # The mean time between failure and mean time off line subject to exponential distribution
        # (from E.S) assumin that MTBF-MTTR obey the exponential distribution
        # NOTE:  we can use torch.Tensor.exponential_ in here too
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
                # harcoded maximal processing time !!!
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

    # When we want to model machine breakdowns using nd_array we can use this method
    def nd_array_check_machine_breakdowns(self, td: TensorDict):
        """
            Method to check for machine breakdowns in the environment.
            Args:
                td: The state of the environment at the current time step (td["time"]).
            Returns:
                The updated td with the modified td["busy_until"] entry.
                If a machine has a breakdown at the current time step (td["time"])
                -> the "busy_until" entry for that machine is adjusted to the time step when the machine is repaired.
        """
        # breakdown of all machines in all bathces
        machine_breakdowns = td["machine_breakdowns"]
        # for batch_id in range(batch_size):
        for batch_id in range(len(machine_breakdowns)):
            for machine_idx in range(self.num_mas):
                # breakdowns of the machine in the current batch
                machine_idx_breakdowns = machine_breakdowns[batch_id][machine_idx]
                for breakdown_no in range(len(machine_idx_breakdowns)):
                    # if current time == machine breakdown time
                    if machine_idx_breakdowns[breakdown_no]["TIME"] == td["time"][batch_id]:
                        # duration of the breakdown
                        duration = machine_idx_breakdowns[breakdown_no]["DURATION"]
                        # machine is busy(not available) until -> " current_time + duration of the breakdown"
                        td["busy_until"][batch_id][machine_idx] = td["time"][batch_id] + duration
        return td

    def nd_array_make_step(self, td: TensorDict) -> TensorDict:
        """
        Use this make_step method only when you model the machine breakdowns as np.ndarray
        """
        batch_idx = torch.arange(td.size(0))
        # 3*(#req_op)
        selected_job, selected_op, selected_machine = self._translate_action(td)
        # mark job as being processed
        td["job_in_process"][batch_idx, selected_job] = 1
        # mark op as schedules
        td["op_scheduled"][batch_idx, selected_op] = True
        # update machine state
        proc_time_of_action = td["proc_times"][batch_idx, selected_machine, selected_op]
        # we may not select a machine that is busy
        assert torch.all(td["busy_until"][batch_idx, selected_machine] <= td["time"])
        # update schedule
        td["start_times"][batch_idx, selected_op] = td["time"]
        td["finish_times"][batch_idx, selected_op] = td["time"] + proc_time_of_action
        td["ma_assignment"][batch_idx, selected_machine, selected_op] = 1
        # update the state of the selected machine
        td["busy_until"][batch_idx, selected_machine] = td["time"] + proc_time_of_action
        # machine breakdown during processing
        """
            Job Interrupt Check
            If there is a machine breakdown in time interval [td["start_times"] - td["finish_times"]]
                Then td["finish_times"] = td["finish_times"] + machine repair time
        """
        for batch_no in range(len(td["machine_breakdowns"])):
            selected_machine_of_the_batch = selected_machine[batch_no].item()
            selected_operation_of_the_batch = selected_op[batch_no].item()
            breakdowns_of_machine = td["machine_breakdowns"][batch_no][selected_machine_of_the_batch]
            # iterate over each breakdown of the machine
            for breakdown_no in range(len(breakdowns_of_machine)):
                # breakdown occurence time
                breakdown_time = breakdowns_of_machine[breakdown_no]["TIME"]
                breakdown_duration = breakdowns_of_machine[breakdown_no]["DURATION"]
                starting_time_of_operation = td["start_times"][batch_no, selected_operation_of_the_batch].item()
                finishing_time_of_operation = td["finish_times"][batch_no, selected_operation_of_the_batch].item()
                # if during operation processing a machine breakdown occurs -> wait until machine is repaired
                # and then process the operation
                # TODO: DO WE HAVE TO ADD <= 9999.0 in here
                if ((starting_time_of_operation < breakdown_time < finishing_time_of_operation) and (
                        finishing_time_of_operation < 9999.0000)):
                    # repairing time of the machine during execution is added
                    # print("before", td["finish_times"][batch_no,selected_operation_of_the_batch])
                    td["finish_times"][batch_no, selected_operation_of_the_batch] += breakdown_duration
                    # print("after", td["finish_times"][batch_no,selected_operation_of_the_batch])
                    # todo: check if this correctly calculates the finish time s yani eski finish time'i mi aliyor yenisini mi
                    td["busy_until"][batch_no, selected_machine_of_the_batch] = td["finish_times"][
                        batch_no, selected_operation_of_the_batch]
                    # print("THIS OPERATION", selected_operation_of_the_batch)
        # removed before job interrupt check
        # td["ma_assignment"][batch_idx, selected_machine, selected_op] = 1
        # # update the state of the selected machine
        # td["busy_until"][batch_idx, selected_machine] = td["time"] + proc_time_of_action
        # update adjacency matrices (remove edges)
        td["proc_times"] = td["proc_times"].scatter(
            2,
            selected_op[:, None, None].expand(-1, self.num_mas, 1),
            torch.zeros_like(td["proc_times"]),
        )
        td["ops_ma_adj"] = td["proc_times"].contiguous().gt(0).to(torch.float32)
        td["num_eligible"] = torch.sum(td["ops_ma_adj"], dim=1)
        # update the positions of an operation in the job (subtract 1 from each operation of the selected job)
        td["ops_sequence_order"] = (
                td["ops_sequence_order"] - gather_by_index(td["job_ops_adj"], selected_job, 1)
        ).clip(0)
        # some checks
        # assert torch.allclose(
        #     td["proc_times"].sum(1).gt(0).sum(1),  # num ops with eligible machine
        #     (~(td["op_scheduled"] + td["pad_mask"])).sum(1),  # num unscheduled ops
        # )
        # Alternatively, these two lines can be used to render the environment at each step
        # #clear_output()
        # render(td,6)
        return td



