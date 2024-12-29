import os

from IPython.core.display_functions import clear_output
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


        #ma_breakdowns = self.generator._simulate_machine_breakdowns_with_mtbf_mttr(batch_size,lambda_mtbf=20 , lambda_mttr=3)
        ma_breakdowns = self._simulate_machine_breakdowns_(td_reset , lambda_mtbf=20 , lambda_mttr=3) #bunu silidigin
        ################################################
        # TODO: check if this one overrides when we dont have any file instance
        td["job_arrival_times"] = torch.zeros((*batch_size, start_op_per_job.size(1)))


        td_reset = td_reset.update(
            {
                "start_times": start_times,
                "finish_times": finish_times,
                "ma_assignment": ma_assignment,
                "busy_until": busy_until,
                "num_eligible": num_eligible,
                "next_op": start_op_per_job.clone().to(torch.int64),
                "ops_ma_adj": ops_ma_adj,
                "machine_breakdowns": ma_breakdowns,
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

        #MACHINE_BREAKDOWNS_IN_RESET()
        # check machine breakdowns and update td["busy_until"] if necessary
        td_reset = self._check_machine_breakdowns(td_reset)

        td_reset.set("action_mask", self.get_action_mask(td_reset))
        # add additional features to tensordict
        td_reset["lbs"] = calc_lower_bound(td_reset)
        td_reset = self._get_features(td_reset)


        return td_reset



    # WARNING: Maybe here we have to clone the tensordict
    def _check_machine_breakdowns(self, td: TensorDict ):
        """
            Method to check for machine breakdowns in the environment.

            Args:
                td: The state of the environment at the current time step (td["time"]).

            Returns:
                The updated td with the modified td["busy_until"] entry.
                If a machine has a breakdown at the current time step (td["time"])
                -> the "busy_until" entry for that machine is adjusted to the time step when the machine is repaired.
        """

        #TODO:!!CLEAN-UP!!
        #batch_size = td.size(0)
        # print("env.py-155" , td.size(0))
        # breakdown of all machines in all bathces
        machine_breakdowns = td["machine_breakdowns"]
        another_batch_size = td["time"].shape[0]

        # for batch_id in range(batch_size):
        for batch_id in range(len(machine_breakdowns)):
            for machine_idx in range(self.num_mas):
                # breakdowns of the machine in the current batch
                # print("BATCH_ID", batch_id)
                # print("MACHINE_ID", machine_idx)
                machine_idx_breakdowns = machine_breakdowns[batch_id][machine_idx]
                for breakdown_no in range(len(machine_idx_breakdowns)):
                    # if current time == machine breakdown time
                    if machine_idx_breakdowns[breakdown_no]["TIME"] == td["time"][batch_id]:
                        # duration of the breakdown
                        duration = machine_idx_breakdowns[breakdown_no]["DURATION"]
                        # machine is busy(not available) until -> " current_time + duration of the breakdown"
                        td["busy_until"][batch_id][machine_idx] = td["time"][batch_id] + duration
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

    @staticmethod
    def list_files(path):
        files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
        assert len(files) > 0, "No files found in the specified path"
        return files


    def _step(self, td: TensorDict):
        # cloning required to avoid inplace operation which avoids gradient backtracking
        td = td.clone()

        td["action"].subtract_(1)

        # (bs)
        dones = td["done"].squeeze(1)

        # specify which batch instances require which operation
        no_op = td["action"].eq(NO_OP_ID)
        no_op = no_op & ~dones
        req_op = ~no_op & ~dones

        # transition to next time for no op instances
        if no_op.any():
            td, dones = self._transit_to_next_time(no_op, td)

        # select only instances that perform a scheduling action
        td_op = td.masked_select(req_op)

        td_op = self._make_step(td_op)
        # update the tensordict
        td[req_op] = td_op

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


        # # acaba sikinti buradan kaynakli mi oluyor diye kendime sormadan edemiyorum
        # if td["done"].all():
        #     render(td, 0)


        return td

    def _make_step(self, td: TensorDict) -> TensorDict:

        """
        Environment transition function
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
                starting_time_of_operation = td["start_times"][batch_no,selected_operation_of_the_batch].item()
                finishing_time_of_operation = td["finish_times"][batch_no,selected_operation_of_the_batch].item()
                # if during operation processing a machine breakdown occurs -> wait until machine is repaired
                # and then process the operation
                # TODO: DO WE HAVE TO ADD <= 9999.0 in here
                if((starting_time_of_operation < breakdown_time < finishing_time_of_operation) and (finishing_time_of_operation<9999.0000)):
                    # repairing time of the machine during execution is added
                    print("before", td["finish_times"][batch_no,selected_operation_of_the_batch])
                    td["finish_times"][batch_no,selected_operation_of_the_batch] += breakdown_duration
                    print("after", td["finish_times"][batch_no,selected_operation_of_the_batch])
                    # todo: check if this correctly calculates the finish time s yani eski finish time'i mi aliyor yenisini mi
                    td["busy_until"][batch_no,selected_machine_of_the_batch] = td["finish_times"][batch_no,selected_operation_of_the_batch]
                    print("THIS OPERATION", selected_operation_of_the_batch)

        # for batch_no in range(td.size(0)):
        #     print(len(td["machine_breakdowns"]))
        #     print(td.size(0))
        #     print(batch_idx)
        #     print(td["proc_times"].size())
        #     selected_machine_of_the_batch = selected_machine[batch_no].item()
        #     selected_operation_of_the_batch = selected_op[batch_no].item()
        #     breakdowns_of_machine = td["machine_breakdowns"][batch_no][selected_machine_of_the_batch]
        #     # iterate over each breakdown of the machine
        #     for breakdown_no in range(len(breakdowns_of_machine)):
        #         # breakdown occurence time
        #         breakdown_time = breakdowns_of_machine[breakdown_no]["TIME"]
        #         breakdown_duration = breakdowns_of_machine[breakdown_no]["DURATION"]
        #         starting_time_of_operation = td["start_times"][batch_no,selected_operation_of_the_batch].item()
        #         finishing_time_of_operation = td["finish_times"][batch_no,selected_operation_of_the_batch].item()
        #         # if during operation processing a machine breakdown occurs -> wait until machine is repaired
        #         # and then process the operation
        #         # TODO: DO WE HAVE TO ADD <= 9999.0 in here
        #         if((starting_time_of_operation < breakdown_time < finishing_time_of_operation) and (finishing_time_of_operation<9999.0000)):
        #             # repairing time of the machine during execution is added
        #             print("before", td["finish_times"][batch_no,selected_operation_of_the_batch])
        #             td["finish_times"][batch_no,selected_operation_of_the_batch] += breakdown_duration
        #             print("after", td["finish_times"][batch_no,selected_operation_of_the_batch])
        #             # todo: check if this correctly calculates the finish time s yani eski finish time'i mi aliyor yenisini mi
        #             td["busy_until"][batch_no,selected_machine_of_the_batch] = td["finish_times"][batch_no,selected_operation_of_the_batch]
        #             print("THIS OPERATION", selected_operation_of_the_batch)



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
        # print(td["time"])
        # print(len(td["machine_breakdowns"]))
        # print(td["proc_times"].shape)
        return td
#######################################################################################################################



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
                shape=(),  #for now ()
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


        '''
        batch_size = td.size(0)
        # TODO: CHECK_MACHINE_BREAKDOWNS_GET_JOB_MACHINE_AVAILABILITY
        td = self._check_machine_breakdowns(td)

        # (bs, jobs, machines)
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

        # Job arrival check via for-loops removed due to performance issues
        #     for batch_no in range(batch_size):
        #        for job_idx in range(0,self.num_jobs):
        #            boo = td["job_arrival_times"][batch_no][job_idx].le(td["time"])
        #            if(not boo[batch_no].item()):
        #                action_mask[batch_no][job_idx].fill_(True)

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


    @staticmethod
    def render(td, idx):
        return render(td, idx)

    def select_start_nodes(self, td: TensorDict, num_starts: int):
        return sample_n_random_actions(td, num_starts)




############################## UNUSED METHODDS #########################################################################





    # This is my main method old GET_JOB_MACHINE_AVAILABAILITJI
    #
    # def _get_job_machine_availability(self, td: TensorDict):
    #     batch_size = td.size(0)
    #
    #     td = self._check_machine_breakdowns(td)
    #
    #     # (bs, jobs, machines)
    #     action_mask = torch.full((batch_size, self.num_jobs, self.num_mas), False).to(
    #         td.device
    #     )
    #
    #     # mask jobs that are done already
    #     action_mask.add_(td["job_done"].unsqueeze(2))
    #
    #
    #     ####################alternative for job arrival_times checking##################
    #     # job_arrival_times = td["job_arrival_times"]
    #     # current_time = td["time"]
    #     # current_time = current_time.unsqueeze(-1)  # Shape: [batch_size, 1]
    #     # job_arrived =   (job_arrival_times <= current_time)  # Shape: [batch_size, num_jobs]
    #     # job_arrived = job_arrived.to(torch.bool)
    #     # action_mask.add_(job_arrived.unsqueeze(2))
    #     # td["job_arrived"]
    #     #################################################################################
    #
    #     # as well as jobs that are currently processed
    #     action_mask.add_(td["job_in_process"].unsqueeze(2))
    #
    #     # mask machines that are currently busy
    #     action_mask.add_(td["busy_until"].gt(td["time"].unsqueeze(1)).unsqueeze(1))
    #
    #     # exclude job-machine combinations, where the machine cannot process the next op of the job
    #     next_ops_proc_times = gather_by_index(
    #         td["proc_times"], td["next_op"].unsqueeze(1), dim=2, squeeze=False
    #     ).transpose(1, 2)
    #     action_mask.add_(next_ops_proc_times == 0)
    #
    #     """
    #             exclude jobs that are not arrived yet
    #             td["job_arrival_times"] has the form:
    #               ->td["job_arrival_times"][batch_no][arr_time_job1, arr_time_job2, ........]
    #     """
    #
    #     for batch_no in range(batch_size):
    #        for job_idx in range(0,self.num_jobs):
    #            boo = td["job_arrival_times"][batch_no][job_idx].le(td["time"])
    #            if(not boo[batch_no].item()):
    #                action_mask[batch_no][job_idx].fill_(True)
    #
    #     return action_mask