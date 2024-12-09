import os
from time import sleep

from rl4co.envs.scheduling.djssp.generator import DJSSPGenerator
from rl4co.envs.scheduling.fjsp import INIT_FINISH, NO_OP_ID
from rl4co.envs.scheduling.fjsp.env import FJSPEnv
from rl4co.envs.scheduling.fjsp.utils import calc_lower_bound
from rl4co.utils.ops import gather_by_index
from einops import einsum, reduce
import torch
from tensordict.tensordict import TensorDict
from torch._tensor import Tensor



class DJSSPEnv(FJSPEnv):
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
            # TODO DJSSPFILE GENERATOR
            if generator_params.get("file_path", None) is not None:
                generator = DJSSPGenerator(**generator_params)
            else:
                generator = DJSSPGenerator(**generator_params)

        super().__init__(generator, generator_params, mask_no_ops, **kwargs)


    # num_mas remains unchanged
    # num_jobs remains unchanged
    # n_ops_max remains unchanged
    # set_instance_params remains unchanged
    # decode_graph_structure

    #FJSP
    def _reset(self, td: TensorDict = None, batch_size=None) -> TensorDict:
        self.set_instance_params(td)

        td_reset = td.clone()

        td_reset, n_ops_max = self._decode_graph_structure(td_reset)

        # schedule
        start_op_per_job = td_reset["start_op_per_job"]
        start_times = torch.zeros((*batch_size, n_ops_max))
        finish_times = torch.full((*batch_size, n_ops_max), INIT_FINISH)
        ma_assignment = torch.zeros((*batch_size, self.num_mas, n_ops_max))

        # TODO: for djssp -> i can change the busy_until initialization in here
        # reset feature space
        busy_until = torch.zeros((*batch_size, self.num_mas))
        # (bs, ma, ops)
        ops_ma_adj = (td_reset["proc_times"] > 0).to(torch.float32)
        # (bs, ops)
        num_eligible = torch.sum(ops_ma_adj, dim=1)
        #TODO: stochastic processing times

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
                "time": torch.min(td["job_arrival_times"], dim=1).values,  # -> advance the starting time to the earliest time a job arrives
                "job_done": torch.full((*batch_size, self.num_jobs), False),
                "done": torch.full((*batch_size, 1), False),
            },
            # changes: "time": torch.zeros((*batch_size,)) -> torch.min(td["job_arrival_times"], dim=1).values
        )

        #TODO: MACHINE_BREAKDOWNS_IN_RESET()
        td_reset = self._check_machine_breakdowns(td_reset)

        td_reset.set("action_mask", self.get_action_mask(td_reset))
        # add additional features to tensordict
        td_reset["lbs"] = calc_lower_bound(td_reset)
        td_reset = self._get_features(td_reset)

        return td_reset

   # # td["time"] = torch.Tensor([2.182248115539551, 2.182248115539551, 2.182248115539551])
   #  batch_idx = 0
   #  for bs in range(batch_idx + 1):
   #      for machine_idx in range(env.num_mas):
   #          m_idx_breakdowns = td["machine_breakdowns"][bs - 1][machine_idx]
   #          for breakdown_no in range(len(m_idx_breakdowns)):
   #
   #              if m_idx_breakdowns[breakdown_no]["TIME"] == td["time"][0]:
   #                  print(2.182248115539551)
   #                  print(f"Machine {machine_idx} - with TIME {m_idx_breakdowns[breakdown_no]['TIME']} ")
   #                  print(td["busy_until"][bs][machine_idx])
   #                  td["busy_until"][bs][machine_idx] = td["time"][0] + m_idx_breakdowns[breakdown_no]["DURATION"]
   #                  print(td["busy_until"][bs][machine_idx])

    # TODO -> here we can take td as an input and we can get the batch_size from it
    # WARNING: Maybe here we have to clone the tensordict
    def _check_machine_breakdowns(self, td: TensorDict ):
        #td["time"] = torch.Tensor([3.000108480453491, 0, 0])
        batch_size = td.size(0)
        # breakdown of all machines in all bathces
        machine_breakdowns = td["machine_breakdowns"]
        for batch_id in range(batch_size):
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

        # print(".............THIS IS GET_ACTION MASK.......")
        # print(mask)
        # print(".............THIS IS GET_ACTION MASK.......")
        return mask

    # TODO: Dynamic Job Arrival
    def _get_job_machine_availability(self, td: TensorDict):
        batch_size = td.size(0)

        # TODO: CHECK_MACHINE_BREAKDOWNS_GET_JOB_MACHINE_AVAILABILITY
        td = self._check_machine_breakdowns(td)

        # (bs, jobs, machines)
        action_mask = torch.full((batch_size, self.num_jobs, self.num_mas), False).to(
            td.device
        )

        # mask jobs that are done already
        action_mask.add_(td["job_done"].unsqueeze(2))
        #################################################################################
        # job_arrival_times = td["job_arrival_times"]
        # current_time = td["time"]
        #
        # # Reshape current_time to match dimensions for broadcasting
        # current_time = current_time.unsqueeze(-1)  # Shape: [batch_size, 1]
        #
        # # Compare arrival times with current time to create a boolean tensor
        # job_arrived =   (job_arrival_times <= current_time)  # Shape: [batch_size, num_jobs]
        #
        # # Ensure the result is of boolean type
        # job_arrived = job_arrived.to(torch.bool)
        #
        # action_mask.add_(job_arrived.unsqueeze(2))

        # td["job_arrived"]

        #################################################################################

        # as well as jobs that are currently processed
        action_mask.add_(td["job_in_process"].unsqueeze(2))




        # mask machines that are currently busy
        action_mask.add_(td["busy_until"].gt(td["time"].unsqueeze(1)).unsqueeze(1))

        # exclude job-machine combinations, where the machine cannot process the next op of the job
        next_ops_proc_times = gather_by_index(
            td["proc_times"], td["next_op"].unsqueeze(1), dim=2, squeeze=False
        ).transpose(1, 2)
        action_mask.add_(next_ops_proc_times == 0)

        # print("-------------------_get_job_machine_availability()_BEFORE ACTION MASK -------------------")
        # print(action_mask)
        # print("-------------------_get_job_machine_availability()_BEFORE ACTION MASK-- END -------------------")

        # TODO exclude jobs that are not arrived yet
        # td["job_arrival_times"][batch_no][arr_time_job1, arr_time_job2, ........]
        # td["time"] = torch.Tensor([42, 42, 42])
        #td["batch_no"]'su sonradan ekledim
        for batch_no in range(batch_size):
            #the problem is about here when there is no job to start -> it gives an error
           for job_idx in range(0,self.num_jobs):
               boo = td["job_arrival_times"][batch_no][job_idx].le(td["time"])
               if(not boo[batch_no].item()):
                   action_mask[batch_no][job_idx].fill_(True) # TODO: check the logic again-here can be False or Int Bool  too
        #            print(f"IN BATCH {batch_no} JOB WITH ID {job_idx} not arrived yet")
        # print("-------------------_get_job_machine_availability()_AFTER ACTION MASK -------------------")
        # print(action_mask)
        # print("-------------------_get_job_machine_availability()_AFTER ACTION MASK-- END -------------------")

        return action_mask

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
        #####################################################################
        # if(torch.all(td["time"]==0) and torch.all(td["job_arrival_times"]>0)):
        #     #birinci fikrim
        #     #print(".action" , td["action_mask"])
        #     #td.set("action_mask", self.get_action_mask(td))
        #     #return td
        #     # 2inci fikrim
        #     print("&&THE CONDIITION IS TRUE")
        #     # td ,_ = self._transit_to_next_time(False, td )
        #     # return td
        #     # 3üncü fikrim
        #     # For each batch, set td["time"] to the minimum of the corresponding batch in td["job_arrival_times"]
        #     td["time"] = torch.min(td["job_arrival_times"], dim=1).values
        #     print("THIS IS JOB ARRIVAL TIMES" , td["job_arrival_times"])
        #     td.set("action_mask" , self.get_action_mask(td))
        #     return self._step(td)
        # burada ben sadece ilk seferinde eger td times sifiersa action mask'i düzelttim ama bu tam olarak da
        # dogru olmayabilir burada hata YAPMIS olabilirim
        #####################################################################

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

        return td



    #TODO for experiment purposes
#TODO: MACHINE BREAKDOWN ICIN STEP VE MAKE_STEP'e bir sey _check_machine_availability() cagrisi koymayi planladim
# ama tam olarak nereye koyabilecegimden tam olarak emin olmadim



#TODO: we not : makestep'i yazdigin zaman proc_times yerine actual_proctimes kullanarak
# stochastic processing time'i kullanmis olursi




