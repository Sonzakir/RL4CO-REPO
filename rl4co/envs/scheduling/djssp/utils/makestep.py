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

######################################################################################################################
        # machine breakdown during processing
        """
            Job Interrupt Check
            If there is a machine breakdown in time interval [td["start_times"] - td["finish_times"]]
                Then td["finish_times"] = td["finish_times"] + machine repair time
        """
        #################################################################################
###########################################################################################################
        # # Assume the tensors are defined and populated as described earlier
        machine_breakdowns = td["machine_breakdowns"]  # [batch_size, num_machines, num_breakdowns * 2]
        start_times = td["start_times"]  # [batch_size, num_operations]
        finish_times = td["finish_times"]  # [batch_size, num_operations]
        busy_until = td["busy_until"]  # [batch_size, num_machines]
        #selected_machine = td["selected_machine"]  # [batch_size]
        #selected_op = td["selected_op"]  # [batch_size]

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

            ################################################# MASK BROKE MACHINES #####################################################
            # # machine is being reapaired
            # mask_ttr = torch.le(occ_time,starting_times)  & \
            #            torch.lt(starting_times , (occ_time+brk_dur)) & \
            #        torch.le(updated_finishing_times , 9999.0)
            #
            # # # # wait until machine is repair
            #
            # updated_finishing_times = torch.where(
            #     mask_ttr,
            #   updated_finishing_times + (occ_time + brk_dur - starting_times) ,#  occ_time + brk_dur + proc_time_of_action ,    #TODO: here logic can be wrong
            #     updated_finishing_times
            # )

            ################################################# MASK BROKE MACHINES #####################################################

            # job interrupt due to machine breakdown during processin
            # Mask for overlapping breakdowns
            mask = ((starting_times < occ_time) ) & \
                   (occ_time < updated_finishing_times) & \
                   (updated_finishing_times < 9999.0)


            # Update finishing times where the mask is true
            updated_finishing_times = torch.where(
                mask,
                updated_finishing_times + brk_dur,
                updated_finishing_times
            )


            # TODO: WHEN THE MACHINE IS NOT WORKING ?????

            # | ((breakdown_time  starting_times) & (starting_times<(breakdown_time + breakdown_duration)))
        # print(f"before batch_no {batch_indices} operation {selected_op}" , td["finish_times"][batch_indices, selected_op] )
        # Write back the updated finish times to the tensor
        finish_times[batch_indices, selected_op] = updated_finishing_times
        # print("485" ,td["finish_times"][batch_indices, selected_op])
        td["finish_times"][batch_indices, selected_op] = updated_finishing_times
        # print("488" ,td["finish_times"][batch_indices, selected_op])

        # print(f"after batch_no {batch_indices} operation {selected_op}" , td["finish_times"][batch_indices, selected_op] )

        # Update busy_until for the selected machines
        #TODO: check if busy_until updated correctly
        busy_until[batch_indices, selected_machine] = updated_finishing_times
################################################################################################################
        ################################################################################
        # for batch_no in range(td["machine_breakdowns"].size(0)):
        #     # machine on which action will be performed
        #     selected_machine_of_the_batch = selected_machine[batch_no].item()
        #     # action == operation
        #     selected_operation_of_the_batch = selected_op[batch_no].item()
        #     # breakdowns of the selected machine in the form [duration , duration , time, duration,....]
        #     breakdowns_of_machine = td["machine_breakdowns"][batch_no,selected_machine_of_the_batch]
        #
        #     # iterate over each breakdown time-duration pair
        #     for breakdown_no in range(int(breakdowns_of_machine.size(0)/2)):
        #         # breakdown occurrence time
        #         breakdown_time = breakdowns_of_machine[ breakdown_no * 2 ]
        #         # breakdown duration
        #         breakdown_duration = breakdowns_of_machine[ breakdown_no*2+1 ]
        #         starting_time_of_operation = td["start_times"][batch_no,selected_operation_of_the_batch].item()
        #         finishing_time_of_operation = td["finish_times"][batch_no,selected_operation_of_the_batch].item()
        #         # if during operation processing a machine breakdown occurs -> wait until machine is repaired
        #         # and then process the operation
        #         if((starting_time_of_operation < breakdown_time < finishing_time_of_operation)
        #                     and (finishing_time_of_operation<9999.0000)):
        #             # repairing time of the machine during execution is added
        #             td["finish_times"][batch_no,selected_operation_of_the_batch] += breakdown_duration
        #             td["busy_until"][batch_no,selected_machine_of_the_batch] = td["finish_times"][batch_no,selected_operation_of_the_batch]

#############################################################################################################################################
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
        # some checks currently commented out
        # assert torch.allclose(
        #     td["proc_times"].sum(1).gt(0).sum(1),  # num ops with eligible machine
        #     (~(td["op_scheduled"] + td["pad_mask"])).sum(1),  # num unscheduled ops
        # )


        # Alternatively, these two lines can be used to render the environment at each step
        # #clear_output()
        # render(td,6)

        return td