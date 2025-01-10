import torch


def test_breakdowns(td_before , td_after):
    #iterate over each batch
    for batch_no in range(td_after["machine_breakdowns"].size(0)):
        # iterate over each operation
        for operation_no in range(td_after["start_times"][batch_no].size(0)):
            # get the machine no
            ma_assignment = td_after["ma_assignment"]
            machine_no = torch.argmax(ma_assignment[batch_no, :, operation_no]).item()
             # get the start time
            start_time_op = td_after["start_times"][batch_no, operation_no]
            # get the finish time
            finish_time_op = td_after["finish_times"][batch_no, operation_no]

            # get the processing time
            proc_time_op = td_before["proc_times"][batch_no,machine_no ,operation_no]

            # accumulator for total delayed time
            acc = 0

            # iterate over all breakdowns of the machine
            breakdowns_of_machine = td_after["machine_breakdowns"][batch_no,machine_no]

            for breakdown_no in range(int(breakdowns_of_machine.size(0)/2)):
                # breakdown occurrence time
                breakdown_time = breakdowns_of_machine[ breakdown_no * 2 ]
                # breakdown duration
                breakdown_duration = breakdowns_of_machine[ breakdown_no*2+1 ]

                # breakdown after dispatch
                if (breakdown_time > start_time_op) and (breakdown_time< (start_time_op + proc_time_op + acc)):
                    if (finish_time_op == start_time_op + proc_time_op) and (breakdown_duration!=0.0):
                        print("There is a breakdown but it is not considered")
                        print(
                            f" BATCH= {batch_no} OP = {operation_no} , MACHINE = {machine_no} , START TIME = {start_time_op} , FINISH TIME = {finish_time_op} , PROC ={proc_time_op} , ACC = {acc} ")

                    elif (finish_time_op - (start_time_op + proc_time_op + breakdown_duration) <0):
                        print("Something is wrong with the calculation")
                    acc += breakdown_duration.item()



                # operation planned on tha machine but machine is broken
                if (breakdown_time <= start_time_op):
                    if ((breakdown_time+breakdown_duration)>start_time_op):
                        acc += ((breakdown_duration+ breakdown_time) - start_time_op ).item()
                        if (finish_time_op == start_time_op + proc_time_op):
                            print("There is a breakdown but it not considered")




            if torch.ne(finish_time_op , (start_time_op + proc_time_op + acc)) :
                print(f"FAILED BATCH= {batch_no} OP = {operation_no} , MACHINE = {machine_no} , START TIME = {start_time_op} , FINISH TIME = {finish_time_op} , PROC ={proc_time_op} , ACC = {acc} ")
                print(f"FAILURE : EXPECTED = {finish_time_op}  BUT GOT {start_time_op +acc + proc_time_op}")
                print("##############################################################")
            print(f"SUCCESS : EXPECTED = {finish_time_op}  CALCULATED {start_time_op +acc + proc_time_op}")

