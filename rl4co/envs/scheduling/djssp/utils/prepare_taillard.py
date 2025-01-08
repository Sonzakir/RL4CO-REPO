import json
import os
import gc
import torch
from torch.utils.data import DataLoader

def prepare_taillard_data(nj, nm):
    # Target folder for Taillard instances
    fp = f"taillard/{nj}x{nm}"

    if not os.path.exists(fp):
        os.makedirs(fp)

    # Load the JSON file
    with open('JSPLIB/instances.json', 'r') as file:
        data = json.load(file)

    # Filter Taillard instances with matching jobs and machines
    instances = [x for x in data if "ta" in x["name"] and x["jobs"] == nj and x["machines"] == nm]
    print(f"Found {len(instances)} instances for {nj} jobs and {nm} machines")

    if not instances:
        raise FileNotFoundError(f"No matching Taillard instances found for {nj}x{nm}")

    # Copy files and validate
    for instance in instances:
        source_path = os.path.join("JSPLIB", instance['path'])
        target_path = os.path.join(fp, f"{instance['name']}.txt")

        # Check if the source file exists
        if os.path.exists(source_path):
            print(f"Copying {source_path} to {target_path}")
            os.system(f"cp {source_path} {target_path}")
        else:
            print(f"Warning: Source file {source_path} does not exist")

    # Verify if files were copied
    files_in_target = os.listdir(fp)
    assert len(files_in_target) > 0, f"No files copied to {fp}. Check source paths."
    print(f"Successfully prepared {len(files_in_target)} files in {fp}")



device = "cuda" if torch.cuda.is_available() else "cpu"

# path to taillard instances
FILE_PATH = "taillard/{nj}x{nm}"

results = {}
instance_types = [(15, 15), (20, 15), (20, 20), (30, 15), (30, 20)]

for instance_type in instance_types:
    print("------------")
    nj, nm = instance_type
    prepare_taillard_data(nj, nm)
    dataset = env.dataset(batch_size=[10], phase="test", filename=FILE_PATH.format(nj=nj, nm=nm))
    dl = DataLoader(dataset, batch_size=5, collate_fn=dataset.collate_fn)
    rewards = []

    for batch in dl:
        td = env.reset(batch).to(device)
        # use policy.generate to avoid grad calculations which can lead to oom
        out = model.policy.generate(td, env=env, phase="test", decode_type="multistart_sampling", num_starts=100, select_best=True)
        rewards.append(out["reward"])

    reward = torch.cat(rewards, dim=0).mean().item()
    results[instance_type] = reward

    print("Done evaluating instance type %s with reward %s" % (instance_type, reward))

    # avoid ooms due to cache not being cleared
    model.rb.empty()
    gc.collect()
    torch.cuda.empty_cache()