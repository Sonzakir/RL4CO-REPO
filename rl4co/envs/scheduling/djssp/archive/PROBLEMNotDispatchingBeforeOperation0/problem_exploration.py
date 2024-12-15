#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time

from rl4co.envs.scheduling.djssp.env import DJSSPEnv
from rl4co.models import L2DPolicy, L2DModel
from rl4co.utils import RL4COTrainer
import gc
from rl4co.envs import JSSPEnv
from rl4co.models.zoo.l2d.model import L2DPPOModel
from rl4co.models.zoo.l2d.policy import L2DPolicy4PPO
from torch.utils.data import DataLoader
import json
import os

generator_params = {
"num_jobs" : 5 ,
"num_machines": 5 ,
"min_processing_time": 1 ,
"max_processing_time": 99 ,
"mtbf" : 17 ,
"mttr" : 4
}
env = DJSSPEnv(generator_params=generator_params,
_torchrl_mode=True,
stepwise_reward=True)


# In[2]:


import torch
if torch.cuda.is_available():
    accelerator = "gpu"
    batch_size = 4
    train_data_size = 2_000
    embed_dim = 128
    num_encoder_layers = 4
else:
    accelerator = "cpu"
    batch_size = 2
    train_data_size = 1_000
    embed_dim = 64
    num_encoder_layers = 2


# In[3]:


# Policy: neural network, in this case with encoder-decoder architecture
policy = L2DPolicy4PPO(
    embed_dim=embed_dim,
    num_encoder_layers=num_encoder_layers,
    env_name="jssp",
    het_emb=False
)


# In[4]:


def make_step(td, decoder):
    """
    Equivalent to FJSP make_step(), adapted for DJSSP where no encoder is used.
    td: TensorDict representing the current state of the environment.
    decoder: The L2DDecoder or policy that generates action logits.
    env: The DJSSP environment instance.
    """
    # Directly decode logits and mask from the raw input state `td`
    hidden, _ = decoder.feature_extractor(td)

    logits, mask = decoder(td, num_starts=0 , hidden = hidden)

    # Mask invalid actions by setting their logits to -inf
    action = logits.masked_fill(~mask, -torch.inf).argmax(1)

    # Update the state with the selected action
    td["action"] = action

    # Step the environment with the selected action
    td = env.step(td)["next"]

    return td


# # Reset the environment

# In[5]:


td = env.reset(batch_size = [3])


# ## Job Arrrival Times

# In[6]:


td["job_arrival_times"]
td["job_arrival_times"][0,1]=torch.Tensor([9])


# ## Processing Times of the operations

# In[7]:


td["proc_times"][0]


# ## Times in batch before make_step is executed

# In[8]:


td["time"]


# ## Start Times

# In[9]:


td["start_times"]


# ## Finish Times

# In[10]:


td["finish_times"]


# ## Make Manuel STEP in the Environment

# In[11]:


from matplotlib import pyplot as plt
from IPython.core.display_functions import clear_output, display

env.render(td, 0)
# Update plot within a for loop
while not td["done"].all():
    # Clear the previous output for the next iteration
    clear_output(wait=True)

    td = make_step(td=td ,decoder = policy.decoder)
    env.render(td, 0)
    # Display updated plot
    display(plt.gcf())

    # Pause for a moment to see the changes
    time.sleep(.4)


# ## Job Arrival Times

# In[12]:


td["job_arrival_times"]   # Unchanged


# ## Start Times
# 

# In[13]:


td["start_times"] #0-4 Job 0


# ## Finish Times

# In[14]:


td["finish_times"]

