

torchrun is the launcher, and its main job is to set up the environment for all the processes it's about to start.

Based on the arguments you give it (like --nproc_per_node and --nnodes), it automatically calculates and sets three critical environment variables before your Python script even begins to run:

    WORLD_SIZE: The total number of processes (e.g., 8)

RANK: The global, unique ID for this specific process (e.g., 5)

LOCAL_RANK: The local ID on this specific machine (e.g., 1)


```
torchrun --nproc_per_node=4 train_script.py
```

