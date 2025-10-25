

torchrun is the launcher, and its main job is to set up the environment for all the processes it's about to start.

Based on the arguments you give it (like --nproc_per_node and --nnodes), it automatically calculates and sets three critical environment variables before your Python script even begins to run:

    WORLD_SIZE: The total number of processes (e.g., 8)

RANK: The global, unique ID for this specific process (e.g., 5)

LOCAL_RANK: The local ID on this specific machine (e.g., 1)


```
torchrun --nproc_per_node=4 train_script.py
```




https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html

 we want the sum of all tensors in the group, we use dist.ReduceOp.SUM as the reduce operator. Generally speaking, any commutative mathematical operation can be used as an operator. Out-of-the-box, PyTorch comes with many such operators, all working at the element-wise level:

    dist.ReduceOp.SUM,

    dist.ReduceOp.PRODUCT,

    dist.ReduceOp.MAX,

    dist.ReduceOp.MIN,

    dist.ReduceOp.BAND,

    dist.ReduceOp.BOR,

    dist.ReduceOp.BXOR,

    dist.ReduceOp.PREMUL_SUM.

The full list of supported operators is here.

In addition to dist.all_reduce(tensor, op, group), there are many additional collectives currently implemented in PyTorch. Here are a few supported collectives.

    dist.broadcast(tensor, src, group): Copies tensor from src to all other processes.

    dist.reduce(tensor, dst, op, group): Applies op to every tensor and stores the result in dst.

    dist.all_reduce(tensor, op, group): Same as reduce, but the result is stored in all processes.

    dist.scatter(tensor, scatter_list, src, group): Copies the ithith tensor scatter_list[i] to the ithith process.

    dist.gather(tensor, gather_list, dst, group): Copies tensor from all processes in dst.

    dist.all_gather(tensor_list, tensor, group): Copies tensor from all processes to tensor_list, on all processes.

    dist.barrier(group): Blocks all processes in group until each one has entered this function.

    dist.all_to_all(output_tensor_list, input_tensor_list, group): Scatters list of input tensors to all processes in a group and return gathered list of tensors in output list.

The full list of supported collectives can be found by looking at the latest documentation for PyTorch Distributed
