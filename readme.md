# Overcoming Catastrophic Forgetting with Hard Attention to the Task

## Abstract

Catastrophic forgetting occurs when a neural network loses the information learned with the first task, after training on a second task. This problem remains a hurdle for artificial intelligence systems with sequential learning capabilities. In this paper, we propose a task-based hard attention mechanism that preserves previous tasks' information without affecting the current task's learning. A hard attention mask is learned concurrently to every task through stochastic gradient descent, and previous masks are exploited to constrain such learning. We show that the proposed mechanism is effective for reducing catastrophic forgetting, cutting current rates by 45 to 80%. We also show that it is robust to different hyperparameter choices, and that it offers a number of monitoring capabilities. The approach features the possibility to control both the stability and compactness of the learned knowledge, which we believe makes it also attractive for online learning or network compression applications.

## Authors

Joan Serra, Didac Suris, Marius Miron, & Alexandros Karatzoglou.

## Reference and Link to Paper



Serrà, J., Surís, D., Miron, M. & Karatzoglou, A.. (2018). Overcoming Catastrophic Forgetting with Hard Attention to the Task. Proceedings of the 35th International Conference on Machine Learning, in PMLR 80:4548-4557

Link: [http://proceedings.mlr.press/v80/serra18a.html](http://proceedings.mlr.press/v80/serra18a.html)

## Installing

1. Create a python 3 conda environment (check the requirements.txt file)

2. The following folder structure is expected at runtime. From the git folder:
    * src/ : Where all the scripts lie (already produced by the repo)
    * dat/ : Place to put/download all data sets
    * res/ : Place to save results
    * tmp/ : Place to store temporary files

3. The main script is src/run.py. To run multiple experiments we use src/run_multi.py or src/work.py; to run the compression experiment we use src/run_compression.sh.

## Notes

* If using this code, parts of it, or developments from it, please cite the above reference. 
* We do not provide any support or assistance for the supplied code nor we offer any other compilation/variant of it. 
* We assume no responsibility regarding the provided code.

