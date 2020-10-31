## Learning Rate Investigation
This project investigates the effect of learning rate updating schemes on the convergence time and model performance. 

Credit goes to the team: Yu Yang and Liwei Huang.

#### Detail of sequences we considered:
- seq1: 1/n
- seq2: 1/sqrt(n)
- seq3: 0.9^n
- seq4: cyclic

#### Detail of strategies we considered:
- by epoch: update parameters every epoch.
- by cutoff: update parameters if the loss is smaller than certain cutoff.
- by oscillate: update parameters every time the validation loss increases.

#### Optimizers we considered:
- SGD
- Adam

#### Comparison
- For every strategy, compare four sequences and benchmark.
- Pick the best combination of each sequence, and compare with SGD benchmark and Adam benchmark.
