# Anderson Accelerated (AA) Iterative-Transformers
Skoltech NLA project on Iterative refinement implementation of attention mechanisms with AA

We found out that the similar idea was used in ALBERT and Alpha Fold. Thus we aim to rebuild the same system and test the applicability and capabilities of non-gradient transformers layers cycling.

**Idea**:\
Our hypotethis is based on effect that is called 'chain of thought'. It shows that the transformers are very clever and we can give them more that one chance to produce more precise result. For each layer $L_i$ of the BERT's encoder we will reiterate it with hidden_state and feed to the same block over $N$ times. After one of these iterations we want to make gradient step and update weights. The moment for weights update will be chosen according to some policy. One can consider this process like *fixed-point iterations* and apply Anderson Acceleration (AA) to boost convergence. We want to study the effect of such iteration method to convergence of NN and its accuracy.
 
Tasks:
- Implement scheme of training and validation for NLP model on the GLUE task.
- Implement non-gradient iterations over encoder layers of BERT.
- Test different learning strategies:
  * Last call updates
  * Random call updates
  * Evaluation's stage refinement at last layer
- Bootstrap Anderson Acceleration inside of cycling
