# Anderson Accelerated (AA) Iterative-Transformers
Skoltech NLA project on Iterative refinement implementation of attention mechanisms with AA

We found out that the similar idea was used in ALBERT and Alpha Fold. Thus we aim to rebuild the same system and test the applicability and capabilities of non-gradient transformers layers cycling.

Tasks:
- Implement non-gradient iterations over encoder layers of BERT.
- Test different learning strategies:
  * Last call updates
  * Random call updates
  * Evaluation's stage refinement at last layer
- Bootstrap Anderson Acceleration inside of cycling
