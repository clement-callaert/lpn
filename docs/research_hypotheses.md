# Research hypotheses

- H1: multi-start or stochastic latent search improves performance when encoder initialization is in a poor basin.
- H2: gains are larger on fixed out-of-distribution tasks than on in-distribution tasks.
- H3: a proximal penalty centered at the encoder initialization reduces observed-example overfitting and improves query generalization.
- H4: multiple candidates preserve program ambiguity that a single latent cannot express.
- H5: gains remain meaningful after decoder calls, gradient/objective evaluations, wall time, candidate count, and total compute are accounted for.

The official implementation is the control. No hypothesis will be tested until a trained baseline checkpoint and unambiguous fixed ID/OOD evaluation sets are available.
