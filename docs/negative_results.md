# Negative and blocked results

## Official CPU reproduction attempt

The exact `pattern_2d` configuration did not complete its first 1,000-step block after approximately 18 minutes on the single exposed CPU device. It was interrupted before any metric was logged. This is an environment/runtime limitation, not a model-quality result.

## W&B under the managed sandbox

W&B 0.17.3 failed before model initialization because its local service could not create a loopback socket. Offline and disabled modes both exhibited this behavior. Running with permission for the local socket resolved it; remote logging remained disabled.

## Exact JAX pin on the RTX 5090

JAX/JAXlib 0.4.26 with its archived CUDA 12 build detected the RTX 5090 but could not compile even a trivial JIT. `ptxas` reported that its `sm_90a` target could not be compiled for the future architecture. Exact dependency reproduction is therefore CPU-only on this host.

## Flax 0.8.4 with JAX 0.6.0

After minimally updating JAX for Blackwell support, the repository-pinned Flax 0.8.4 failed during model initialization with `AttributeError: 'EvalTrace' object has no attribute 'level'`. Flax 0.10.2 fixes this compatibility failure, but the resulting environment is explicitly not an exact pinned reproduction.

## Dubious git ownership under root execution

When the agent process ran as root against a repository owned by `calla`, plain `git` commands failed with dubious ownership. That broke:

1. JSON finalization in `scripts/evaluate_pattern_checkpoint.py` after an otherwise complete seed-1 ablation;
2. `wandb.init` during the first seed-2 training attempt (`MailboxError: transport failed`).

Both were operational failures, not scientific result failures. Remediation used one-shot `safe.directory` overrides (`git -c ...` in the evaluator; `GIT_CONFIG_*` environment variables for training) without writing git config. The ablation and training were then re-run successfully.
