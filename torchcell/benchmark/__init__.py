# torchcell/benchmark/__init__.py
"""The public benchmark: submission contract, grader, and the ``tc-bench`` service.

Submitters upload predictions, never scores. The modules, in the order a submission
passes through them:

- :mod:`torchcell.benchmark.submission`: the pydantic contract a submission must meet
  (one prediction row, the method metadata).
- :mod:`torchcell.benchmark.bundle`: one benchmark dataset on disk (public split file
  and template, private labels, all sha256-pinned).
- :mod:`torchcell.benchmark.validation`: parses an uploaded CSV against the public
  template and returns explicit rejection reasons.
- :mod:`torchcell.benchmark.grading`: the metrics (Pearson, Spearman, MSE, MAE, R2).
- :mod:`torchcell.benchmark.integrity`: flags on the validation and test pair.
- :mod:`torchcell.benchmark.ratelimit`: the per-account submission quota.
- :mod:`torchcell.benchmark.storage`: the zip archive kept for every scored submission.
- :mod:`torchcell.benchmark.oidc`, :mod:`torchcell.benchmark.security`,
  :mod:`torchcell.benchmark.db`: sign-in through CILogon, session tokens and the
  account policy, and the SQL schema.
- :mod:`torchcell.benchmark.app`: the FastAPI service that ties them together.

Nothing here imports torch, PyG, or a dataset loader, so the service image stays slim
and the validator can be run by a submitter with only pydantic installed.
"""
