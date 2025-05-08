## New Data

notes/torchcell.scratch.load_batch_005.md
torchcell/scratch/load_batch_005.py

This is where we use transforms. You can see how we used to use transforms when we weren't using `coo` data formatting
experiments/005-kuzmin2018-tmi/scripts/dango.py
torchcell/transforms/regression_to_classification_coo.py
tests/torchcell/transforms/test_regression_to_classification_coo.py

## Old Transforms With Named Phenotypes

torchcell/transforms/regression_to_classification.py
tests/torchcell/transforms/test_regression_to_classification.py

## Old Test Results

Results from test_regression_to_classification.py

```bash
(torchcell) michaelvolk@M1-MV torchcell % python -m pytest /Users/michaelvolk/Documents/projects/torchcell/tests/torchcell/transforms/test_regression_to_classification.py -v
===================================================================== test session starts =====================================================================
platform darwin -- Python 3.11.11, pytest-8.3.4, pluggy-1.5.0 -- /Users/michaelvolk/opt/miniconda3/envs/torchcell/bin/python
cachedir: .pytest_cache
rootdir: /Users/michaelvolk/Documents/projects/torchcell
configfile: pyproject.toml
plugins: anyio-4.8.0, hydra-core-1.3.2
collected 22 items                                                                                                                                            

tests/torchcell/transforms/test_regression_to_classification.py::TestLabelNormalizationTransform::test_minmax_normalization PASSED                      [  4%]
tests/torchcell/transforms/test_regression_to_classification.py::TestLabelNormalizationTransform::test_standard_normalization PASSED                    [  9%]
tests/torchcell/transforms/test_regression_to_classification.py::TestLabelNormalizationInverse::test_inverse_minmax PASSED                              [ 13%]
tests/torchcell/transforms/test_regression_to_classification.py::TestLabelNormalizationInverse::test_inverse_standard PASSED                            [ 18%]
tests/torchcell/transforms/test_regression_to_classification.py::TestLabelNormalizationInverse::test_inverse_with_nans PASSED                           [ 22%]
tests/torchcell/transforms/test_regression_to_classification.py::TestLabelNormalizationRoundTrip::test_round_trip_minmax PASSED                         [ 27%]
tests/torchcell/transforms/test_regression_to_classification.py::TestLabelNormalizationRoundTrip::test_round_trip_standard PASSED                       [ 31%]
tests/torchcell/transforms/test_regression_to_classification.py::TestLabelBinningTransform::test_soft_binning_forward PASSED                            [ 36%]
tests/torchcell/transforms/test_regression_to_classification.py::TestLabelBinningTransform::test_categorical_binning_forward PASSED                     [ 40%]
tests/torchcell/transforms/test_regression_to_classification.py::TestLabelBinningTransform::test_inverse_soft_binning PASSED                            [ 45%]
tests/torchcell/transforms/test_regression_to_classification.py::TestLabelBinningTransform::test_inverse_categorical_binning PASSED                     [ 50%]
tests/torchcell/transforms/test_regression_to_classification.py::TestInverseCompose::test_inverse_compose PASSED                                        [ 54%]
tests/torchcell/transforms/test_regression_to_classification.py::TestOrdinalBinning::test_ordinal_forward PASSED                                        [ 59%]
tests/torchcell/transforms/test_regression_to_classification.py::TestOrdinalBinning::test_ordinal_inverse PASSED                                        [ 63%]
tests/torchcell/transforms/test_regression_to_classification.py::TestOrdinalBinning::test_ordinal_round_trip PASSED                                     [ 68%]
tests/torchcell/transforms/test_regression_to_classification.py::TestOrdinalBinning::test_ordinal_with_nans PASSED                                      [ 72%]
tests/torchcell/transforms/test_regression_to_classification.py::TestOrdinalNormBinning::test_full_round_trip_with_stages PASSED                        [ 77%]
tests/torchcell/transforms/test_regression_to_classification.py::TestOrdinalNormBinning::test_ordinal_roundtrip_bin_containment PASSED                  [ 81%]
tests/torchcell/transforms/test_regression_to_classification.py::TestModelOutputSimulation::test_categorical_logits PASSED                              [ 86%]
tests/torchcell/transforms/test_regression_to_classification.py::TestModelOutputSimulation::test_soft_logits PASSED                                     [ 90%]
tests/torchcell/transforms/test_regression_to_classification.py::TestModelOutputSimulation::test_ordinal_logits PASSED                                  [ 95%]
tests/torchcell/transforms/test_regression_to_classification.py::TestInverseComposeWithGrads::test_inverse_compose_with_grad_tensors PASSED             [100%]

====================================================================== warnings summary =======================================================================
../../../opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:68
  /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:68: UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
    Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
    Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
    warnings.warn(f"An issue occurred while importing 'pyg-lib'. "

../../../opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:124
  /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/torch_geometric/typing.py:124: UserWarning: An issue occurred while importing 'torch-sparse'. Disabling its usage. Stacktrace: dlopen(/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so, 0x0006): Library not loaded: /Library/Frameworks/Python.framework/Versions/3.11/Python
    Referenced from: <B4DF21CE-3AD4-3ED1-8E22-0F66900D55D2> /Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/libpyg.so
    Reason: tried: '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/System/Volumes/Preboot/Cryptexes/OS/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file), '/Library/Frameworks/Python.framework/Versions/3.11/Python' (no such file)
    warnings.warn(f"An issue occurred while importing 'torch-sparse'. "

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=============================================================== 22 passed, 2 warnings in 2.55s ================================================================
```

Now wait for my command...
