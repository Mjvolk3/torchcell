# COO Transforms Implementation Status

## Background

The torchcell project is migrating from a label-based data format to a COO (Coordinate) format for phenotype data. In the old format, phenotype values were stored as `data["gene"]["fitness"]` or `data["gene"]["gene_interaction"]`. In the new COO format, all phenotype values are stored in a single tensor with accompanying index tensors:

- `phenotype_values`: The actual values
- `phenotype_type_indices`: Indices indicating which phenotype type
- `phenotype_sample_indices`: Sample indices  
- `phenotype_types`: List of phenotype names (e.g., ["gene_interaction"])

## Completed Tasks

### 1. Created `coo_regression_to_classification.py` ✓

- Implemented `COOLabelNormalizationTransform` that handles normalization in COO format
- Implemented `COOLabelBinningTransform` for binning continuous values (categorical, soft, ordinal)
- Implemented `COOInverseCompose` for applying inverse transforms in reverse order
- All transforms properly handle the COO data structure

### 2. Created comprehensive tests ✓

- File: `tests/torchcell/transforms/test_coo_regression_to_classification.py`
- Tests cover normalization, binning, inverse transforms, batch processing
- Tests handle mixed phenotypes, NaN values, and model output simulation

### 3. Updated experiment script ✓

- Modified `experiments/005-kuzmin2018-tmi/scripts/hetero_cell_bipartite_dango_gi.py`
- Changed imports from original transforms to COO versions
- Enabled transform application (previously commented out)

### 4. Updated trainer ✓

- Modified `torchcell/trainers/int_hetero_cell.py`
- Updated inverse transform section to create COO format data for predictions
- Properly handles device placement for tensors

## Current Issue: Failing Tests

### Problem Description

Three tests are failing related to the inverse transform in `COOLabelNormalizationTransform`:

1. `test_inverse_minmax_coo` - Returns normalized values unchanged instead of denormalizing
2. `test_inverse_with_nans_coo` - Same issue with NaN handling
3. `test_model_output_inverse_coo` - Inverse transform not applying denormalization

### Root Cause Analysis

Through debugging, we discovered:

1. The `denormalize()` method works correctly (converts [0.0, 0.25, 0.5, 1.0] → [0.0, 0.5, 1.0, 2.0])
2. The issue is with `copy.copy(data)` creating a shallow copy that doesn't properly isolate the HeteroData object
3. When we trace through the inverse method manually, it works correctly
4. But when called through the transform, the updates don't persist

Debug output shows:

```
Are they the same object? True  # phenotype_values tensor is the same object after copy
Are temp_data and denormalized the same? False  # HeteroData objects are different
```

This indicates that while `copy.copy()` creates a new HeteroData object, the nested tensors remain shared.

## Detailed Fix Plan

### Step 1: Fix the Data Copying Issue

The core issue is that `copy.copy(data)` doesn't create a deep enough copy for HeteroData objects. We need to:

1. **Option A: Use deepcopy selectively**

   ```python
   def inverse(self, data: Union[HeteroData, Batch]) -> Union[HeteroData, Batch]:
       # Create a new HeteroData object
       if isinstance(data, Batch):
           # Handle batch case
           new_data = Batch()
       else:
           new_data = HeteroData()
       
       # Copy all attributes properly
       for key in data.keys:
           new_data[key] = {}
           for attr, value in data[key].items():
               if isinstance(value, torch.Tensor):
                   new_data[key][attr] = value.clone()
               else:
                   new_data[key][attr] = copy.deepcopy(value)
   ```

2. **Option B: Follow the pattern from the original transforms**
   - Check how the original `LabelNormalizationTransform` handles this
   - It seems to work by directly modifying `data["gene"][label]`
   - We might need to ensure our updates propagate correctly

3. **Option C: Create a completely new data structure**
   - Build a fresh HeteroData object from scratch
   - Copy over all attributes manually
   - This ensures complete isolation

### Step 2: Test the Fix

1. Run the specific failing tests to verify the fix works
2. Ensure batch processing still works correctly
3. Test with the actual experiment to ensure end-to-end functionality

### Step 3: Optimize if Needed

1. If deepcopy is too slow, implement a custom copy method
2. Consider caching mechanisms for frequently used transforms

## Remaining Tasks

### 1. Fix the inverse transform issue (PRIORITY)

- Implement proper data copying in `COOLabelNormalizationTransform.inverse()` and `.forward()`
- Ensure the same fix is applied to `COOLabelBinningTransform` if it has similar issues
- All tests should pass after this fix

### 2. Verify binning transforms

- The binning transform implementation is more complex due to dimension changes
- Ensure the inverse binning (reconstruction from bins) works correctly
- Test with actual experimental data

### 3. Integration testing

- Run the full experiment pipeline with transforms enabled
- Verify that:
  - Learning happens in transformed space
  - Metrics are computed in original space (after inverse transform)
  - Visualizations use original scale values

### 4. Performance optimization

- Profile the transform operations
- Optimize if they become a bottleneck during training

### 5. Documentation

- Add docstring examples showing COO format usage
- Document the differences from the original transforms
- Add migration guide for users

## Code Locations

### Main implementation files:

- `torchcell/transforms/coo_regression_to_classification.py` - COO transforms
- `tests/torchcell/transforms/test_coo_regression_to_classification.py` - Tests

### Integration points:

- `experiments/005-kuzmin2018-tmi/scripts/hetero_cell_bipartite_dango_gi.py` - Experiment script
- `torchcell/trainers/int_hetero_cell.py` - Trainer with inverse transform usage

### Reference files:

- `torchcell/transforms/regression_to_classification.py` - Original transforms
- `torchcell/scratch/load_batch_005.py` - Data loading example

## Next Session Plan

1. **Start with the fix**: Implement proper data copying in the inverse/forward methods
2. **Run tests**: Verify all tests pass
3. **Integration test**: Run a small experiment to ensure transforms work end-to-end
4. **Clean up**: Remove debug files and finalize the implementation

The key insight is that we need to ensure proper isolation of data objects when doing transforms, especially with PyTorch Geometric's HeteroData structure which has nested attributes.
