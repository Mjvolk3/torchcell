---
id: kc5gf01wjiky3thanyoge5l
title: SupCR
desc: ''
updated: 1738213327128
created: 1738213297004
---
## First Implementation - Trying match Math

```python
class WeightedSupCRLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        weights: Optional[torch.Tensor] = None,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.eps = eps

        if weights is None:
            weights = torch.ones(2)  # Default to 2 dimensions with equal weights
        self.register_buffer("weights", weights / weights.sum())

    def compute_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise cosine similarities between embeddings."""
        embeddings_norm = F.normalize(embeddings, dim=1)
        return torch.mm(embeddings_norm, embeddings_norm.t()) / self.temperature

    def compute_dimension_loss(
        self,
        perturbed_embeddings: torch.Tensor,
        intact_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SupCR loss for a single dimension, handling NaN values."""
        device = perturbed_embeddings.device
        batch_size = labels.shape[0]

        # Mask for valid (non-NaN) labels
        valid_mask = ~torch.isnan(labels)
        if valid_mask.sum() < 2:  # Need at least 2 valid samples to compute loss
            return torch.tensor(0.0, device=device)

        # Filter out invalid samples
        valid_labels = labels[valid_mask]
        valid_perturbed = perturbed_embeddings[valid_mask]
        valid_intact = intact_embeddings[valid_mask]

        # Compute similarity matrices for valid samples
        perturbed_sims = self.compute_similarity(valid_perturbed)
        intact_sims = self.compute_similarity(valid_intact)

        # Compute pairwise label distances
        label_dists = torch.abs(valid_labels.unsqueeze(0) - valid_labels.unsqueeze(1))

        # Create mask for valid pairs (excluding self-pairs)
        valid_pairs = ~torch.eye(valid_labels.shape[0], dtype=torch.bool, device=device)

        # Compute loss terms
        loss = torch.tensor(0.0, device=device)
        for i in range(valid_labels.shape[0]):
            for j in range(valid_labels.shape[0]):
                if i == j:
                    continue

                # Count samples where d(y_i, y_k) >= d(y_i, y_j)
                d_ij = label_dists[i, j]
                indicator = (label_dists[i] >= d_ij).float()

                # Compute loss for perturbed embeddings
                numerator_p = torch.exp(perturbed_sims[i, j])
                denominator_p = (
                    (torch.exp(perturbed_sims[i]) * indicator).sum().clamp(min=self.eps)
                )
                loss -= torch.log(numerator_p / denominator_p)

                # Compute loss for intact embeddings
                numerator_i = torch.exp(intact_sims[i, j])
                denominator_i = (
                    (torch.exp(intact_sims[i]) * indicator).sum().clamp(min=self.eps)
                )
                loss -= torch.log(numerator_i / denominator_i)

        # Normalize by the number of valid pairs
        num_valid_pairs = valid_pairs.sum().clamp(min=1)
        return loss / num_valid_pairs

    def forward(
        self,
        perturbed_embeddings: torch.Tensor,
        intact_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass computing weighted loss across dimensions."""
        device = perturbed_embeddings.device
        num_dims = labels.shape[1]
        dim_losses = torch.zeros(num_dims, device=device)

        # Compute loss for each dimension
        for dim in range(num_dims):
            dim_losses[dim] = self.compute_dimension_loss(
                perturbed_embeddings, intact_embeddings, labels[:, dim]
            )

        # Apply dimension weights
        weights = self.weights
        weight_sum = weights.sum().clamp(min=self.eps)
        weighted_loss = (dim_losses * weights).sum() / weight_sum

        return weighted_loss, dim_losses
```

```python
michaelvolk@M1-MV torchcell % /Users/michaelvolk/opt/miniconda3/envs/torchcell/bin/python /Users/michaelvolk/Documents/projects/torchcell
/torchcell/losses/multi_dim_nan_tolerant.py

Testing WeightedSupCRLoss:

Input shapes:
Perturbed embeddings: torch.Size([4, 8])
Intact embeddings: torch.Size([4, 8])
Labels: torch.Size([4, 2])

Labels:
tensor([[1.2000, 2.1000],
        [0.4000,    nan],
        [   nan, 3.2000],
        [1.4000, 2.3000]])

Loss values:
Total loss: 2.3407
Dimension losses: tensor([2.5210, 2.1603])

Weighted loss values (weights=tensor([0.7000, 0.3000])):
Total weighted loss: 2.4128
Weighted dimension losses: tensor([2.5210, 2.1603])

Testing with NaN dimension:
Labels:
tensor([[1.2000,    nan],
        [0.4000,    nan],
        [1.8000,    nan],
        [1.4000,    nan]])
Total loss: 2.0708
Dimension losses: tensor([2.9582, 0.0000])
```

## Some Tensor Use and Matched Output

```python
class WeightedSupCRLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        weights: Optional[torch.Tensor] = None,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.eps = eps

        if weights is None:
            weights = torch.ones(2)
        self.register_buffer("weights", weights / weights.sum())

    def compute_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise cosine similarities between embeddings."""
        embeddings_norm = F.normalize(embeddings, dim=1)
        return torch.mm(embeddings_norm, embeddings_norm.t()) / self.temperature

    def compute_dimension_loss(
        self,
        perturbed_embeddings: torch.Tensor,
        intact_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SupCR loss for a single dimension using vectorized operations."""
        device = perturbed_embeddings.device

        # Handle valid samples
        valid_mask = ~torch.isnan(labels)
        if valid_mask.sum() < 2:
            return torch.tensor(0.0, device=device)

        valid_labels = labels[valid_mask]
        valid_perturbed = perturbed_embeddings[valid_mask]
        valid_intact = intact_embeddings[valid_mask]

        # Compute similarities and label distances
        perturbed_sims = self.compute_similarity(valid_perturbed)  # [n_valid, n_valid]
        intact_sims = self.compute_similarity(valid_intact)  # [n_valid, n_valid]
        label_dists = torch.abs(
            valid_labels.unsqueeze(0) - valid_labels.unsqueeze(1)
        )  # [n_valid, n_valid]

        # Create mask for valid pairs
        n_valid = valid_labels.size(0)
        mask = ~torch.eye(
            n_valid, dtype=torch.bool, device=device
        )  # [n_valid, n_valid]

        # Compute indicators for all pairs at once
        # Reshape label_dists for broadcasting: [n_valid, 1, n_valid]
        label_dists_i = label_dists.unsqueeze(1)
        # Reshape label_dists for broadcasting: [n_valid, n_valid, 1]
        label_dists_j = label_dists.unsqueeze(2)
        indicators = (
            label_dists_i >= label_dists_j
        ).float()  # [n_valid, n_valid, n_valid]

        # Compute exp similarities
        exp_perturbed = torch.exp(perturbed_sims)  # [n_valid, n_valid]
        exp_intact = torch.exp(intact_sims)  # [n_valid, n_valid]

        # Compute numerators (no need for unsqueeze)
        numerators_p = exp_perturbed[mask]  # [n_valid * (n_valid-1)]
        numerators_i = exp_intact[mask]  # [n_valid * (n_valid-1)]

        # Compute denominators with proper broadcasting
        denominators_p = torch.sum(
            exp_perturbed.unsqueeze(1) * indicators, dim=2
        )  # [n_valid, n_valid]
        denominators_i = torch.sum(
            exp_intact.unsqueeze(1) * indicators, dim=2
        )  # [n_valid, n_valid]

        # Apply mask to denominators
        denominators_p = denominators_p[mask]  # [n_valid * (n_valid-1)]
        denominators_i = denominators_i[mask]  # [n_valid * (n_valid-1)]

        # Compute loss
        loss = -torch.log(numerators_p / denominators_p.clamp(min=self.eps)).sum()
        loss += -torch.log(numerators_i / denominators_i.clamp(min=self.eps)).sum()

        return loss / mask.sum()

    def forward(
        self,
        perturbed_embeddings: torch.Tensor,
        intact_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_dims = labels.shape[1]

        # Compute losses for all dimensions at once
        dim_losses = torch.stack(
            [
                self.compute_dimension_loss(
                    perturbed_embeddings, intact_embeddings, labels[:, d]
                )
                for d in range(num_dims)
            ]
        )

        # Apply weights
        weights = self.weights
        weight_sum = weights.sum().clamp(min=self.eps)
        weighted_loss = (dim_losses * weights).sum() / weight_sum

        return weighted_loss, dim_losses

```

```python
michaelvolk@M1-MV torchcell % /Users/michaelvolk/opt/miniconda3/envs/torchcell/bin/python /Users/michaelvolk/Documents/projects/torchcell
/torchcell/losses/multi_dim_nan_tolerant.py

Testing WeightedSupCRLoss:

Input shapes:
Perturbed embeddings: torch.Size([4, 8])
Intact embeddings: torch.Size([4, 8])
Labels: torch.Size([4, 2])

Labels:
tensor([[1.2000, 2.1000],
        [0.4000,    nan],
        [   nan, 3.2000],
        [1.4000, 2.3000]])

Loss values:
Total loss: 2.3407
Dimension losses: tensor([2.5210, 2.1603])

Weighted loss values (weights=tensor([0.7000, 0.3000])):
Total weighted loss: 2.4128
Weighted dimension losses: tensor([2.5210, 2.1603])

Testing with NaN dimension:
Labels:
tensor([[1.2000,    nan],
        [0.4000,    nan],
        [1.8000,    nan],
        [1.4000,    nan]])
Total loss: 2.0708
Dimension losses: tensor([2.9582, 0.0000])
```

## Use Smarter Sorting

```python
class WeightedSupCRLoss(nn.Module):
    """
    An O(N^2 * log N) implementation of the SupCR loss that avoids building
    an (N x N x N) indicator tensor. Instead, it does a row-wise sort and
    suffix-sum trick to handle the "distance >= threshold" condition.

    For each dimension (excluding NaN labels), we do:

      1) Compute pairwise label distances and exponentiated similarities.
      2) For each anchor i in [0..N-1]:
         - Sort row i's label distances d_i,k in ascending order.
         - Compute a suffix sum over exp(sim_i,k) in that sorted order.
         - For each j != i, use binary search (torch.searchsorted) to find
           all k with d_i,k >= d_i,j. Summation is a single suffix-sum lookup.
         - Accumulate -log(numer/denom) for both
           the perturbed and the intact embeddings.

    Complexity is roughly O(N^2 log N) because each anchor row i
    does a sort (O(N log N)) plus an O(N) pass. For large N, this
    is typically faster and more memory-efficient than an O(N^3)
    broadcast approach.

    The final dimension loss is the total sum across i != j, scaled
    by 1/(N*(N-1)). We then sum dimension losses (with optional
    dimension weighting) and return a scalar plus the vector of
    per-dimension losses.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        weights: Optional[torch.Tensor] = None,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.eps = eps

        # By default, assume 2 label dimensions with equal weight
        if weights is None:
            weights = torch.ones(2)
        self.register_buffer("weights", weights / weights.sum())

    def _reversed_cumsum(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute cumulative sum in reverse order.
        For x = [x0, x1, ..., xN-1],
        _reversed_cumsum(x)[m] = sum_{k=m..N-1} x_k.
        """
        return torch.flip(torch.cumsum(torch.flip(x, dims=(0,)), dim=0), dims=(0,))

    def compute_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute the (N x N) matrix of un-exponentiated similarities,
        using cosine similarity scaled by 1/temperature.
        """
        normed = F.normalize(embeddings, p=2, dim=1)  # [N, d]
        # similarity[i,j] = dot(normed[i], normed[j]) / temperature
        return torch.mm(normed, normed.t()) / self.temperature

    def compute_dimension_loss(
        self,
        perturbed_embeddings: torch.Tensor,
        intact_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the SupCR loss for one label dimension, ignoring samples
        with NaN in that dimension. Uses a row-wise sort + suffix sum approach
        to handle the "distance >= threshold" condition in O(N log N) per anchor i.

        The returned scalar is the average over all valid i != j pairs.
        """
        device = labels.device

        # 1) Filter out NaNs
        valid_mask = ~torch.isnan(labels)
        if valid_mask.sum() < 2:
            return torch.tensor(0.0, device=device)

        valid_labels = labels[valid_mask]  # shape [M]
        valid_pert = perturbed_embeddings[valid_mask]  # shape [M, d]
        valid_int = intact_embeddings[valid_mask]  # shape [M, d]
        M = valid_labels.size(0)

        # 2) Build pairwise label distances & exponentiated sims
        #    shape [M, M]
        dists = torch.abs(valid_labels.unsqueeze(0) - valid_labels.unsqueeze(1))
        pert_sims = self.compute_similarity(valid_pert)  # un-exponentiated
        int_sims = self.compute_similarity(valid_int)

        exp_pert = torch.exp(pert_sims)  # shape [M, M]
        exp_int = torch.exp(int_sims)  # shape [M, M]

        # We'll skip diagonal i==j
        eye_mask = ~torch.eye(M, dtype=torch.bool, device=device)

        accum_loss = torch.tensor(0.0, device=device)
        accum_count = 0

        # 3) For each anchor i, sort row i's distances (d_i,j) in ascending order
        for i in range(M):
            row_dists = dists[i]  # shape [M], distances d(i,j) for j in [0..M-1]

            # Sort ascending
            sorted_dists_i, idx_i = row_dists.sort()  # shape [M]

            # The exponentiated similarities for row i, in that sorted order
            sorted_pert_i = exp_pert[i][idx_i]  # shape [M]
            sorted_int_i = exp_int[i][idx_i]  # shape [M]

            # Build suffix sums so suffix[m] = sum_{k=m..end} sorted_pert_i[k]
            suffix_pert_i = self._reversed_cumsum(sorted_pert_i)
            suffix_int_i = self._reversed_cumsum(sorted_int_i)

            # For each j, we want to quickly find how many k have d(i,k) >= d(i,j).
            # That is a single searchsorted call on sorted_dists_i.
            insertion_positions = torch.searchsorted(
                sorted_dists_i, row_dists, side="left"
            )  # shape [M]

            # Denominator for row i, each j => suffix_pert_i[index], suffix_int_i[index]
            denom_pert_i = suffix_pert_i[insertion_positions].clamp(min=self.eps)
            denom_int_i = suffix_int_i[insertion_positions].clamp(min=self.eps)

            # Numerator = exp_sims[i,j]
            numer_pert_i = exp_pert[i]  # shape [M]
            numer_int_i = exp_int[i]  # shape [M]

            row_loss_pert = -torch.log(numer_pert_i / denom_pert_i)
            row_loss_int = -torch.log(numer_int_i / denom_int_i)
            row_loss = row_loss_pert + row_loss_int

            # Exclude j==i
            valid_j_mask = eye_mask[i]
            accum_loss += row_loss[valid_j_mask].sum()
            accum_count += valid_j_mask.sum().item()

        # Final average over i!=j
        return accum_loss / max(accum_count, 1)

    def forward(
        self,
        perturbed_embeddings: torch.Tensor,
        intact_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Summation of dimension losses (NaN-tolerant). Then apply dimension weights.
        Returns (weighted_loss, dim_losses).
        """
        num_dims = labels.shape[1]
        device = labels.device
        dim_losses = torch.zeros(num_dims, device=device)

        for dim in range(num_dims):
            dim_losses[dim] = self.compute_dimension_loss(
                perturbed_embeddings, intact_embeddings, labels[:, dim]
            )

        # Weighted combination of dimension losses
        w = self.weights
        total_loss = (dim_losses * w).sum() / w.sum().clamp(min=self.eps)
        return total_loss, dim_losses
```

```python
michaelvolk@M1-MV torchcell % /Users/michaelvolk/opt/miniconda3/envs/torchcell/bin/python /Users/michaelvolk/Documents/projects/torchcell
/torchcell/losses/multi_dim_nan_tolerant.py

Testing WeightedSupCRLoss:

Input shapes:
Perturbed embeddings: torch.Size([4, 8])
Intact embeddings: torch.Size([4, 8])
Labels: torch.Size([4, 2])

Labels:
tensor([[1.2000, 2.1000],
        [0.4000,    nan],
        [   nan, 3.2000],
        [1.4000, 2.3000]])

Loss values:
Total loss: 2.3407
Dimension losses: tensor([2.5210, 2.1603])

Weighted loss values (weights=tensor([0.7000, 0.3000])):
Total weighted loss: 2.4128
Weighted dimension losses: tensor([2.5210, 2.1603])

Testing with NaN dimension:
Labels:
tensor([[1.2000,    nan],
        [0.4000,    nan],
        [1.8000,    nan],
        [1.4000,    nan]])
Total loss: 2.0708
Dimension losses: tensor([2.9582, 0.0000])
```

***

**Comparison of Three SupCR Implementations**

Below is a concise overview comparing the three SupCR loss implementations. They all compute the same mathematical quantity in principle, but differ in how they handle the triple summation (over \(i\), \(j\), and \(k\)), which heavily impacts **readability**, **memory usage**, and **runtime complexity**.

***

## 1. **Loopy Implementation (Matches Math Directly)**

**Snippet**  

```python
for i in range(valid_labels.shape[0]):
    for j in range(valid_labels.shape[0]):
        if i == j:
            continue
        d_ij = label_dists[i, j]
        indicator = (label_dists[i] >= d_ij).float()
        numerator_p = exp_perturbed[i, j]
        denominator_p = (exp_perturbed[i] * indicator).sum()
        loss -= torch.log(numerator_p / denominator_p)
        # etc.
```

**Key Points**  

1. **Clarity**: Matches the SupCR formula line by line. You see the loops over \(i\neq j\), the indicator for \(\lvert y_i-y_k\rvert\ge \lvert y_i-y_j\rvert\), and the ratio of exponentiated similarities.  
2. **Ease of Debugging**: Trivial to confirm that it does exactly what the math says.  
3. **Complexity**: For each dimension, you have an \(O(N^2)\) double loop, and within that loop the code does an \(O(N)\) operation for the summation over \(k\). This leads to an **\(O(N^3)\)** time complexity if \(N\) is large.  
4. **Memory**: This version does not explicitly build an \(N\times N\times N\) tensor; it does the summation over \(k\) in a vector operation each time inside the loop. That is still quite slow for large \(N\).  

**Summary**  

- *Most straightforward* but *slowest* for large \(N\).  
- Minimal “fancy” PyTorch broadcasting, so it is easy to verify correctness step by step.  

***

## 2. **Fully Vectorized \(O(N^3)\) Approach**

**Snippet**  

```python
# Build a 3D indicator = [n_valid, n_valid, n_valid]
indicators = (label_dists.unsqueeze(2) >= label_dists.unsqueeze(1)).float()

# denominator[i,j] = sum_{k} indicators[i,j,k] * exp_sims[i,k]
# Then you exclude diagonal (i==j) and average
```

**Key Points**  

1. **One‐Shot Tensor Ops**: Replaces the inner double‐loop with a single 3D broadcast (`indicator = label_dists[:,None,:] >= label_dists[:,:,None]`).  
2. **Same Complexity**: Still **\(O(N^3)\)** in the worst case. Both the memory for the 3D “indicator” (if actually stored) and the arithmetic can be large.  
3. **Fewer Python Loops**: Good for small to moderate \(N\). The GPU can handle the 3D ops in parallel, but memory might blow up for large \(N\).  

**Summary**  

- *Cleaner vector code* than the double loop, but *still \(O(N^3)\)* in memory and compute.  
- Yields exactly the same numerical results (modulo small floating‐point diffs) as the loopy version.  

***

## 3. **Row‐Wise Sort + Suffix‐Sum (\(O(N^2 \log N)\))**

**Snippet**  

```python
for i in range(M):
    # Sort dists[i] in ascending order
    sorted_dists_i, idx_i = row_dists.sort()

    # suffix_pert_i[m] = sum_{k=m..end} of exp_pert[i][idx_i[k]]
    suffix_pert_i = reversed_cumsum(sorted_pert_i)
    # then searchsorted(...) to find all k with d(i,k) >= d(i,j)
    denom_pert_i = suffix_pert_i[insertion_positions[j]]
```

**Key Points**  

1. **Lower Complexity**:  
   - Sorting each row \(i\) is \(O(N \log N)\).  
   - Doing that for all \(i\) is \(O(N^2 \log N)\).  
   - Then for each row, you handle all \(j\) in \(O(N)\) via `searchsorted`.  
   - Overall \(O(N^2 \log N)\) is *less* than \(O(N^3)\) for large \(N\).  
2. **Memory Efficiency**: No need for a \((N\times N\times N)\) boolean indicator.  
3. **Complexity**: The code is more involved: building suffix sums, sorting each row, using `searchsorted`, etc.  
4. **Exact Same Result**: Provided the logic is correct, it should match the other two approaches on small test examples.  

**Summary**  

- *Faster for large \(N\)* than the \(O(N^3)\) methods.  
- More complicated to implement and debug, but better scaling.  

***

## Numerical Differences or Matching

- All three methods **should** produce the *same final numeric result* on the same batch data (aside from minor floating‐point round‐off differences). In practice, small floating‐point discrepancies might arise because the summation order differs, but the results should be extremely close.  
- If you see big numeric mismatches, it usually means a mistake in broadcasting or reversed inequality in the “indicator” condition.  

***

## Conclusion

1. **Which is Most Faithful to the Paper?**  
   - The **loopy** version literally replicates the formula: iterate over \(i,j\) pairs, build an indicator for the relevant \(k\), compute \(-\log\) of the ratio, sum, then average. It is easiest to map line‐by‐line to the SupCR equation.  

2. **Which is Easiest to Maintain for Small Batches?**  
   - If your batch sizes are small, **either** the loopy or the fully broadcasted vector approach is simpler to read.  

3. **Which is Most Efficient for Large Batches?**  
   - The **row‐wise sorting** approach is best for large \(N\). It uses \(O(N^2 \log N)\) time instead of \(O(N^3)\) and also less memory.  

Overall, pick the method that meets your **performance** vs. **clarity** trade‐off. For small or moderate \(N\), the naive or broadcast approaches are straightforward and match the math exactly. For large \(N\), the row‐wise approach is significantly faster and more memory‐friendly.

***

Using fastest for now.
