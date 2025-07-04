# InferenceDataset Class Plan

## Overview

A general-purpose dataset class for inference that follows the Neo4jCellDataset pattern but removes the complexity of database queries, conversion, deduplication, and aggregation. This dataset uses LMDB for efficient storage and retrieval of experiments, following the same format as Neo4jCellDataset. Put the new class in `/Users/michaelvolk/Documents/projects/torchcell/torchcell/data/`

## Class Architecture

### Core Components

```python
class InferenceDataset(Dataset):
    """
    A simplified dataset for inference that uses LMDB to store experiments
    and applies graph structures without database dependencies.
    
    Follows the same LMDB storage format as Neo4jCellDataset:
    - Each entry is a JSON list containing experiment and reference data
    - Integer keys for indexing (0, 1, 2, ...)
    - Supports all standard indices (gene, phenotype, dataset_name)
    """
    
    def __init__(
        self,
        root: str,
        gene_set: GeneSet,
        graphs: dict[str, GeneGraph] | None = None,
        node_embeddings: dict[str, BaseEmbeddingDataset] | None = None,
        graph_processor: GraphProcessor | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
    ):
        """
        Args:
            root: Root directory for processed data (LMDB will be created in root/processed/lmdb)
            gene_set: GeneSet for the organism
            graphs: Dict of graph names to GeneGraph objects
            node_embeddings: Dict of embedding names to embedding datasets
            graph_processor: Processor for converting experiments to graph data
            transform: Optional transform to apply to each data object
            pre_transform: Optional pre-transform (applied once during processing)
        """
```

### Key Methods

1. **`get(idx: int) -> CellData`**
   - Load experiment from LMDB at index
   - Deserialize JSON data
   - Apply graph processor to convert to CellData
   - Handle embeddings and graph structures
   - Apply transforms if specified

2. **`len() -> int`**
   - Return number of experiments in LMDB

3. **`process()`**
   - Load experiments from external sources
   - Store in LMDB with proper serialization
   - Build indices for quick lookup
   - Compute phenotype info

### Data Flow

```text
External Data Sources
    ├── Triple Combinations (pickle)
    ├── Experiment Lists
    └── CSV/TSV Files
           ↓
    load_experiments_to_lmdb()
           ↓
       LMDB Storage
           ↓
    InferenceDataset.get()
           ↓
    GraphProcessor.process()
           ↓
        CellData
           ↓
      Model Input
```

## Implementation Details

### 1. LMDB Storage Format

```python
# Each LMDB entry follows Neo4jCellDataset format:
{
    "0": [  # Key is string of integer index
        {
            "experiment": {
                "experiment_type": "FitnessExperiment",
                "dataset_name": "inference_dataset",
                "genotype": {...},
                "environment": {...},
                "phenotype": {...}
            },
            "experiment_reference": {
                "experiment_reference_type": "FitnessExperimentReference",
                ...
            }
        }
    ]
}
```

### 2. LMDB Initialization

```python
def _init_lmdb_read(self):
    """Initialize LMDB for reading (same as Neo4jCellDataset)."""
    self.env = lmdb.open(
        osp.join(self.processed_dir, "lmdb"),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
        max_spare_txns=16,
    )

def _init_lmdb_write(self):
    """Initialize LMDB for writing during processing."""
    self.env = lmdb.open(
        osp.join(self.processed_dir, "lmdb"),
        map_size=int(1e12),
        readonly=False,
    )
```

### 3. Loading Experiments to LMDB

```python
def load_experiments_to_lmdb(self, source: str | list[Experiment], source_type: str = "auto"):
    """
    Load experiments from various sources into LMDB.
    
    Args:
        source: Path to file or list of experiments
        source_type: "pickle", "csv", "experiments", or "auto"
    """
    self._init_lmdb_write()
    
    # Load experiments based on source type
    if source_type == "pickle" or (source_type == "auto" and isinstance(source, str) and source.endswith('.pkl')):
        experiments = self._load_from_pickle(source)
    elif source_type == "csv" or (source_type == "auto" and isinstance(source, str) and source.endswith('.csv')):
        experiments = self._load_from_csv(source)
    elif isinstance(source, list):
        experiments = source
    else:
        raise ValueError(f"Unknown source type: {source_type}")
    
    # Store in LMDB
    with self.env.begin(write=True) as txn:
        for idx, exp in enumerate(experiments):
            # Create data structure matching Neo4jCellDataset format
            data_list = [{
                "experiment": exp.model_dump(),
                "experiment_reference": self._create_default_reference(exp).model_dump()
            }]
            
            # Serialize and store
            key = str(idx).encode('utf-8')
            value = json.dumps(data_list).encode('utf-8')
            txn.put(key, value)
    
    self.close_lmdb()
```

### 4. Helper Functions for Triple Combinations

```python
@staticmethod
def create_experiment_from_triple(
    triple: tuple[str, str, str],
    dataset_name: str = "triple_inference",
    environment: Environment = None
) -> FitnessExperiment:
    """
    Create a FitnessExperiment from a triple of gene names.
    
    Args:
        triple: Tuple of three systematic gene names
        dataset_name: Name for the dataset
        environment: Environment conditions (defaults to standard)
    
    Returns:
        FitnessExperiment object ready for inference
    """
    if environment is None:
        environment = Environment(
            media=Media(name="YPD", state="solid"),
            temperature=Temperature(value=30.0)
        )
    
    perturbations = [
        SgaKanMxDeletionPerturbation(
            systematic_gene_name=gene,
            strain_id=f"{gene}_deletion"
        )
        for gene in triple
    ]
    
    genotype = Genotype(
        perturbations=perturbations,
        species="Saccharomyces cerevisiae",
        strain="S288C"
    )
    
    # Phenotype is None for inference
    phenotype = FitnessPhenotype(
        fitness=float('nan'),
        fitness_std=float('nan')
    )
    
    return FitnessExperiment(
        dataset_name=dataset_name,
        genotype=genotype,
        environment=environment,
        phenotype=phenotype
    )

@staticmethod
def load_triples_from_generate_script(pickle_path: str) -> list[Experiment]:
    """
    Load triple combinations from generate_triple_combinations.py output.
    
    Args:
        pickle_path: Path to pickle file from generate_triple_combinations.py
        
    Returns:
        List of FitnessExperiment objects
    """
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    experiments = []
    for triple in data['final_filtered_triples']:
        exp = InferenceDataset.create_experiment_from_triple(triple)
        experiments.append(exp)
    
    return experiments
```

### 5. Get Method Implementation

```python
def get(self, idx: int) -> CellData:
    """Get processed data for an experiment (matching Neo4jCellDataset)."""
    if self.env is None:
        self._init_lmdb_read()
    
    with self.env.begin() as txn:
        serialized_data = txn.get(f"{idx}".encode('utf-8'))
        if serialized_data is None:
            return None
        data_list = json.loads(serialized_data.decode('utf-8'))
        
        # Reconstruct experiment objects
        data = []
        for item in data_list:
            experiment_class = EXPERIMENT_TYPE_MAP[
                item["experiment"]["experiment_type"]
            ]
            experiment_reference_class = EXPERIMENT_REFERENCE_TYPE_MAP[
                item["experiment_reference"]["experiment_reference_type"]
            ]
            reconstructed_data = {
                "experiment": experiment_class(**item["experiment"]),
                "experiment_reference": experiment_reference_class(
                    **item["experiment_reference"]
                ),
            }
            data.append(reconstructed_data)
        
        # Process through graph processor
        processed_graph = self.process_graph.process(
            self.cell_graph, self.phenotype_info, data
        )
    
    return processed_graph
```

### 6. Index Building (Following Neo4jCellDataset)

```python
def compute_is_any_perturbed_gene_index(self) -> dict[str, list[int]]:
    """Build index of genes to experiment indices."""
    print("Computing is any perturbed gene index...")
    is_any_perturbed_gene_index = {}
    
    self._init_lmdb_read()
    
    try:
        with self.env.begin() as txn:
            cursor = txn.cursor()
            entries = [(key, value) for key, value in cursor]
        
        for key, value in entries:
            try:
                idx = int(key.decode())
                data_list = json.loads(value.decode())
                for data in data_list:
                    for pert in data["experiment"]["genotype"]["perturbations"]:
                        gene = pert["systematic_gene_name"]
                        if gene not in is_any_perturbed_gene_index:
                            is_any_perturbed_gene_index[gene] = set()
                        is_any_perturbed_gene_index[gene].add(idx)
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"Error processing entry {key}: {e}")
    
    finally:
        self.close_lmdb()
    
    # Convert sets to sorted lists
    return {
        gene: sorted(list(indices))
        for gene, indices in is_any_perturbed_gene_index.items()
    }
```

## Usage Examples

### 1. Loading Triple Combinations from generate_triple_combinations.py

```python
# Load filtered triples from the generation script
dataset = InferenceDataset(
    root="data/torchcell/inference/triples",
    gene_set=genome.gene_set,
    graphs={
        "physical": graph.G_physical,
        "regulatory": graph.G_regulatory,
    },
    node_embeddings={
        "fudt_3prime": fudt_3prime_dataset,
        "fudt_5prime": fudt_5prime_dataset,
    },
    graph_processor=SubgraphRepresentation(),
)

# Load experiments from pickle file
triple_pickle = "experiments/006-kuzmin-tmi/results/triple_combinations_2024.pkl"
experiments = InferenceDataset.load_triples_from_generate_script(triple_pickle)
dataset.load_experiments_to_lmdb(experiments)

print(f"Loaded {len(dataset)} triple experiments")
```

### 2. Creating Experiments from CSV

```python
# CSV format: gene1,gene2,gene3
dataset = InferenceDataset(
    root="data/torchcell/inference/custom",
    gene_set=genome.gene_set,
    graphs=graphs,
    node_embeddings=embeddings,
    graph_processor=SubgraphRepresentation(),
)

# Load from CSV file
dataset.load_experiments_to_lmdb("path/to/triples.csv", source_type="csv")
```

### 3. Direct Experiment Creation

```python
# Create experiments programmatically
experiments = []
for triple in selected_triples:
    exp = InferenceDataset.create_experiment_from_triple(
        triple,
        dataset_name="my_inference_run"
    )
    experiments.append(exp)

dataset = InferenceDataset(
    root="data/torchcell/inference/direct",
    gene_set=genome.gene_set,
    graphs=graphs,
    node_embeddings=embeddings,
    graph_processor=SubgraphRepresentation(),
)
dataset.load_experiments_to_lmdb(experiments)
```

### 4. Using with DataLoader

```python
from torch_geometric.loader import DataLoader

# Create dataloader for batched inference
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
)

# Run inference
model.eval()
predictions = []
with torch.no_grad():
    for batch in tqdm(loader):
        pred = model(batch)
        predictions.extend(pred.cpu().numpy())

# Export results
dataset.export_predictions(predictions, "inference_results.csv")
```

## Key Differences from Neo4jCellDataset

1. **No Database Queries**: Experiments loaded from files or created directly
2. **No Deduplication**: Assumes experiments are already unique
3. **No Aggregation**: Each experiment processed independently
4. **Simplified Processing**: No conversion step needed
5. **Default References**: Auto-generates experiment references if not provided

## Additional Features

### 1. Batch Loading

```python
def load_experiments_batch(self, sources: list[str], source_types: list[str] = None):
    """Load experiments from multiple sources."""
    all_experiments = []
    for i, source in enumerate(sources):
        source_type = source_types[i] if source_types else "auto"
        if source_type == "pickle":
            experiments = self.load_triples_from_generate_script(source)
        else:
            experiments = self._load_from_source(source, source_type)
        all_experiments.extend(experiments)
    
    self.load_experiments_to_lmdb(all_experiments)
```

### 2. Export Utilities

```python
def export_predictions(self, predictions: np.ndarray, output_file: str):
    """Export predictions with experiment metadata."""
    results = []
    
    self._init_lmdb_read()
    for idx, pred in enumerate(predictions):
        with self.env.begin() as txn:
            data = json.loads(txn.get(str(idx).encode()).decode())
            exp = data[0]["experiment"]
            
            results.append({
                'index': idx,
                'genes': [p["systematic_gene_name"] for p in exp["genotype"]["perturbations"]],
                'num_perturbations': len(exp["genotype"]["perturbations"]),
                'dataset_name': exp["dataset_name"],
                'prediction': float(pred),
            })
    
    pd.DataFrame(results).to_csv(output_file, index=False)
    self.close_lmdb()
```

### 3. Filtering and Subsetting

```python
def filter_by_genes(self, genes: list[str]) -> list[int]:
    """Get indices of experiments containing any of the specified genes."""
    if not hasattr(self, '_is_any_perturbed_gene_index'):
        self._is_any_perturbed_gene_index = self.compute_is_any_perturbed_gene_index()
    
    indices = set()
    for gene in genes:
        if gene in self._is_any_perturbed_gene_index:
            indices.update(self._is_any_perturbed_gene_index[gene])
    
    return sorted(list(indices))
```

## Performance Considerations

1. **LMDB Optimization**
   - Use read-only mode for inference
   - Disable file locking for parallel access
   - Configure appropriate map_size for dataset

2. **Memory Management**
   - Close LMDB environments when not in use
   - Use lazy loading for embeddings
   - Cache frequently accessed indices

3. **Parallel Processing**
   - Thread-safe LMDB read access
   - Support multi-worker DataLoader
   - Batch processing for predictions

## Implementation Checklist

1. [ ] Core InferenceDataset class with LMDB support
2. [ ] Helper functions for triple experiment creation
3. [ ] Loading utilities for various file formats
4. [ ] Index computation methods
5. [ ] Export and filtering utilities
6. [ ] Unit tests following Neo4jCellDataset patterns
7. [ ] Example notebooks for common use cases
8. [ ] Performance benchmarks