---
id: 782wdwxi7q23h39asbcgmsy
title: '131724'
desc: ''
updated: 1733777874374
created: 1733771848235
---
```python
import matplotlib.pyplot as plt
import numpy as np

# plt.figure(figsize=(12, 6))

for i in range(16):
    print(f"continuous ({i}): {batch['gene'].fitness_continuous[i]}")
    print(f"original ({i}): {batch['gene'].fitness_original[i]}")
    plt.plot(np.array(batch['gene'].fitness[i]))

plt.xticks(np.arange(0, 32, 1))
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
```
