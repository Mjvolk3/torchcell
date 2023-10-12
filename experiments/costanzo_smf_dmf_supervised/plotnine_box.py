import numpy as np
import pandas as pd
from plotnine import *

# Define bins
bins = [
    (0.0, 0.4),
    (0.4, 0.5),
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.9, 1.0),
    (1.0, 1.1),
    (1.1, 1.2),
]

# Use a set number of samples for consistency
num_samples = 50

predicted = []
measured = []

np.random.seed(42)

for start, end in bins:
    center = (start + end) / 2

    # Simulate "Predicted growth" as the center of the bin
    predicted_values = [center] * num_samples

    # Simulate "Measured growth" values to vary around the bin center, but keep within the bounds
    stdev = (end - start) / 2
    values = np.clip(np.random.normal(center, stdev / 3, num_samples), start, end)
    measured.extend(values)
    predicted.extend(predicted_values)

df = pd.DataFrame({"Predicted growth": predicted, "Measured growth": measured})

# Create the plot
plot = (
    ggplot(df, aes(x="Predicted growth", y="Measured growth"))
    + geom_boxplot(fill="#F6A9A3", color="#D86B2B", width=0.05)
    + geom_hline(yintercept=1, color="black")
    + geom_vline(xintercept=1, color="black")
    + scale_x_continuous(
        breaks=[(b[0] + b[1]) / 2 for b in bins],
        labels=[str(round((b[0] + b[1]) / 2, 1)) for b in bins],
        limits=(0, 1.3),
    )
    + theme_minimal()
    + theme(
        panel_background=element_rect(fill="white"),
        axis_title_x=element_text(vjust=-0.5),
        axis_title_y=element_text(vjust=0.5),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
    )
)

print(plot)
