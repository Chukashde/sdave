import numpy as np
import matplotlib.pyplot as plt

sizes = np.array([500, 1000, 2000, 5000])

cut_mean = np.array([
    61.62,
    61.92,
    61.29,
    44.13
])

cut_std = np.array([
    1.96,
    1.86,
    1.49,
    14.90
])

plt.figure(figsize=(7, 5))

plt.errorbar(
    sizes,
    cut_mean,
    yerr=cut_std,
    marker='o',
    capsize=5,
    linewidth=2
)

# Training graph size
plt.axvline(
    1000,
    linestyle='--',
    linewidth=1,
    label='Training size (n=1000)'
)

plt.xlabel('Graph Size (Number of Nodes)')
plt.ylabel('Cut Ratio (%)')
plt.title('Generalization Across Graph Sizes')

plt.xticks(sizes)
plt.grid(alpha=0.25)
plt.legend()

plt.tight_layout()

plt.savefig(
    'figures/size_generalization.pdf',
    bbox_inches='tight'
)

plt.savefig(
    'figures/size_generalization.png',
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print("Saved:")
print(" figures/size_generalization.pdf")
print(" figures/size_generalization.png")
