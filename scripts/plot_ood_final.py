import numpy as np
import matplotlib.pyplot as plt

labels = ['IID SBM', 'OOD Size', 'OOD RR', 'OOD WS', 'OOD GEO']

# PI-GNN: 5 seeds mean ± std
pig_mean = np.array([60.46, 55.22, 58.08, 30.82, 42.42])
pig_std  = np.array([2.76, 4.56, 2.67, 12.13, 11.68])

# Baselines
random = np.array([50.6, 50.2, 49.9, 50.0, 50.1])
spectral = np.array([72.7, 72.6, 70.1, 63.1, 55.7])

x = np.arange(len(labels))
width = 0.25

plt.figure(figsize=(9, 5.5))

plt.bar(
    x - width,
    pig_mean,
    width,
    yerr=pig_std,
    capsize=4,
    label='PI-GNN'
)

plt.bar(
    x,
    random,
    width,
    label='Random'
)

plt.bar(
    x + width,
    spectral,
    width,
    label='Spectral'
)

plt.ylabel('Cut Ratio (%)')
plt.xlabel('Test Distribution')
plt.title('PI-GNN Generalization Across Graph Distributions')

plt.xticks(x, labels)
plt.ylim(0, 80)

plt.grid(axis='y', alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    'figures/ood_topology_baselines.pdf',
    bbox_inches='tight'
)

plt.savefig(
    'figures/ood_topology_baselines.png',
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print("Saved:")
print(" figures/ood_topology_baselines.pdf")
print(" figures/ood_topology_baselines.png")
