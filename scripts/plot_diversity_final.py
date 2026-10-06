import numpy as np
import matplotlib.pyplot as plt

# 5-seed, 100k-step results
ntrain = np.array([1, 5, 10, 20, 50])

train_mean = np.array([67.70, 63.66, 63.72, 65.94, 64.32])
train_std  = np.array([2.21, 6.90, 5.88, 2.15, 4.24])

unseen_mean = np.array([54.00, 59.20, 61.22, 64.58, 64.20])
unseen_std  = np.array([6.92, 7.99, 5.31, 1.63, 3.98])

gap_mean = np.array([-13.70, -4.46, -2.50, -1.36, -0.12])
gap_std  = np.array([4.84, 1.87, 1.04, 1.11, 0.96])

# ==========================================================
# Figure 1: Train vs Unseen
# ==========================================================

plt.figure(figsize=(7, 5))

plt.errorbar(
    ntrain,
    train_mean,
    yerr=train_std,
    marker='o',
    capsize=4,
    linewidth=2,
    label='Train'
)

plt.errorbar(
    ntrain,
    unseen_mean,
    yerr=unseen_std,
    marker='s',
    capsize=4,
    linewidth=2,
    label='Unseen (IID)'
)

plt.xlabel('Number of Training Graphs')
plt.ylabel('Cut Ratio (%)')
plt.title('Effect of Training-Graph Diversity on IID Generalization')
plt.xticks(ntrain)
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()

plt.savefig(
    'figures/diversity_train_unseen.pdf',
    bbox_inches='tight'
)
plt.savefig(
    'figures/diversity_train_unseen.png',
    dpi=300,
    bbox_inches='tight'
)

plt.close()

# ==========================================================
# Figure 2: Generalization gap
# gap = unseen - train
# ==========================================================

plt.figure(figsize=(7, 5))

plt.errorbar(
    ntrain,
    gap_mean,
    yerr=gap_std,
    marker='o',
    capsize=4,
    linewidth=2
)

plt.axhline(
    0,
    linestyle='--',
    linewidth=1
)

plt.xlabel('Number of Training Graphs')
plt.ylabel('Generalization Gap (percentage points)')
plt.title('Train–Unseen Generalization Gap')
plt.xticks(ntrain)
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    'figures/diversity_gap.pdf',
    bbox_inches='tight'
)
plt.savefig(
    'figures/diversity_gap.png',
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print("Saved:")
print(" figures/diversity_train_unseen.pdf")
print(" figures/diversity_train_unseen.png")
print(" figures/diversity_gap.pdf")
print(" figures/diversity_gap.png")
