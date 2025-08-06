import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
accuracy_4o=np.load("")
accuracy_o3=np.load("")

fig, axes = plt.subplots (2, 1, figsize=(12,16))
sns.heatmap(accuracy_4o, annot=True, fmt=".1f", cmap="plasma", vmin=0, vmax=100, xticklables=range(1,21), yticklabels=range(1,21), ax=axes[0], cbar=True)


axes[0].set_title("Accuracy of GPT-4o-2025-08-05")
axes[0].set_xlabel("Digits Number 1")
axes[0].set_ylabel("Digits Number 2")



sns.heatmap(accuracy_o3, annot=True, fmt=".1f", cmap="plasma", vmin=0, vmax=100, xticklables=range(1,21), yticklabels=range(1,21), ax=axes[1], cbar=True)


axes[1].set_title("Accuracy of GPT-4o-2024-08-06")
axes[1].set_xlabel("Digits Number 1")
axes[1].set_ylabel("Digits Number 2")



plt.tight_layout()
plt.show()
