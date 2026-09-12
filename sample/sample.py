import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("sample.csv")
y = df["label"]
X = df.drop(columns=["label"])

rows = 4
cols = 6
fig, axes = plt.subplots(rows, cols, figsize=(8, rows * 2))

for i, ax in enumerate(axes.flat):
    image = X.iloc[i].values.reshape(28, 28)

    ax.imshow(image, cmap="gray")
    ax.set_title(f"Label: {y.iloc[i]}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("sample.png", bbox_inches="tight")
plt.close()
