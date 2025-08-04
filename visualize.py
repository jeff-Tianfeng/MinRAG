import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

plt.rcParams.update({
    'font.size': 16,
    'font.family': 'Times New Roman',
    'axes.edgecolor': 'white',
    'axes.grid': False,
    'axes.facecolor': '#f7f7f9',
    'figure.facecolor': '#f7f7f9'
})
sns.set(style="whitegrid")


df = pd.read_csv("type2_entropy_averaged.csv")
df = df[df["method"].isin(["greedy", "random"])]

pivot_df = df.pivot(index="sample_size", columns="method", values=["H_element", "H_element_norm"])

# Greedy/Random 
pivot_df["H_element_ratio"] = pivot_df["H_element"]["greedy"] / pivot_df["H_element"]["random"]
pivot_df["H_element_norm_ratio"] = pivot_df["H_element_norm"]["greedy"] / pivot_df["H_element_norm"]["random"]

# Smooth the curves using rolling mean
window = 30
pivot_df["H_element_ratio_smooth"] = pivot_df["H_element_ratio"].rolling(window=window, center=True).mean()
pivot_df["H_element_norm_ratio_smooth"] = pivot_df["H_element_norm_ratio"].rolling(window=window, center=True).mean()
pivot_df["H_element_greedy_smooth"] = pivot_df["H_element"]["greedy"].rolling(window=window, center=True).mean()
pivot_df["H_element_random_smooth"] = pivot_df["H_element"]["random"].rolling(window=window, center=True).mean()
pivot_df["H_element_norm_greedy_smooth"] = pivot_df["H_element_norm"]["greedy"].rolling(window=window, center=True).mean()
pivot_df["H_element_norm_random_smooth"] = pivot_df["H_element_norm"]["random"].rolling(window=window, center=True).mean()
pivot_df = pivot_df.iloc[::1].copy()

# ----------------------------
# 2. Shannon Entropy 
# ----------------------------
plt.figure(figsize=(8, 5))
sns.lineplot(x=pivot_df.index, y=pivot_df["H_element"]["greedy"], label="Greedy")
sns.lineplot(x=pivot_df.index, y=pivot_df["H_element"]["random"], label="Random")
plt.axvline(x=1250, color='r', linestyle='--', linewidth=1.5)
plt.fill_between(pivot_df.index, pivot_df["H_element"]["greedy"], color='tab:blue', alpha=0.2)
plt.fill_between(pivot_df.index, pivot_df["H_element"]["random"], color='tab:orange', alpha=0.2)

plt.ylim(bottom=6)
plt.title("Entropy")
plt.xlabel("Sample Size")
plt.ylabel("Entropy")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
plt.tight_layout()
plt.savefig("Figure_2F.svg", format='svg')
plt.savefig("Figure_2F.pdf", format='pdf')
plt.close()

# ----------------------------
# 3.Normalized Entropy
# ----------------------------
post_mask = pivot_df.index > 1500
x_post = pivot_df.index[post_mask].values.reshape(-1, 1)
y_post = pivot_df["H_element_norm"]["greedy"][post_mask].values

model = LinearRegression().fit(x_post, y_post)
y_pred = model.predict(x_post)
a, b = model.coef_[0], model.intercept_
print(f"Fitted line: y = {a:.6f} * x + {b:.6f}")

plt.figure(figsize=(8, 5))
sns.lineplot(x=pivot_df.index, y=pivot_df["H_element_norm"]["greedy"],
             label="MinRAG", color="tab:blue", linewidth=1.5)
plt.plot(x_post.flatten(), y_pred, label="MinRAG (Linear Fit)",
         linestyle="--", color="tab:red", linewidth=1, alpha=0.5)
sns.lineplot(x=pivot_df.index, y=pivot_df["H_element_norm"]["random"],
             label="Random Selection", color="tab:orange", linewidth=1.5)

plt.axvline(x=1500, color='r', linestyle='--', linewidth=1.2)
plt.fill_between(pivot_df.index, pivot_df["H_element_norm"]["random"], color='tab:orange', alpha=0.15)
plt.fill_between(pivot_df.index, pivot_df["H_element_norm"]["greedy"], color='tab:blue', alpha=0.15)

plt.ylim(bottom=1.05)
plt.xlabel("Sample Size")
plt.ylabel("Entropy / log(n)")
plt.title("Normalized Entropy")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
plt.tight_layout()
plt.savefig("entropy_H_element_norm_with_fit.svg", format='svg')
plt.savefig("entropy_H_element_norm_with_fit.pdf", format='pdf')
plt.close()

# ----------------------------
# 4. Entropy Ratio (Greedy / Random)
# ----------------------------
pivot_df_ratio = pivot_df[pivot_df.index <= 10000]
plt.figure(figsize=(8, 5))
y_vals = pivot_df_ratio["H_element_ratio_smooth"]
sns.lineplot(x=pivot_df_ratio.index, y=y_vals, color="tab:green", linewidth=2, label="Entropy Ratio")

plt.fill_between(pivot_df_ratio.index, y_vals, 1.0,
                 where=(y_vals > 1.0), interpolate=True, color='green', alpha=0.1)
plt.title("Entropy Ratio (Greedy / Random)")
plt.xlabel("Sample Size")
plt.ylabel("Entropy Ratio")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
plt.tight_layout()
plt.savefig("entropy_ratio_H_element.svg", format='svg')
plt.savefig("entropy_ratio_H_element.pdf", format='pdf')
plt.close()

# ----------------------------
# 5. Normalized Entropy Ratio
# ----------------------------
plt.figure(figsize=(8, 5))
sns.lineplot(x=pivot_df_ratio.index, y=pivot_df_ratio["H_element_norm_ratio_smooth"], color='tab:blue')
plt.title("Normalized Entropy Ratio (Greedy / Random)")
plt.xlabel("Sample Size")
plt.ylabel("Normalized Entropy Ratio")
plt.tight_layout()
plt.savefig("entropy_ratio_H_element_norm.svg", format='svg')
plt.close()
