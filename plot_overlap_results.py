#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

CSV = "runs/specbridge_align_chemberta_pub_v3g_msgym_mapper_spec/eval_overlap_filtered_ckpt_001200.csv"
OUT_PREFIX = "runs/specbridge_align_chemberta_pub_v3g_msgym_mapper_spec/overlap_plots"

df = pd.read_csv(CSV)
print(df)

# Sort by threshold (high → low so it's intuitive)
df = df.sort_values("threshold", ascending=False)

# --- 1) Retrieval metrics vs threshold ---
plt.figure(figsize=(6,4))
plt.plot(df["threshold"], df["R@1"],  marker="o", label="R@1")
plt.plot(df["threshold"], df["R@5"],  marker="o", label="R@5")
plt.plot(df["threshold"], df["R@20"], marker="o", label="R@20")
plt.plot(df["threshold"], df["MRR"],  marker="o", label="MRR")

plt.xlabel("Cosine similarity threshold (removed if ≥ threshold)")
plt.ylabel("Score")
plt.ylim(0.7, 1.0)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}_metrics.png", dpi=200)
plt.savefig(f"{OUT_PREFIX}_metrics.pdf")
plt.close()

# --- 2) Size of remaining test set vs threshold ---
plt.figure(figsize=(6,4))
plt.plot(df["threshold"], df["filtered_remaining"], marker="o", label="# remaining")
plt.plot(df["threshold"], df["overlapping"], marker="o", label="# removed (overlapping)")

plt.xlabel("Cosine similarity threshold")
plt.ylabel("# spectra")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_PREFIX}_sizes.png", dpi=200)
plt.savefig(f"{OUT_PREFIX}_sizes.pdf")
plt.close()

print("Saved plots to", OUT_PREFIX + "_*.png/.pdf")
