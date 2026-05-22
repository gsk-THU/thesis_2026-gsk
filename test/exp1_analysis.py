import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

out_dir = Path("./figures")
out_dir.mkdir(parents=True, exist_ok=True)

# ===================== Data =====================
categories = ["boot", "fs", "interrupt", "memory", "process", "sync"]
cat_data = {
    "boot":       {"recall@1":1.000,"recall@3":1.000,"recall@5":1.000,"mrr":1.000,"ndcg@5":0.978,"avg_score":0.198},
    "fs":         {"recall@1":0.333,"recall@3":0.667,"recall@5":0.667,"mrr":0.500,"ndcg@5":0.515,"avg_score":0.210},
    "interrupt":  {"recall@1":0.667,"recall@3":0.667,"recall@5":1.000,"mrr":0.750,"ndcg@5":0.721,"avg_score":0.267},
    "memory":     {"recall@1":1.000,"recall@3":1.000,"recall@5":1.000,"mrr":1.000,"ndcg@5":0.890,"avg_score":0.297},
    "process":    {"recall@1":1.000,"recall@3":1.000,"recall@5":1.000,"mrr":1.000,"ndcg@5":0.867,"avg_score":0.363},
    "sync":       {"recall@1":0.600,"recall@3":0.600,"recall@5":0.800,"mrr":0.650,"ndcg@5":0.676,"avg_score":0.296},
}

queries = [
    {"id":"proc_01","cat":"process","recall@1":1.0,"recall@3":1.0,"recall@5":1.0,"mrr":1.0,"ndcg@5":0.679,"avg_score":0.351},
    {"id":"proc_02","cat":"process","recall@1":1.0,"recall@3":1.0,"recall@5":1.0,"mrr":1.0,"ndcg@5":1.000,"avg_score":0.492},
    {"id":"proc_03","cat":"process","recall@1":1.0,"recall@3":1.0,"recall@5":1.0,"mrr":1.0,"ndcg@5":0.921,"avg_score":0.245},
    {"id":"mem_01","cat":"memory","recall@1":1.0,"recall@3":1.0,"recall@5":1.0,"mrr":1.0,"ndcg@5":1.000,"avg_score":0.339},
    {"id":"mem_02","cat":"memory","recall@1":1.0,"recall@3":1.0,"recall@5":1.0,"mrr":1.0,"ndcg@5":0.850,"avg_score":0.311},
    {"id":"mem_03","cat":"memory","recall@1":1.0,"recall@3":1.0,"recall@5":1.0,"mrr":1.0,"ndcg@5":0.709,"avg_score":0.221},
    {"id":"mem_04","cat":"memory","recall@1":1.0,"recall@3":1.0,"recall@5":1.0,"mrr":1.0,"ndcg@5":1.000,"avg_score":0.316},
    {"id":"sync_01","cat":"sync","recall@1":1.0,"recall@3":1.0,"recall@5":1.0,"mrr":1.0,"ndcg@5":1.000,"avg_score":0.392},
    {"id":"sync_02","cat":"sync","recall@1":1.0,"recall@3":1.0,"recall@5":1.0,"mrr":1.0,"ndcg@5":1.000,"avg_score":0.508},
    {"id":"sync_03","cat":"sync","recall@1":1.0,"recall@3":1.0,"recall@5":1.0,"mrr":1.0,"ndcg@5":0.950,"avg_score":0.209},
    {"id":"fs_01","cat":"fs","recall@1":1.0,"recall@3":1.0,"recall@5":1.0,"mrr":1.0,"ndcg@5":0.921,"avg_score":0.211},
    {"id":"fs_02","cat":"fs","recall@1":0.0,"recall@3":1.0,"recall@5":1.0,"mrr":0.5,"ndcg@5":0.624,"avg_score":0.222},
    {"id":"fs_03","cat":"fs","recall@1":0.0,"recall@3":0.0,"recall@5":0.0,"mrr":0.0,"ndcg@5":0.000,"avg_score":0.198},
    {"id":"intr_01","cat":"interrupt","recall@1":1.0,"recall@3":1.0,"recall@5":1.0,"mrr":1.0,"ndcg@5":0.701,"avg_score":0.155},
    {"id":"intr_02","cat":"interrupt","recall@1":1.0,"recall@3":1.0,"recall@5":1.0,"mrr":1.0,"ndcg@5":1.000,"avg_score":0.375},
    {"id":"intr_03","cat":"interrupt","recall@1":0.0,"recall@3":0.0,"recall@5":1.0,"mrr":0.25,"ndcg@5":0.462,"avg_score":0.270},
    {"id":"boot_01","cat":"boot","recall@1":1.0,"recall@3":1.0,"recall@5":1.0,"mrr":1.0,"ndcg@5":1.000,"avg_score":0.259},
    {"id":"boot_02","cat":"boot","recall@1":1.0,"recall@3":1.0,"recall@5":1.0,"mrr":1.0,"ndcg@5":0.955,"avg_score":0.137},
    {"id":"comp_01","cat":"sync","recall@1":0.0,"recall@3":0.0,"recall@5":1.0,"mrr":0.25,"ndcg@5":0.431,"avg_score":0.341},
    {"id":"comp_02","cat":"sync","recall@1":0.0,"recall@3":0.0,"recall@5":0.0,"mrr":0.0,"ndcg@5":0.000,"avg_score":0.029},
]

# ===================== Figure (a) Category Recall@K =====================
fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(categories)); width = 0.25
r1 = [cat_data[c]["recall@1"] for c in categories]
r3 = [cat_data[c]["recall@3"] for c in categories]
r5 = [cat_data[c]["recall@5"] for c in categories]
bars1 = ax.bar(x - width, r1, width, label='Recall@1', color='#4472C4')
bars2 = ax.bar(x, r3, width, label='Recall@3', color='#ED7D31')
bars3 = ax.bar(x + width, r5, width, label='Recall@5', color='#70AD47')
ax.set_ylabel('Recall Rate', fontsize=12)
ax.set_title('(a) Per-Category Recall@K Comparison', fontsize=13, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=11)
ax.legend(loc='upper left', fontsize=10); ax.set_ylim(0, 1.15); ax.grid(axis='y', alpha=0.3)
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8)
fig.savefig(out_dir / 'fig7_2a_category_recall.png', dpi=300, bbox_inches='tight'); plt.close()

# ===================== Figure (b) MRR & NDCG@5 =====================
fig, ax1 = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(categories)); width = 0.35
mrr_vals = [cat_data[c]["mrr"] for c in categories]
ndcg_vals = [cat_data[c]["ndcg@5"] for c in categories]
bars1 = ax1.bar(x - width/2, mrr_vals, width, label='MRR', color='#5B9BD5')
ax1.set_ylabel('MRR', color='#5B9BD5', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#5B9BD5'); ax1.set_ylim(0, 1.15)
ax1.set_xticks(x); ax1.set_xticklabels(categories, fontsize=11)
ax1.set_title('(b) MRR and NDCG@5 by OS Concept Category', fontsize=13, fontweight='bold')
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, ndcg_vals, width, label='NDCG@5', color='#ED7D31')
ax2.set_ylabel('NDCG@5', color='#ED7D31', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#ED7D31'); ax2.set_ylim(0, 1.15)
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}', xy=(bar.get_x()+bar.get_width()/2, height),
                xytext=(0,2), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax2.annotate(f'{height:.2f}', xy=(bar.get_x()+bar.get_width()/2, height),
                xytext=(0,2), textcoords="offset points", ha='center', va='bottom', fontsize=8)
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right', fontsize=10)
ax1.grid(axis='y', alpha=0.3)
fig.savefig(out_dir / 'fig7_2b_mrr_ndcg.png', dpi=300, bbox_inches='tight'); plt.close()

# ===================== Figure (c) Scatter =====================
fig, ax = plt.subplots(figsize=(12, 5.5))
cat_colors = {"boot":"#A5A5A5", "fs":"#FFC000", "interrupt":"#4472C4",
              "memory":"#ED7D31", "process":"#70AD47", "sync":"#9E480E"}
for q in queries:
    ax.scatter(q["avg_score"], q["recall@1"], color=cat_colors[q["cat"]],
               s=120, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax.annotate(q["id"], (q["avg_score"], q["recall@1"]),
                textcoords="offset points", xytext=(5, 5), fontsize=8)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=cat_colors[c], edgecolor='black', label=c) for c in categories]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
ax.set_xlabel('Average Relevance Score (Top-5)', fontsize=12)
ax.set_ylabel('Recall@1', fontsize=12)
ax.set_title('(c) Per-Query Avg Score vs Recall@1 (Color = Category)', fontsize=13, fontweight='bold')
ax.set_ylim(-0.1, 1.2); ax.grid(alpha=0.3)
fig.savefig(out_dir / 'fig7_2c_query_scatter.png', dpi=300, bbox_inches='tight'); plt.close()

# ===================== Figure (d) Weak Queries =====================
weak = [q for q in queries if q["recall@1"] < 1.0]
fig, ax = plt.subplots(figsize=(10, 5.5)); x = np.arange(len(weak)); width = 0.2
bars1 = ax.bar(x - 1.5*width, [q["recall@1"] for q in weak], width, label='Recall@1', color='#C55A11')
bars2 = ax.bar(x - 0.5*width, [q["recall@3"] for q in weak], width, label='Recall@3', color='#4472C4')
bars3 = ax.bar(x + 0.5*width, [q["recall@5"] for q in weak], width, label='Recall@5', color='#70AD47')
bars4 = ax.bar(x + 1.5*width, [q["ndcg@5"] for q in weak], width, label='NDCG@5', color='#9E480E')
ax.set_ylabel('Metric Value', fontsize=12)
ax.set_title('(d) Under-performing Queries: Recall & Ranking Metrics', fontsize=13, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([q["id"] for q in weak], rotation=15, ha='right')
ax.legend(loc='upper right', fontsize=10); ax.set_ylim(0, 1.15); ax.grid(axis='y', alpha=0.3)
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        if height > 0.05:
            ax.annotate(f'{height:.2f}', xy=(bar.get_x()+bar.get_width()/2, height),
                       xytext=(0,2), textcoords="offset points", ha='center', va='bottom', fontsize=7)
fig.savefig(out_dir / 'fig7_2d_weak_queries.png', dpi=300, bbox_inches='tight'); plt.close()

# ===================== Figure (f) Overall Metrics (v2: 3 metrics only) =====================
fig, ax = plt.subplots(figsize=(9, 5.5))
metrics_names = ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5"]
overall = {"recall@1":{"mean":0.750,"std":0.444},"recall@3":{"mean":0.800,"std":0.410},
         "recall@5":{"mean":0.900,"std":0.308},"mrr":{"mean":0.800,"std":0.368},
         "ndcg@5":{"mean":0.760,"std":0.318}}
means = [overall[m]["mean"] for m in metrics_names]; stds = [overall[m]["std"] for m in metrics_names]
x = np.arange(len(metrics_names))
colors_bar = ['#4472C4','#4472C4','#4472C4','#ED7D31','#ED7D31']
bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors_bar, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Overall Retrieval Metrics (Mean ± Std)', fontsize=13, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(metrics_names, fontsize=11); ax.set_ylim(0, 1.15); ax.grid(axis='y', alpha=0.3)
for bar, mean_val in zip(bars, means):
    ax.annotate(f'{mean_val:.3f}', xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
               xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
fig.savefig(out_dir / 'fig7_2f_overall_metrics_v2.png', dpi=300, bbox_inches='tight'); plt.close()

print("All 5 figures generated successfully.")