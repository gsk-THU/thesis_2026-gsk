#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验三数据分析脚本：从 detail.jsonl 提取指标、输出数据、生成论文图表
使用方法：
    python exp3_analysis.py --input /home/gsk/results/exp3_os_exam/exp3_20260521_073446_detail.jsonl --output ./figures
"""

import argparse
import json
import statistics
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

LEVELS = ["A", "B", "C", "D"]
LEVEL_NUM = {"A": 4, "B": 3, "C": 2, "D": 1}


def load_detail_jsonl(file_path: str):
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except Exception:
                continue
    return samples


def compute_metrics(samples):
    graded = [s for s in samples if s.get('scores')]
    failed = [s for s in samples if not s.get('scores')]

    level_scores = {lvl: [] for lvl in LEVELS}
    chapter_level_scores = {}
    records = []

    for s in graded:
        scores = [q.get("final_score", 0) for q in s.get("scores", [])]
        avg = statistics.mean(scores) if scores else 0
        level = s.get("level", "")
        chapter = s.get("chapter", "")

        if level in level_scores:
            level_scores[level].append(avg)
        chapter_level_scores.setdefault(chapter, {lvl: [] for lvl in LEVELS})
        chapter_level_scores[chapter][level].append(avg)

        pred = "A" if avg >= 8.5 else "B" if avg >= 7.0 else "C" if avg >= 5.0 else "D"
        records.append({
            "sample_id": s.get("sample_id"),
            "chapter": chapter,
            "true_level": level,
            "true_num": LEVEL_NUM.get(level, 0),
            "avg_score": avg,
            "pred_level": pred,
            "overall_confidence": s.get("overall", {}).get("confidence", 0),
        })

    desc = {}
    for lvl in LEVELS:
        vals = level_scores[lvl]
        if vals:
            desc[lvl] = {
                "n": len(vals), "mean": statistics.mean(vals),
                "median": statistics.median(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0,
                "min": min(vals), "max": max(vals),
            }

    mono_ok = 0
    for ch, sc in chapter_level_scores.items():
        means = [statistics.mean(sc[l]) if sc[l] else 0 for l in LEVELS]
        if all(means[i] > means[i + 1] for i in range(3)):
            mono_ok += 1
    monotonicity = mono_ok / len(chapter_level_scores) if chapter_level_scores else 0

    true_nums = [r["true_num"] for r in records]
    avg_scores = [r["avg_score"] for r in records]
    rho, pval = stats.spearmanr(true_nums, avg_scores) if len(records) > 2 else (0.0, 1.0)

    correct = sum(1 for r in records if r["pred_level"] == r["true_level"])
    acc = correct / len(records) if records else 0
    adj = sum(1 for r in records if abs(LEVEL_NUM[r["pred_level"]] - r["true_num"]) <= 1)
    adj_acc = adj / len(records) if records else 0

    means = [desc[l]["mean"] for l in LEVELS if l in desc]
    deltas = {"AB": means[0]-means[1], "BC": means[1]-means[2], "CD": means[2]-means[3]} if len(means) == 4 else {"AB": 0, "BC": 0, "CD": 0}

    consistency = {"高": 0, "中": 0, "低": 0}
    reeval_cnt = 0
    total_q = 0
    stage1_variances = []

    for s in graded:
        for q in s.get("scores", []):
            total_q += 1
            c = q.get("confidence", "中")
            consistency[c] = consistency.get(c, 0) + 1
            rt = q.get("reevaluation", {})
            if isinstance(rt, dict) and rt.get("trigger_report", {}).get("triggered", False):
                reeval_cnt += 1
            tscores = [t.get("score") for t in q.get("teacher_scores", []) if t.get("score") is not None]
            if len(tscores) > 1:
                stage1_variances.append(statistics.stdev(tscores))

    chapter_rhos = []
    chapter_rho_map = {}
    for ch in chapter_level_scores:
        ch_records = [r for r in records if r["chapter"] == ch]
        if len(ch_records) > 2:
            tn = [r["true_num"] for r in ch_records]
            av = [r["avg_score"] for r in ch_records]
            try:
                r_ch, _ = stats.spearmanr(tn, av)
                chapter_rhos.append(r_ch)
                chapter_rho_map[ch] = float(r_ch)
            except:
                pass
    cross_chapter_stability = statistics.stdev(chapter_rhos) if len(chapter_rhos) > 1 else 0

    lat_summary = {}
    for key in ["answer_gen_ms", "scoring_total_ms", "chairman_ms", "per_question_ms"]:
        vals = [s.get("latency", {}).get(key, 0) for s in graded if s.get("latency", {}).get(key, 0) > 0]
        if vals:
            lat_summary[key] = {
                "mean": statistics.mean(vals), "median": statistics.median(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0,
                "min": min(vals), "max": max(vals),
            }

    return {
        "graded": graded, "failed": failed, "records": records,
        "level_scores": level_scores, "chapter_level_scores": chapter_level_scores,
        "desc": desc, "monotonicity": monotonicity, "mono_ok": mono_ok,
        "total_chapters": len(chapter_level_scores),
        "rho": rho, "pval": pval,
        "acc": acc, "adj_acc": adj_acc, "deltas": deltas,
        "consistency": consistency, "total_q": total_q,
        "reeval_rate": reeval_cnt / total_q if total_q else 0,
        "stage1_variances": stage1_variances,
        "cross_chapter_stability": cross_chapter_stability,
        "chapter_rho_map": chapter_rho_map,
        "lat_summary": lat_summary,
    }


def plot_fig1(metrics, output_dir):
    """图1：等级得分分布箱线图 + 均值柱状图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    LEVELS = ["A", "B", "C", "D"]
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]

    # ====== 输出数据 ======
    print("\n" + "="*60)
    print("【图 7-1 数据】Score Distribution by True Level")
    print("="*60)
    print("\n表 7-1 各等级描述性统计")
    print("-"*60)
    print(f"{'等级':<6} {'N':<4} {'Mean':<8} {'Median':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("-"*60)
    for lvl in LEVELS:
        d = metrics["desc"].get(lvl, {})
        print(f"{lvl:<6} {d.get('n',0):<4} {d.get('mean',0):<8.2f} {d.get('median',0):<8.2f} "
              f"{d.get('std',0):<8.2f} {d.get('min',0):<8.2f} {d.get('max',0):<8.2f}")
    print("-"*60)

    # 箱线图原始数据
    print("\n箱线图五数概括（用于论文描述）:")
    for lvl in LEVELS:
        vals = sorted(metrics["level_scores"][lvl])
        if vals:
            q1 = np.percentile(vals, 25)
            q3 = np.percentile(vals, 75)
            print(f"  {lvl}: min={min(vals):.2f}, Q1={q1:.2f}, median={statistics.median(vals):.2f}, "
                  f"Q3={q3:.2f}, max={max(vals):.2f}, IQR={q3-q1:.2f}")

    # ====== 绘图 ======
    data = [metrics["level_scores"][l] for l in LEVELS]
    bp = axes[0].boxplot(data, tick_labels=[f"{l} (n={len(metrics['level_scores'][l])})" for l in LEVELS], 
                          patch_artist=True, widths=0.6, showmeans=True,
                          meanprops=dict(marker='D', markerfacecolor='black', markeredgecolor='black', markersize=6))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_ylabel("Score", fontsize=12)
    axes[0].set_title("(a) Score Distribution by True Level", fontsize=13, fontweight='bold')
    axes[0].set_ylim(0, 11)
    axes[0].axhline(y=8.5, color='green', linestyle='--', alpha=0.5)
    axes[0].axhline(y=7.0, color='blue', linestyle='--', alpha=0.5)
    axes[0].axhline(y=5.0, color='orange', linestyle='--', alpha=0.5)
    axes[0].grid(axis='y', alpha=0.3)

    means = [metrics["desc"][l]["mean"] for l in LEVELS]
    stds = [metrics["desc"][l]["std"] for l in LEVELS]
    bars = axes[1].bar(LEVELS, means, yerr=stds, capsize=6, color=colors, edgecolor='black', alpha=0.8)
    axes[1].set_ylabel("Mean Score", fontsize=12)
    axes[1].set_title("(b) Mean Score ± SD by Level", fontsize=13, fontweight='bold')
    axes[1].set_ylim(0, 11)
    for bar, m, s in zip(bars, means, stds):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.2, 
                    f"{m:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{output_dir}/fig1_score_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[SAVE] fig1_score_distribution.png -> {output_dir}")


def plot_fig2(metrics, output_dir):
    """图2：混淆矩阵 + 各章节单调性趋势"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    LEVELS = ["A", "B", "C", "D"]

    # ====== 输出数据 ======
    print("\n" + "="*60)
    print("【图 7-2 数据】Classification & Monotonicity")
    print("="*60)

    cm = {t: {p: 0 for p in LEVELS} for t in LEVELS}
    for r in metrics["records"]:
        cm[r["true_level"]][r["pred_level"]] += 1

    print("\n混淆矩阵数据（行=真实等级，列=预测等级）：")
    print("-"*50)
    header = "      " + "  ".join([f"Pred {p:<4}" for p in LEVELS])
    print(header)
    print("-"*50)
    for t in LEVELS:
        row = "  ".join([f"{cm[t][p]:<8}" for p in LEVELS])
        print(f"True {t}  {row}")
    print("-"*50)

    total = len(metrics["records"])
    correct = sum(1 for r in metrics["records"] if r["pred_level"] == r["true_level"])
    adj = sum(1 for r in metrics["records"] if abs(LEVEL_NUM[r["pred_level"]] - r["true_num"]) <= 1)
    print(f"\n严格准确率: {correct}/{total} = {correct/total:.2%}")
    print(f"相邻准确率: {adj}/{total} = {adj/total:.2%}")

    print("\n各章节等级平均得分（单调性检验）：")
    print("-"*60)
    print(f"{'章节':<8} {'A':<8} {'B':<8} {'C':<8} {'D':<8} {'单调?':<6}")
    print("-"*60)
    for ch in sorted(metrics["chapter_level_scores"].keys()):
        means = [statistics.mean(metrics["chapter_level_scores"][ch][l]) 
                 if metrics["chapter_level_scores"][ch][l] else 0 for l in LEVELS]
        is_mono = all(means[i] > means[i+1] for i in range(3))
        mono_str = "✓" if is_mono else "✗"
        print(f"{ch:<8} {means[0]:<8.2f} {means[1]:<8.2f} {means[2]:<8.2f} {means[3]:<8.2f} {mono_str:<6}")
    print("-"*60)
    print(f"单调章节数: {metrics['mono_ok']}/{metrics['total_chapters']} = {metrics['monotonicity']:.2%}")

    # ====== 绘图 ======
    mat = [[cm[t][p] for p in LEVELS] for t in LEVELS]
    im = axes[0].imshow(mat, cmap="Blues", aspect='auto')
    axes[0].set_xticks(range(4))
    axes[0].set_yticks(range(4))
    axes[0].set_xticklabels([f"Pred {l}" for l in LEVELS], fontsize=11)
    axes[0].set_yticklabels([f"True {l}" for l in LEVELS], fontsize=11)
    axes[0].set_title("(a) Classification Confusion Matrix", fontsize=13, fontweight='bold')
    for i in range(4):
        for j in range(4):
            color = 'white' if mat[i][j] > max([max(row) for row in mat]) / 2 else 'black'
            axes[0].text(j, i, mat[i][j], ha="center", va="center", fontsize=14, fontweight="bold", color=color)
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    for ch in sorted(metrics["chapter_level_scores"].keys()):
        means = [statistics.mean(metrics["chapter_level_scores"][ch][l]) 
                 if metrics["chapter_level_scores"][ch][l] else 0 for l in LEVELS]
        axes[1].plot(LEVELS, means, marker="o", markersize=8, linewidth=2, label=ch)
    axes[1].set_ylabel("Mean Score", fontsize=12)
    axes[1].set_xlabel("True Level", fontsize=12)
    axes[1].set_title("(b) Score Trend by Chapter (Monotonicity)", fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=9, ncol=2, loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 11)

    plt.tight_layout()
    fig.savefig(f"{output_dir}/fig2_classification_monotonicity.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[SAVE] fig2_classification_monotonicity.png -> {output_dir}")


def plot_fig3(metrics, output_dir):
    """图3：相邻分差 + 置信度 + 重评触发率"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # ====== 输出数据 ======
    print("\n" + "="*60)
    print("【图 7-3 数据】Score Gaps, Confidence & Re-evaluation")
    print("="*60)

    d = metrics["deltas"]
    print("\n相邻等级平均分差：")
    print(f"  Δ(A-B) = {d['AB']:.2f}")
    print(f"  Δ(B-C) = {d['BC']:.2f}")
    print(f"  Δ(C-D) = {d['CD']:.2f}")

    c = metrics["consistency"]
    total_q = metrics["total_q"]
    print(f"\n主席置信度分布 (N={total_q}):")
    print(f"  高置信度: {c['高']} ({c['高']/total_q:.2%})")
    print(f"  中置信度: {c['中']} ({c['中']/total_q:.2%})")
    print(f"  低置信度: {c['低']} ({c['低']/total_q:.2%})")

    reeval_rate = metrics["reeval_rate"]
    print(f"\n重评触发率: {metrics['reeval_rate']*100:.2f}%")

    if metrics["stage1_variances"]:
        sv = metrics["stage1_variances"]
        print(f"\n阶段1评分方差统计 (N={len(sv)}):")
        print(f"  均值: {statistics.mean(sv):.3f}")
        print(f"  中位数: {statistics.median(sv):.3f}")
        print(f"  标准差: {statistics.stdev(sv):.3f}")
        print(f"  最小值: {min(sv):.3f}")
        print(f"  最大值: {max(sv):.3f}")

    # ====== 绘图 ======
    delta_labels = ["A-B", "B-C", "C-D"]
    delta_vals = [metrics["deltas"]["AB"], metrics["deltas"]["BC"], metrics["deltas"]["CD"]]
    colors_delta = ["#27ae60", "#2980b9", "#c0392b"]
    bars = axes[0].bar(delta_labels, delta_vals, color=colors_delta, edgecolor='black', alpha=0.8, width=0.6)
    axes[0].set_ylabel("Score Difference", fontsize=12)
    axes[0].set_title("(a) Adjacent-Level Score Gaps", fontsize=13, fontweight='bold')
    for bar, v in zip(bars, delta_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f"{v:.2f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[0].set_ylim(0, max(delta_vals) * 1.3)
    axes[0].grid(axis='y', alpha=0.3)

    c = metrics["consistency"]
    sizes = [c["高"], c["中"], c["低"]]
    labels = [f"High\n({c['高']})" if metrics['total_q'] else "High",
              f"Medium\n({c['中']})" if metrics['total_q'] else "Medium",
              f"Low\n({c['低']})" if metrics['total_q'] else "Low"]
    colors_pie = ["#2ecc71", "#f1c40f", "#e74c3c"]
    axes[1].pie(sizes, labels=labels, autopct='', startangle=90, colors=colors_pie, explode=(0.03, 0.03, 0.03))
    axes[1].set_title("(b) Chairman Confidence Distribution", fontsize=13, fontweight='bold')

    reeval_rate = metrics["reeval_rate"]
    bar = axes[2].bar(["Re-eval Rate"], [reeval_rate * 100], color="#e67e22", edgecolor='black', alpha=0.8, width=0.4)
    axes[2].set_ylabel("Trigger Rate (%)", fontsize=11)
    axes[2].set_title("(c) Re-evaluation & Stage-1 Variance", fontsize=13, fontweight='bold')
    for b in bar:
        axes[2].text(b.get_x() + b.get_width()/2, b.get_height() + 0.1, 
                    f"{reeval_rate*100:.2f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
    if metrics["stage1_variances"]:
        ax_var = axes[2].twinx()
        bp = ax_var.boxplot([metrics["stage1_variances"]], positions=[1.4], widths=0.3, 
                            patch_artist=True, showmeans=True)
        bp["boxes"][0].set_facecolor("#3498db")
        bp["boxes"][0].set_alpha(0.6)
        ax_var.set_ylabel("Stage-1 Score Std. Dev.", fontsize=11)

    plt.tight_layout()
    fig.savefig(f"{output_dir}/fig3_consistency_reeval.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[SAVE] fig3_consistency_reeval.png -> {output_dir}")


def plot_fig4(metrics, output_dir):
    """图4：延迟分布 + 跨章节稳定性"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ====== 输出数据 ======
    print("\n" + "="*60)
    print("【图 7-4 数据】Latency & Cross-Chapter Stability")
    print("="*60)

    print("\n系统延迟统计 (ms):")
    print("-"*70)
    print(f"{'阶段':<20} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-"*70)
    for key, label in [("answer_gen_ms", "Answer Gen"), ("scoring_total_ms", "Council Scoring"), 
                       ("chairman_ms", "Chairman Eval"), ("per_question_ms", "Per Question")]:
        if key in metrics["lat_summary"]:
            v = metrics["lat_summary"][key]
            print(f"{label:<20} {v['mean']:<10.0f} {v['median']:<10.0f} {v['std']:<10.0f} "
                  f"{v['min']:<10.0f} {v['max']:<10.0f}")
    print("-"*70)

    print("\n跨章节 Spearman ρ 稳定性:")
    print("-"*40)
    for ch, rho in sorted(metrics["chapter_rho_map"].items()):
        print(f"  {ch}: ρ = {rho:.3f}")
    print("-"*40)
    print(f"  跨章节 σ(ρ) = {metrics['cross_chapter_stability']:.3f}")

    # ====== 绘图 ======
    lat_keys = ["answer_gen_ms", "scoring_total_ms", "chairman_ms"]
    lat_labels = ["Answer Gen", "Council Scoring", "Chairman Eval"]
    lat_data = []
    for k in lat_keys:
        if k in metrics["lat_summary"]:
            vals = [s.get("latency", {}).get(k, 0) for s in metrics["graded"] if s.get("latency", {}).get(k, 0) > 0]
            lat_data.append(vals)
        else:
            lat_data.append([])

    bp = axes[0].boxplot(lat_data, tick_labels=lat_labels, patch_artist=True, widths=0.6)
    colors_lat = ["#3498db", "#9b59b6", "#e74c3c"]
    for patch, color in zip(bp["boxes"], colors_lat):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_ylabel("Latency (ms, log scale)", fontsize=12)
    axes[0].set_title("(a) System Latency Breakdown", fontsize=13, fontweight='bold')
    axes[0].set_yscale("log")
    axes[0].grid(axis='y', alpha=0.3)

    chapter_rhos = []
    chapter_labels = []
    for ch in sorted(metrics["chapter_level_scores"].keys()):
        ch_records = [r for r in metrics["records"] if r["chapter"] == ch]
        if len(ch_records) > 2:
            tn = [r["true_num"] for r in ch_records]
            av = [r["avg_score"] for r in ch_records]
            try:
                r_ch, _ = stats.spearmanr(tn, av)
                chapter_rhos.append(r_ch)
                chapter_labels.append(ch)
            except:
                pass

    x_pos = range(len(chapter_labels))
    bars = axes[1].bar(x_pos, chapter_rhos, color="#2c3e50", edgecolor='black', alpha=0.8, width=0.6)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(chapter_labels, fontsize=11)
    axes[1].set_ylabel("Spearman ρ", fontsize=12)
    axes[1].set_title("Cross-Chapter Discrimination Stability", fontsize=13, fontweight='bold')
    axes[1].set_ylim(0.5, 1.05)
    axes[1].axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Strong threshold')
    axes[1].axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Moderate threshold')

    if len(chapter_rhos) > 1:
        std_rho = statistics.stdev(chapter_rhos)
        axes[1].text(len(chapter_labels)/2 - 0.5, 0.6, 
                    f"σ(ρ) = {std_rho:.3f}", fontsize=11, fontweight='bold', color='#c0392b',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    for bar, v in zip(bars, chapter_rhos):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f"{v:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{output_dir}/fig4_latency_stability.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[SAVE] fig4_latency_stability.png -> {output_dir}")


def plot_fig5(metrics, samples, output_dir):
    """图5：追问深度与得分关系"""
    # 计算每个样本的对话深度
    depth_analysis = []
    for s in samples:
        if not s.get("scores"):
            continue
        total_rounds = 0
        count = 0
        for qid, dialogue in s.get("dialogues", {}).items():
            rounds = len(dialogue) // 2
            total_rounds += rounds
            count += 1
        avg_depth = total_rounds / count if count > 0 else 1

        scores = [q.get("final_score", 0) for q in s.get("scores", [])]
        avg_score = statistics.mean(scores) if scores else 0

        depth_analysis.append({
            "sample_id": s["sample_id"],
            "chapter": s["chapter"],
            "level": s["level"],
            "avg_depth": avg_depth,
            "avg_score": avg_score,
            "num_questions": count,
        })

    df_depth = pd.DataFrame(depth_analysis)

    # ====== 输出数据 ======
    print("\n" + "="*60)
    print("【图 7-5 数据】Follow-up Depth vs. Score")
    print("="*60)

    print("\n各等级追问深度统计：")
    print("-"*60)
    print(f"{'等级':<6} {'N':<4} {'MeanDepth':<10} {'StdDepth':<10} {'DepthRange':<15}")
    print("-"*60)
    for lvl in LEVELS:
        subset = df_depth[df_depth["level"] == lvl]
        if len(subset) > 0:
            mean_d = subset["avg_depth"].mean()
            std_d = subset["avg_depth"].std() if len(subset) > 1 else 0
            min_d = subset["avg_depth"].min()
            max_d = subset["avg_depth"].max()
            corr = subset["avg_depth"].corr(subset["avg_score"])
            print(f"{lvl:<6} {len(subset):<4} {mean_d:<10.2f} {std_d:<10.2f} [{min_d:.1f}, {max_d:.1f}]")
            if not pd.isna(corr):
                print(f"       深度-得分相关系数: r = {corr:.3f}")
    print("-"*60)

    # ====== 绘图 ======
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"A": "#2ecc71", "B": "#3498db", "C": "#f39c12", "D": "#e74c3c"}
    markers = {"A": "o", "B": "s", "C": "^", "D": "D"}

    for lvl in LEVELS:
        subset = df_depth[df_depth["level"] == lvl]
        ax.scatter(subset["avg_depth"], subset["avg_score"], 
                  c=colors[lvl], marker=markers[lvl], s=100, alpha=0.7, 
                  edgecolors='black', linewidth=0.5, label=f"Level {lvl} (n={len(subset)})")
        if len(subset) > 2 and subset["avg_depth"].nunique() > 1:
            z = np.polyfit(subset["avg_depth"], subset["avg_score"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(subset["avg_depth"].min(), subset["avg_depth"].max(), 50)
            ax.plot(x_line, p(x_line), color=colors[lvl], linestyle='--', alpha=0.8, linewidth=2)

    ax.set_xlabel("Average Dialogue Depth (rounds)", fontsize=13)
    ax.set_ylabel("Average Score", fontsize=13)
    ax.set_title("Fig. 7-5  Relationship between Follow-up Depth and Scores by Level", 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 11)

    plt.tight_layout()
    fig.savefig(f"{output_dir}/fig5_depth_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[SAVE] fig5_depth_analysis.png -> {output_dir}")


def save_data_json(metrics, output_dir):
    """将所有指标数据保存为 JSON，方便论文直接引用"""
    export = {
        "descriptive_stats": metrics["desc"],
        "monotonicity": {
            "ratio": metrics["monotonicity"],
            "monotonic_chapters": metrics["mono_ok"],
            "total_chapters": metrics["total_chapters"],
        },
        "spearman": {
            "rho": float(metrics["rho"]),
            "p_value": float(metrics["pval"]),
        },
        "classification": {
            "accuracy": metrics["acc"],
            "adjacent_accuracy": metrics["adj_acc"],
            "deltas": metrics["deltas"],
        },
        "consistency": {
            "high_ratio": metrics["consistency"]["高"] / metrics["total_q"] if metrics["total_q"] else 0,
            "medium_ratio": metrics["consistency"]["中"] / metrics["total_q"] if metrics["total_q"] else 0,
            "low_ratio": metrics["consistency"]["低"] / metrics["total_q"] if metrics["total_q"] else 0,
            "raw_counts": metrics["consistency"],
        },
        "reevaluation": {
            "trigger_rate": metrics["reeval_rate"],
            "stage1_variance_summary": {
                "mean": statistics.mean(metrics["stage1_variances"]) if metrics["stage1_variances"] else 0,
                "median": statistics.median(metrics["stage1_variances"]) if metrics["stage1_variances"] else 0,
                "std": statistics.stdev(metrics["stage1_variances"]) if len(metrics["stage1_variances"]) > 1 else 0,
            },
        },
        "cross_chapter_stability": {
            "std_of_rho": metrics["cross_chapter_stability"],
            "chapter_rho_map": metrics["chapter_rho_map"],
        },
        "latency": metrics["lat_summary"],
    }

    out_path = Path(output_dir) / "metrics_data.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] 指标数据 JSON -> {out_path}")
    return export


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to exp3_*_detail.jsonl")
    parser.add_argument("--output", default="./figures", help="Output directory for figures")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] Reading {args.input}...")
    samples = load_detail_jsonl(args.input)
    print(f"[LOAD] {len(samples)} samples loaded")

    print("[COMPUTE] Computing metrics...")
    metrics = compute_metrics(samples)

    print(f"\n{'='*60}")
    print("核心指标汇总")
    print(f"{'='*60}")
    print(f"  总样本数: {len(samples)}")
    print(f"  评分成功: {len(metrics['graded'])}")
    print(f"  评分失败: {len(metrics['failed'])}")
    print(f"  总评分题目: {metrics['total_q']}")
    print(f"  单调性: {metrics['monotonicity']:.2%} ({metrics['mono_ok']}/{metrics['total_chapters']})")
    print(f"  Spearman ρ: {metrics['rho']:.3f} (p={metrics['pval']:.4f})")
    print(f"  严格准确率: {metrics['acc']:.2%}")
    print(f"  相邻准确率: {metrics['adj_acc']:.2%}")
    print(f"  重评触发率: {metrics['reeval_rate']:.2%}")
    print(f"  跨章节 σ(ρ): {metrics['cross_chapter_stability']:.3f}")
    print(f"{'='*60}")

    print("[PLOT] Generating figures with data output...")
    plot_fig1(metrics, str(output_dir))
    plot_fig2(metrics, str(output_dir))
    plot_fig3(metrics, str(output_dir))
    plot_fig4(metrics, str(output_dir))
    plot_fig5(metrics, samples, str(output_dir))

    save_data_json(metrics, str(output_dir))

    print(f"\n[DONE] All outputs saved to {output_dir}")


if __name__ == "__main__":
    main()