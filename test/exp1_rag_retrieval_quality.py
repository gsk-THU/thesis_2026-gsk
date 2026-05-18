#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验一：RAG 知识库检索质量评估
================================
评估 OS 课程知识库（ucore/rcore）的检索效果：
- Recall@K：期望概念在 Top-K 中的召回率
- MRR：首个相关结果的平均倒数排名
- NDCG@K：归一化折损累积增益
- 语义相关性分布与来源多样性

运行方式：
    cd /home/gsk/thesis_2026-gsk/test
    python exp1_rag_retrieval_quality.py [--collection rcore_2025s|ucore_2025s]

依赖（可选，用于图表）：
    pip install matplotlib numpy
"""

import argparse
import json
import sys
import os
import math
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

# ==================== 路径配置 ====================
_PROJECT_ROOTS = [
    "/home/gsk/thesis_2026-gsk/questions",
    "/home/gsk/thesis_2026-gsk/chroma",
    os.path.dirname(os.path.abspath(__file__)),
]
for _p in _PROJECT_ROOTS:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from database import KnowledgeBase
except ImportError as e:
    print(f"[ERROR] 无法导入 KnowledgeBase: {e}")
    sys.exit(1)

# 可视化可选
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    _HAS_VIZ = True
except ImportError:
    _HAS_VIZ = False


# ==================== 数据模型 ====================

@dataclass
class TestQuery:
    """测试查询定义"""
    id: str
    query: str
    expected_keywords: List[str]      # 期望在结果内容中出现的关键词
    expected_sources: List[str]       # 期望来源标识（文件名/路径片段）
    category: str                   # OS 概念类别
    difficulty: str                   # easy / medium / hard
    description: str = ""


@dataclass
class RetrievalResult:
    """单个查询的检索结果"""
    query_id: str
    query: str
    category: str
    results: List[Dict[str, Any]]   # 原始检索结果（Top-K）
    metrics: Dict[str, float] = field(default_factory=dict)


# ==================== 评估器 ====================

class RAGRetrievalEvaluator:
    """RAG 检索质量评估器"""

    # 预定义 OS 课程测试查询集（20个，覆盖核心概念）
    DEFAULT_QUERIES: List[TestQuery] = [
        # 进程管理
        TestQuery("proc_01", "进程调度算法实现",
                  ["schedule", "scheduler", "stride", "priority", "round_robin"],
                  ["sched", "process", "task"], "process", "medium",
                  "查询进程调度相关实现"),
        TestQuery("proc_02", "进程 fork 和 exec 的实现原理",
                  ["fork", "exec", "clone", "spawn", "process"],
                  ["process", "fork", "task"], "process", "hard",
                  "查询进程创建与执行"),
        TestQuery("proc_03", "进程控制块 PCB 结构设计",
                  ["pcb", "process_control_block", "task_struct", "state"],
                  ["pcb", "process", "task"], "process", "medium",
                  "查询 PCB 结构"),

        # 内存管理
        TestQuery("mem_01", "页表遍历与地址转换机制",
                  ["page_table", "pte", "pagetable", "vpn", "ppn", "translation"],
                  ["mm", "memory", "page", "paging"], "memory", "hard",
                  "查询页表机制"),
        TestQuery("mem_02", "缺页异常 page fault 处理流程",
                  ["page_fault", "pgfault", "trap", "handle_page_fault"],
                  ["trap", "page", "fault", "exception"], "memory", "hard",
                  "查询缺页处理"),
        TestQuery("mem_03", "内存分配与释放算法",
                  ["malloc", "free", "alloc", "heap", "buddy", "slab"],
                  ["mm", "memory", "alloc"], "memory", "medium",
                  "查询内存分配"),
        TestQuery("mem_04", "虚拟内存映射 mmap 实现",
                  ["mmap", "unmap", "mapping", "vma", "virtual_memory"],
                  ["mm", "mmap", "memory"], "memory", "hard",
                  "查询 mmap 实现"),

        # 同步机制
        TestQuery("sync_01", "信号量 semaphore 同步机制",
                  ["semaphore", "sem", "wait", "signal", "p_v"],
                  ["sync", "semaphore", "lock"], "sync", "medium",
                  "查询信号量"),
        TestQuery("sync_02", "互斥锁 mutex 实现原理",
                  ["mutex", "lock", "unlock", "spinlock", "critical"],
                  ["sync", "lock", "mutex"], "sync", "medium",
                  "查询互斥锁"),
        TestQuery("sync_03", "条件变量 condition variable",
                  ["condvar", "condition", "wait", "notify", "broadcast"],
                  ["sync", "cond", "condition"], "sync", "hard",
                  "查询条件变量"),

        # 文件系统
        TestQuery("fs_01", "inode 结构设计与文件索引",
                  ["inode", "dinode", "file_index", "direct", "indirect"],
                  ["fs", "file", "inode"], "fs", "medium",
                  "查询 inode"),
        TestQuery("fs_02", "文件系统挂载 mount 流程",
                  ["mount", "vfs", "superblock", "filesystem"],
                  ["fs", "mount", "vfs"], "fs", "hard",
                  "查询文件系统挂载"),
        TestQuery("fs_03", "目录项 dentry 缓存机制",
                  ["dentry", "directory", "path", "lookup"],
                  ["fs", "dentry", "dir"], "fs", "medium",
                  "查询目录项"),

        # 中断与系统调用
        TestQuery("intr_01", "中断处理程序 IRQ handler",
                  ["interrupt", "irq", "handler", "trap", "idt"],
                  ["trap", "interrupt", "irq"], "interrupt", "medium",
                  "查询中断处理"),
        TestQuery("intr_02", "系统调用 syscall 分发机制",
                  ["syscall", "system_call", "trap", "dispatch", "handler"],
                  ["trap", "syscall", "interrupt"], "interrupt", "hard",
                  "查询系统调用"),
        TestQuery("intr_03", "时钟中断与定时器",
                  ["timer", "tick", "clock", "time_interrupt"],
                  ["trap", "timer", "clock"], "interrupt", "easy",
                  "查询时钟中断"),

        # 启动与架构
        TestQuery("boot_01", "内核启动流程 boot loader",
                  ["boot", "loader", "entry", "start", "init"],
                  ["boot", "entry", "start"], "boot", "medium",
                  "查询启动流程"),
        TestQuery("boot_02", "多核 CPU 启动 AP 初始化",
                  ["smp", "ap", "cpu", "multiprocessor", "start_ap"],
                  ["smp", "cpu", "boot"], "boot", "hard",
                  "查询多核启动"),

        # 综合
        TestQuery("comp_01", "死锁检测与避免算法",
                  ["deadlock", "detect", "avoid", "banker", "cycle"],
                  ["sync", "deadlock"], "sync", "hard",
                  "查询死锁"),
        TestQuery("comp_02", "RCU 读拷贝更新机制",
                  ["rcu", "read_copy_update", "quiescent"],
                  ["sync", "rcu"], "sync", "hard",
                  "查询 RCU"),
    ]

    def __init__(self, persist_dir: str = "/home/gsk/chroma",
                 collection_name: str = "rcore_2025s",
                 top_k_values: List[int] = None):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.top_k_values = top_k_values or [1, 3, 5]
        self.kb = KnowledgeBase(persist_dir, collection_name)
        self.results: List[RetrievalResult] = []

        # 日志
        self.log_file = Path(
            f"exp1_log_{collection_name}_{datetime.now():%Y%m%d_%H%M%S}.txt"
        )

        # 打印知识库状态
        try:
            stats = self.kb.get_stats()
            self.log(f"[INIT] 知识库状态: {stats}")
        except Exception as e:
            self.log(f"[INIT] 无法获取知识库状态: {e}")

    def log(self, msg: str):
        print(msg)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now():%H:%M:%S}] {msg}\n")

    # -------------------- 核心评估逻辑 --------------------

    def evaluate_all(self, queries: Optional[List[TestQuery]] = None) -> List[RetrievalResult]:
        """执行全部查询评估"""
        queries = queries or self.DEFAULT_QUERIES
        self.log(
            f"[RUN] collection={self.collection_name}, queries={len(queries)}, "
            f"top_k={self.top_k_values}"
        )

        for q in queries:
            self.log(f"  [QUERY] [{q.id}] {q.query} (cat={q.category}, diff={q.difficulty})")
            result = self._evaluate_single(q)
            self.results.append(result)
            self.log(
                f"    -> recall@1={result.metrics.get('recall@1', 0):.2f}, "
                f"mrr={result.metrics.get('mrr', 0):.2f}, "
                f"ndcg@5={result.metrics.get('ndcg@5', 0):.2f}"
            )

        self.log(f"[DONE] 共评估 {len(self.results)} 个查询")
        return self.results

    def _evaluate_single(self, query: TestQuery) -> RetrievalResult:
        """评估单个查询"""
        max_k = max(self.top_k_values)
        raw_results = self.kb.query(query.query, n_results=max_k)

        # 无结果保护
        if not raw_results:
            return RetrievalResult(
                query_id=query.id,
                query=query.query,
                category=query.category,
                results=[],
                metrics={
                    **{f"recall@{k}": 0.0 for k in self.top_k_values},
                    **{f"source_recall@{k}": 0.0 for k in self.top_k_values},
                    "mrr": 0.0, "ndcg@5": 0.0,
                    "avg_score": 0.0, "max_score": 0.0, "min_score": 0.0,
                }
            )

        metrics: Dict[str, float] = {}

        # 1. Recall@K（关键词 + 来源）
        for k in self.top_k_values:
            top_k = raw_results[:k]

            # 关键词召回：至少命中一个关键词即算召回
            keyword_hit = any(
                any(kw.lower() in r.get("content", "").lower() for r in top_k)
                for kw in query.expected_keywords
            )
            metrics[f"recall@{k}"] = 1.0 if keyword_hit else 0.0

            # 来源召回
            source_hit = any(
                any(src.lower() in r.get("metadata", {}).get("source", "").lower()
                    for r in top_k)
                for src in query.expected_sources
            )
            metrics[f"source_recall@{k}"] = 1.0 if source_hit else 0.0

        # 2. MRR
        mrr = 0.0
        for i, r in enumerate(raw_results, 1):
            content = r.get("content", "")
            if any(kw.lower() in content.lower() for kw in query.expected_keywords):
                mrr = 1.0 / i
                break
        metrics["mrr"] = mrr

        # 3. NDCG@5（简化分级：2=强相关，1=弱相关，0=不相关）
        def _rel_score(result: Dict) -> int:
            content = result.get("content", "")
            hits = sum(1 for kw in query.expected_keywords if kw.lower() in content.lower())
            return min(hits, 2)

        rels = [_rel_score(r) for r in raw_results[:5]]
        dcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(rels))
        ideal = sorted(rels, reverse=True)
        idcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal))
        metrics["ndcg@5"] = dcg / idcg if idcg > 0 else 0.0

        # 4. 相似度分数统计
        scores = [r.get("relevance_score", 0.0) for r in raw_results[:5]]
        metrics["avg_score"] = statistics.mean(scores) if scores else 0.0
        metrics["max_score"] = max(scores) if scores else 0.0
        metrics["min_score"] = min(scores) if scores else 0.0

        # 5. 来源多样性（Top-5 中不同来源的数量）
        sources = {r.get("metadata", {}).get("source", "unknown") for r in raw_results[:5]}
        metrics["source_diversity"] = len(sources)

        return RetrievalResult(
            query_id=query.id,
            query=query.query,
            category=query.category,
            results=raw_results[:max_k],
            metrics=metrics
        )

    # -------------------- 汇总统计 --------------------

    def compute_summary(self) -> Dict[str, Any]:
        """计算汇总统计"""
        if not self.results:
            return {}

        summary = {
            "collection": self.collection_name,
            "total_queries": len(self.results),
            "timestamp": datetime.now().isoformat(),
            "overall": {},
            "by_category": {},
        }

        # 收集指标
        all_metrics: Dict[str, List[float]] = {}
        cat_metrics: Dict[str, Dict[str, List[float]]] = {}

        for r in self.results:
            for k, v in r.metrics.items():
                all_metrics.setdefault(k, []).append(v)
                cat_metrics.setdefault(r.category, {}).setdefault(k, []).append(v)

        # 总体统计
        for metric_name, values in all_metrics.items():
            summary["overall"][metric_name] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }

        # 按类别统计
        for cat, metrics_dict in cat_metrics.items():
            cat_summary = {}
            for metric_name, values in metrics_dict.items():
                cat_summary[metric_name] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                }
            summary["by_category"][cat] = cat_summary

        return summary

    # -------------------- 结果保存 --------------------

    def save_results(self, output_dir: str = "results") -> Dict[str, Any]:
        """保存评估结果并生成报告"""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{self.collection_name}_{ts}"

        # 1. 详细结果 JSON
        detail = {
            "metadata": {
                "collection": self.collection_name,
                "persist_dir": self.persist_dir,
                "timestamp": datetime.now().isoformat(),
                "total_queries": len(self.results),
                "top_k_evaluated": self.top_k_values,
            },
            "queries": [
                {
                    "id": r.query_id,
                    "query": r.query,
                    "category": r.category,
                    "metrics": r.metrics,
                    "top_results": [
                        {
                            "source": res.get("metadata", {}).get("source", "unknown"),
                            "score": round(res.get("relevance_score", 0), 4),
                            "content_preview": res.get("content", "")[:200].replace("\n", " "),
                        }
                        for res in r.results[:3]
                    ]
                }
                for r in self.results
            ]
        }

        json_path = out_path / f"exp1_{prefix}_detail.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(detail, f, ensure_ascii=False, indent=2)
        self.log(f"[SAVE] 详细结果: {json_path}")

        # 2. 汇总 JSON
        summary = self.compute_summary()
        sum_path = out_path / f"exp1_{prefix}_summary.json"
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        self.log(f"[SAVE] 汇总结果: {sum_path}")

        # 3. Markdown 报告
        self._generate_markdown(summary, detail, out_path / f"exp1_{prefix}_report.md")

        # 4. 可视化图表
        if _HAS_VIZ:
            self._generate_charts(summary, out_path / f"exp1_{prefix}_charts.png")

        return summary

    def _generate_markdown(self, summary: Dict, detail: Dict, path: Path):
        """生成 Markdown 报告"""
        lines = [
            "# 实验一：RAG 知识库检索质量评估报告",
            "",
            f"- **知识库 Collection**: `{self.collection_name}`",
            f"- **测试时间**: {summary.get('timestamp', 'N/A')}",
            f"- **查询总数**: {summary.get('total_queries', 0)}",
            "",
            "## 一、总体指标汇总",
            "",
            "| 指标 | 均值 | 中位数 | 标准差 | 最小值 | 最大值 |",
            "|------|------|--------|--------|--------|--------|",
        ]

        overall = summary.get("overall", {})
        for metric_name, stats in sorted(overall.items()):
            lines.append(
                f"| {metric_name} | {stats['mean']:.3f} | {stats['median']:.3f} | "
                f"{stats['std']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} |"
            )

        lines.extend(["", "## 二、按 OS 概念类别统计", ""])
        by_cat = summary.get("by_category", {})
        for cat, metrics in sorted(by_cat.items()):
            lines.append(f"### {cat}")
            lines.append("| 指标 | 均值 | 中位数 |")
            lines.append("|------|------|--------|")
            for metric_name, stats in sorted(metrics.items()):
                lines.append(
                    f"| {metric_name} | {stats['mean']:.3f} | {stats['median']:.3f} |"
                )
            lines.append("")

        lines.extend(["", "## 三、逐查询详细结果", ""])
        for q in detail.get("queries", []):
            lines.append(f"### [{q['id']}] {q['query']} (`{q['category']}`)")
            lines.append("")
            lines.append("**指标**:")
            for k, v in q["metrics"].items():
                lines.append(f"- {k}: `{v:.3f}`")
            lines.append("")
            lines.append("**Top-3 检索结果**:")
            for i, res in enumerate(q.get("top_results", []), 1):
                lines.append(f"{i}. `{res['source']}` (score={res['score']})")
                preview = res['content_preview'][:120]
                lines.append(f"   > {preview}...")
            lines.append("")
            lines.append("---")
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        self.log(f"[SAVE] Markdown 报告: {path}")

    def _generate_charts(self, summary: Dict, path: Path):
        """生成可视化图表"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"RAG Retrieval Quality — {self.collection_name}",
                     fontsize=14, fontweight='bold')

        # 1. Recall@K 对比
        ax1 = axes[0, 0]
        k_vals = [k for k in [1, 3, 5] if f"recall@{k}" in summary["overall"]]
        recall_means = [summary["overall"][f"recall@{k}"]["mean"] for k in k_vals]
        src_recall_means = [summary["overall"].get(f"source_recall@{k}", {}).get("mean", 0)
                            for k in k_vals]

        x = np.arange(len(k_vals))
        w = 0.35
        ax1.bar(x - w / 2, recall_means, w, label='Keyword Recall', color='steelblue')
        ax1.bar(x + w / 2, src_recall_means, w, label='Source Recall', color='coral')
        ax1.set_xlabel('K')
        ax1.set_ylabel('Recall')
        ax1.set_title('Recall@K')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'@{k}' for k in k_vals])
        ax1.legend()
        ax1.set_ylim(0, 1.1)

        # 2. 各类别 MRR
        ax2 = axes[0, 1]
        categories = []
        mrr_vals = []
        for cat, metrics in sorted(summary.get("by_category", {}).items()):
            if "mrr" in metrics:
                categories.append(cat)
                mrr_vals.append(metrics["mrr"]["mean"])

        if categories:
            colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
            bars = ax2.barh(categories, mrr_vals, color=colors)
            ax2.set_xlabel('MRR')
            ax2.set_title('MRR by Category')
            ax2.set_xlim(0, 1.1)
            for bar, val in zip(bars, mrr_vals):
                ax2.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                         f'{val:.2f}', va='center', fontsize=9)

        # 3. NDCG@5 分布
        ax3 = axes[1, 0]
        ndcg_vals = [r.metrics.get("ndcg@5", 0) for r in self.results]
        ax3.hist(ndcg_vals, bins=10, color='teal', edgecolor='black', alpha=0.7)
        ax3.set_xlabel('NDCG@5')
        ax3.set_ylabel('Count')
        ax3.set_title('NDCG@5 Distribution')
        if ndcg_vals:
            ax3.axvline(statistics.mean(ndcg_vals), color='red', linestyle='--',
                        label=f'Mean: {statistics.mean(ndcg_vals):.3f}')
        ax3.legend()

        # 4. 相关性分数分布
        ax4 = axes[1, 1]
        avg_scores = [r.metrics.get("avg_score", 0) for r in self.results]
        ax4.hist(avg_scores, bins=10, color='purple', edgecolor='black', alpha=0.7)
        ax4.set_xlabel('Avg Relevance Score')
        ax4.set_ylabel('Count')
        ax4.set_title('Relevance Score Distribution')
        if avg_scores:
            ax4.axvline(statistics.mean(avg_scores), color='red', linestyle='--',
                        label=f'Mean: {statistics.mean(avg_scores):.3f}')
        ax4.legend()

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        self.log(f"[SAVE] 图表: {path}")


# ==================== 入口 ====================

def main():
    parser = argparse.ArgumentParser(description="实验一：RAG 检索质量评估")
    parser.add_argument("--collection", default="rcore_2025s",
                        choices=["rcore_2025s", "ucore_2025s"],
                        help="知识库 collection 名称")
    parser.add_argument("--persist-dir", default="/home/gsk/chroma",
                        help="ChromaDB 持久化目录")
    parser.add_argument("--output", default="results",
                        help="结果输出目录")
    parser.add_argument("--queries", default=None,
                        help="自定义查询集 JSON 文件路径（可选）")
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 3, 5],
                        help="评估的 K 值，如：--top-k 1 3 5 10")

    args = parser.parse_args()

    # 加载自定义查询集
    queries = None
    if args.queries and os.path.exists(args.queries):
        with open(args.queries, "r", encoding="utf-8") as f:
            data = json.load(f)
            queries = [TestQuery(**q) for q in data]
        print(f"[INFO] 已加载自定义查询集: {len(queries)} 个查询")

    evaluator = RAGRetrievalEvaluator(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        top_k_values=args.top_k
    )

    evaluator.evaluate_all(queries)
    summary = evaluator.save_results(args.output)

    # 终端打印关键指标
    print("\n" + "=" * 55)
    print("实验一关键指标汇总")
    print("=" * 55)
    overall = summary.get("overall", {})
    key_metrics = ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5", "source_diversity"]
    for metric in key_metrics:
        if metric in overall:
            s = overall[metric]
            print(f"  {metric:20s}: mean={s['mean']:.3f}  median={s['median']:.3f}  "
                  f"std={s['std']:.3f}")
    print("=" * 55)
    print(f"完整结果保存于: {Path(args.output).absolute()}/")


if __name__ == "__main__":
    main()