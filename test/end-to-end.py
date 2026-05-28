import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==================== 配置 ====================
DEBUG_DIR = "/home/gsk/thesis_2026-gsk/debug/latency"  # 你的本机路径
OUTPUT_DIR = "/home/gsk/thesis_2026-gsk/test/results/end-to-end"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 读取数据函数 ====================
def load_latency_data(debug_dir: str) -> pd.DataFrame:
    """读取所有延迟记录JSON文件"""
    import glob
    files = glob.glob(os.path.join(debug_dir, "latency_*.json"))
    if not files:
        raise FileNotFoundError(f"未找到延迟记录文件: {debug_dir}")
    
    records = []
    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        summary = data.get('summary', {})
        stages = summary.get('stages_ms', {})
        
        records.append({
            'file': os.path.basename(filepath),
            'evaluation_id': summary.get('evaluation_id', ''),
            'end_to_end_ms': summary.get('end_to_end_ms', 0),
            'asr_total_ms': stages.get('asr_total', 0),
            'llm_generate_ms': stages.get('llm_generate', 0),
            'tts_synthesize_ms': stages.get('tts_synthesize', 0),
        })
    
    df = pd.DataFrame(records)
    df['other_ms'] = df['end_to_end_ms'] - (
        df['asr_total_ms'] + df['llm_generate_ms'] + df['tts_synthesize_ms']
    )
    df['other_ms'] = df['other_ms'].clip(lower=0)
    return df

# ==================== 统计分析 ====================
def generate_statistics(df: pd.DataFrame) -> dict:
    """生成描述性统计"""
    e2e = df['end_to_end_ms']
    
    stats = {
        'total_samples': len(df),
        'end_to_end_ms': {
            'mean': round(float(e2e.mean()), 2),
            'median': round(float(e2e.median()), 2),
            'std': round(float(e2e.std()), 2),
            'min': round(float(e2e.min()), 2),
            'max': round(float(e2e.max()), 2),
            'q25': round(float(e2e.quantile(0.25)), 2),
            'q75': round(float(e2e.quantile(0.75)), 2),
        }
    }
    
    stages = [
        ('ASR Recognition', 'asr_total_ms'),
        ('LLM Generation', 'llm_generate_ms'),
        ('TTS Synthesis', 'tts_synthesize_ms'),
        ('Other (Gap)', 'other_ms'),
    ]
    
    stats['stage_breakdown'] = []
    for name, col in stages:
        mean_val = df[col].mean()
        stats['stage_breakdown'].append({
            'stage': name,
            'mean_ms': round(float(mean_val), 2),
            'median_ms': round(float(df[col].median()), 2),
            'std_ms': round(float(df[col].std()), 2),
            'percentage': round(float(mean_val / e2e.mean() * 100), 1)
        })
    
    sorted_stages = sorted(stats['stage_breakdown'], key=lambda x: x['percentage'], reverse=True)
    stats['bottleneck_analysis'] = {
        'primary_bottleneck': f"{sorted_stages[0]['stage']} ({sorted_stages[0]['percentage']}%)",
        'secondary_bottleneck': f"{sorted_stages[1]['stage']} ({sorted_stages[1]['percentage']}%)",
        'combined_percentage': round(sorted_stages[0]['percentage'] + sorted_stages[1]['percentage'], 1)
    }
    
    return stats

# ==================== 可视化函数 ====================
def plot_end_to_end_distribution(df: pd.DataFrame, output_dir: str):
    """图1：端到端延迟分布"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    e2e = df['end_to_end_ms']
    
    axes[0].hist(e2e, bins=max(8, len(df)//3), color='#4C78A8', edgecolor='white', alpha=0.85)
    axes[0].axvline(e2e.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {e2e.mean():.0f}ms')
    axes[0].axvline(e2e.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {e2e.median():.0f}ms')
    axes[0].set_xlabel('End-to-End Latency (ms)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('(a) End-to-End Latency Distribution', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)
    
    bp = axes[1].boxplot(e2e, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='#4C78A8', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2))
    axes[1].set_ylabel('Latency (ms)', fontsize=11)
    axes[1].set_title('(b) End-to-End Latency Boxplot', fontsize=12, fontweight='bold')
    axes[1].set_xticklabels(['Overall'])
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig6_1_end_to_end_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_stage_decomposition(df: pd.DataFrame, output_dir: str):
    """图2：阶段分解"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    stage_cols = ['asr_total_ms', 'llm_generate_ms', 'tts_synthesize_ms', 'other_ms']
    stage_labels = ['ASR\nRecognition', 'LLM\nGeneration', 'TTS\nSynthesis', 'Other\n(Gap)']
    colors = ['#E45756', '#F58518', '#72B7B2', '#Eeca3b']
    
    bottom = np.zeros(len(df))
    for i, (col, label, color) in enumerate(zip(stage_cols, stage_labels, colors)):
        axes[0].bar(range(len(df)), df[col], bottom=bottom, 
                    label=label.replace('\n', ' '), color=color, edgecolor='white', linewidth=0.5)
        bottom += df[col].values
    
    axes[0].set_xlabel('Sample ID', fontsize=11)
    axes[0].set_ylabel('Latency (ms)', fontsize=11)
    axes[0].set_title('(a) Per-Sample Latency Decomposition', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper left', fontsize=9, ncol=2)
    axes[0].grid(axis='y', alpha=0.3)
    
    stage_means = [df[col].mean() for col in stage_cols]
    wedges, texts, autotexts = axes[1].pie(
        stage_means, labels=[l.replace('\n', ' ') for l in stage_labels], colors=colors,
        autopct='%1.1f%%', startangle=90, pctdistance=0.75,
        textprops={'fontsize': 9}
    )
    for autotext in autotexts:
        autotext.set_fontsize(8)
        autotext.set_fontweight('bold')
    
    axes[1].set_title('(b) Average Stage Proportion', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig6_2_stage_decomposition.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_stage_boxplot(df: pd.DataFrame, output_dir: str):
    """图3：阶段箱线图对比"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stage_data = [
        df['asr_total_ms'].values,
        df['llm_generate_ms'].values,
        df['tts_synthesize_ms'].values,
        df['other_ms'].values
    ]
    labels = ['ASR\nRecognition', 'LLM\nGeneration', 'TTS\nSynthesis', 'Other\n(Gap)']
    colors = ['#E45756', '#F58518', '#72B7B2', '#Eeca3b']
    
    bp = ax.boxplot(stage_data, vert=True, patch_artist=True,
                    labels=[l.replace('\n', ' ') for l in labels])
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Latency (ms)', fontsize=11)
    ax.set_title('Stage-Level Latency Comparison (Boxplot)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for i, d in enumerate(stage_data):
        mean_val = np.mean(d)
        ax.scatter(i+1, mean_val, marker='D', color='red', s=50, zorder=5)
        ax.text(i+1.15, mean_val, f'{mean_val:.0f}', fontsize=8, va='center', color='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig6_3_stage_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

# ==================== 主入口 ====================
def main():
    # 读取数据
    df = load_latency_data(DEBUG_DIR)
    
    # 统计
    stats = generate_statistics(df)
    
    # 保存统计JSON
    with open(os.path.join(OUTPUT_DIR, 'latency_statistics.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 生成图表
    plot_end_to_end_distribution(df, OUTPUT_DIR)
    plot_stage_decomposition(df, OUTPUT_DIR)
    plot_stage_boxplot(df, OUTPUT_DIR)
    
    # 打印摘要
    print("=" * 60)
    print("分析完成")
    print("=" * 60)
    e2e = stats['end_to_end_ms']
    print(f"端到端延迟: 均值={e2e['mean']}ms, 中位数={e2e['median']}ms, 标准差={e2e['std']}ms")
    print(f"范围: [{e2e['min']}, {e2e['max']}] ms")
    print(f"\n主要瓶颈: {stats['bottleneck_analysis']['primary_bottleneck']}")
    print(f"次要瓶颈: {stats['bottleneck_analysis']['secondary_bottleneck']}")
    print(f"两者合计: {stats['bottleneck_analysis']['combined_percentage']}%")
    print(f"\n输出文件: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()