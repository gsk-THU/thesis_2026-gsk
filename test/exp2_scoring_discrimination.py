#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验二 & 三：多LLM评分区分度、一致性与重评机制有效性（修复版 v2）

关键修复：
  1. 多重评分路径：直接导入 backend.main 评分函数，绕过 server.py
  2. 强制评分模式：--force-grading 动态尝试多种导入方式
  3. 评分数据诊断：打印 qa_pairs 结构和评分结果结构
  4. 评分失败样本保留：失败样本标记为 grading_failed，仍参与分析

运行方式:
    cd /home/gsk/thesis_2026-gsk/test
    python exp2_scoring_discrimination.py --topics T1 --per-level 1 --force-grading
"""

import argparse
import asyncio
import json
import math
import os
import sys
import time
import traceback
import uuid
import statistics
import re
import importlib.util
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ==================== 路径配置 ====================
_ROOT = "/home/gsk/thesis_2026-gsk"
_PATHS_TO_TRY = [
    f"{_ROOT}/api",
    f"{_ROOT}/llm-council/backend",
    f"{_ROOT}/questions",
    f"{_ROOT}/api/llm-council/backend",
    os.path.expanduser("~/thesis_2026-gsk/api"),
    os.path.expanduser("~/thesis_2026-gsk/llm-council/backend"),
    os.path.expanduser("~/thesis_2026-gsk/questions"),
]
for p in _PATHS_TO_TRY:
    if p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)
        print(f"[PATH] Added: {p}")

# ==================== 评分模块导入（多重路径尝试）====================
_GRADING_AVAILABLE = False
_run_council = None
_chairman_assess = None
_import_errors = []

def _try_import_module(module_name, paths):
    """尝试从多个路径导入模块"""
    for p in paths:
        init_file = os.path.join(p, "__init__.py")
        if not os.path.exists(init_file):
            # 创建空 __init__.py 使目录成为包
            try:
                Path(init_file).touch()
            except:
                pass
        try:
            if p not in sys.path:
                sys.path.insert(0, p)
            module = importlib.import_module(module_name)
            return module
        except Exception as e:
            _import_errors.append(f"{module_name} from {p}: {type(e).__name__}: {e}")
    return None

# 尝试1: 直接导入 backend.main 的评分函数
print("\n[IMPORT] 尝试导入评分模块...")
try:
    from backend.main import run_grading_council, stage1_teacher_scoring, stage2_peer_review
    from backend.main import stage3_chairman_final_grade, calculate_scoring_consensus
    from backend.kimi import query_model
    from backend.config import CHAIRMAN_MODEL
    _run_council = run_grading_council
    _GRADING_AVAILABLE = True
    print("[IMPORT] ✓ backend.main 评分函数导入成功")
except Exception as e:
    _import_errors.append(f"backend.main direct: {e}")
    print(f"[IMPORT] ✗ backend.main 直接导入失败: {e}")

# 尝试2: 从 server.py 导入
if not _GRADING_AVAILABLE:
    try:
        from server import run_council_on_qa_pairs, chairman_overall_assessment
        _run_council = run_council_on_qa_pairs
        _chairman_assess = chairman_overall_assessment
        _GRADING_AVAILABLE = True
        print("[IMPORT] ✓ server.py 评分函数导入成功")
    except Exception as e:
        _import_errors.append(f"server.py: {e}")
        print(f"[IMPORT] ✗ server.py 导入失败: {e}")

# 尝试3: 动态加载 backend/main.py
if not _GRADING_AVAILABLE:
    backend_main_path = os.path.expanduser("/home/gsk/thesis_2026-gsk/llm-council/backend/main.py")
    if os.path.exists(backend_main_path):
        try:
            spec = importlib.util.spec_from_file_location("backend_main", backend_main_path)
            backend_main = importlib.util.module_from_spec(spec)
            # 需要确保 backend 包在路径中
            backend_pkg = os.path.dirname(backend_main_path)
            if backend_pkg not in sys.path:
                sys.path.insert(0, backend_pkg)
            spec.loader.exec_module(backend_main)
            _run_council = getattr(backend_main, 'run_grading_council', None)
            if _run_council:
                _GRADING_AVAILABLE = True
                print("[IMPORT] ✓ 动态加载 backend/main.py 成功")
        except Exception as e:
            _import_errors.append(f"dynamic backend/main.py: {e}")
            print(f"[IMPORT] ✗ 动态加载失败: {e}")

if not _GRADING_AVAILABLE:
    print("\n[IMPORT] 所有评分导入方式均失败，详细错误:")
    for err in _import_errors:
        print(f"  - {err}")
    print("[IMPORT] 评分功能将被禁用。使用 --force-grading 强制尝试。")

# ==================== LLM 模块导入 ====================
_MessageClass = None
_AgentResponseClass = None
_create_kimi = None

try:
    from question_proposer import create_kimi_callback, AgentResponse
    _create_kimi = create_kimi_callback
    _AgentResponseClass = AgentResponse
    print("[IMPORT] ✓ question_proposer 导入成功")
except Exception as e:
    print(f"[IMPORT] ✗ question_proposer 导入失败: {e}")

try:
    from question_proposer import Message as QPMessage
    _MessageClass = QPMessage
    print(f"[IMPORT] ✓ Message class: {QPMessage}")
except Exception:
    @dataclass
    class FallbackMessage:
        role: str
        content: str
        def to_dict(self):
            return {"role": self.role, "content": self.content}
    _MessageClass = FallbackMessage
    print("[IMPORT] ⚠ 使用 FallbackMessage")

try:
    from question_proposer import get_questions
    _get_questions = get_questions
    print("[IMPORT] ✓ get_questions 导入成功")
except Exception as e:
    _get_questions = None
    print(f"[IMPORT] ✗ get_questions 导入失败: {e}")

# OpenAI fallback
try:
    import openai
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

# 可视化
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    from scipy import stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ==================== 数据模型 ====================

@dataclass
class Topic:
    id: str
    module: str
    question: str
    expert_answer: str

@dataclass
class SimulatedSample:
    sample_id: str
    topic_id: str
    level: str
    original_question: str
    initial_answer: str
    exam_questions: List[Dict[str, str]] = field(default_factory=list)
    exam_answers: Dict[str, str] = field(default_factory=dict)
    scores: List[Dict[str, Any]] = field(default_factory=list)
    overall: Dict[str, Any] = field(default_factory=dict)
    latency: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    grading_error: Optional[str] = None  # 评分失败原因


# ==================== 实验常量 ====================

LEVELS = ["A", "B", "C", "D"]
LEVEL_NUM = {"A": 4, "B": 3, "C": 2, "D": 1}

LEVEL_DESC = {
    "A": "优秀。该学生真正完成过实验，能较准确解释 OS 机制和代码实现，能说明关键边界情况。",
    "B": "良好。该学生基本完成实验，理解主要流程，但对少数底层细节、异常路径或设计取舍解释不完整。",
    "C": "及格。该学生看过实验材料，也能说出一些关键词，但回答偏记忆化，缺少对代码和底层机制的深入解释。",
    "D": "不及格。该学生没有真正理解实验，只能给出空泛描述，可能混淆概念，无法解释关键机制。",
}

TOPICS = [
    Topic("T1", "进程管理",
          "请说明你在 rCore 中实现任务分时轮转调度的核心代码逻辑，并解释任务切换时上下文保存与恢复的过程。",
          "在rCore中，分时轮转调度通过时间片轮转实现。每次时钟中断时递减时间片，归零时触发调度。调度器从就绪队列选取下一个任务，通过__switch汇编函数切换上下文。上下文保存包括通用寄存器、程序计数器和栈指针，保存在任务的TrapContext中。恢复时从目标任务的TrapContext加载这些寄存器。"),
    Topic("T2", "系统调用",
          "请说明你如何在内核中添加一个新的系统调用，包括用户态接口、系统调用号分发和内核处理函数。",
          "添加系统调用三步：1) 用户库定义接口函数，使用ecall触发陷阱；2) 在syscall.h分配系统调用号；3) 内核syscall分发表注册处理函数，根据a7寄存器跳转。处理函数从a0-a5获取参数，执行内核逻辑后返回值写入a0。"),
    Topic("T3", "内存管理",
          "请解释多级页表映射的建立过程，并说明页表项有效位、物理页号和访问权限位的作用。",
          "多级页表通过页目录项指向下一级页表，最终到达页表项。建立过程：分配根页表，对虚拟地址的各级索引查找或创建中间页表，最终设置页表项。有效位标记该映射是否可用；物理页号存储实际物理地址的高位；权限位控制读/写/执行/用户态访问。"),
    Topic("T4", "中断与异常",
          "请说明 trap 处理流程中用户态到内核态切换的关键步骤，以及 trap 上下文为何需要保存。",
          "Trap发生时：1) 硬件保存部分寄存器到内核栈；2) 切换到内核地址空间；3) 查询stvec寄存器跳转到trap_handler。trap_handler保存完整上下文（所有通用寄存器、sepc等）到内核栈或任务结构。保存上下文是为了在中断/异常处理完成后能精确恢复到用户态执行点，保证透明性。"),
    Topic("T5", "文件系统",
          "请解释 inode、文件描述符和文件读写接口之间的关系，并说明一次 read 系统调用的大致路径。",
          "inode是文件的元数据和数据块索引；文件描述符是进程打开文件表的索引，指向内核file结构；file结构关联inode和当前读写偏移。read系统调用路径：fd查进程表得file，file查inode，inode通过直接/间接块定位数据，拷贝到用户缓冲区并更新偏移。"),
    Topic("T6", "同步互斥",
          "请说明信号量或互斥锁在内核并发控制中的作用，并分析死锁可能出现的条件。",
          "信号量通过PV操作实现资源计数和等待唤醒；互斥锁保证临界区互斥访问。内核中用于保护共享数据结构（如进程队列、页表）。死锁四条件：互斥、持有并等待、不可抢占、循环等待。内核中可通过资源有序分配、超时释放、避免嵌套锁来预防。"),
    Topic("T7", "页面置换",
          "请比较 FIFO、LRU 和 CLOCK 页面置换算法，并说明你会如何在教学内核中实现 CLOCK。",
          "FIFO简单但可能淘汰常用页；LRU最优但开销大需精确记录访问时间；CLOCK是LRU近似，用访问位和环形链表，淘汰时扫描找访问位为0的页。教学内核实现：页表项保留访问位，缺页时若页框满则遍历页框队列，清访问位直到找到0，置换该页。"),
    Topic("T8", "地址空间",
          "请解释地址空间切换时页表根寄存器变化的意义，以及为什么不同进程需要隔离的虚拟地址空间。",
          "页表根寄存器（如satp）指向当前进程页表物理地址。切换时更新该寄存器并刷新TLB，使CPU使用新进程的地址映射。不同进程需要隔离虚拟地址空间以防止相互读写内存、保证独立运行，同时让每个进程拥有从0开始的统一地址布局，简化编程模型。"),
]


# ==================== Prompt 构建 ====================

def build_initial_prompt(topic: Topic, level: str) -> str:
    return f"""你正在模拟一名操作系统课程学生。请根据指定水平，回答下面的实验口试题。

【原始题目】
{topic.question}

【学生水平】
{LEVEL_DESC[level]}

【生成要求】
1. 只输出学生答案本身，不要输出评分、解释或分析。
2. 答案必须像真实学生口头或书面回答，不要过于完美。
3. 答案长度控制在 150-350 字。
4. 如果是优秀水平，需要体现机制理解、代码关联、边界情况或设计取舍。
5. 如果是良好水平，需要核心正确但保留少量遗漏。
6. 如果是及格水平，需要有一些关键词，但解释浅、细节不足。
7. 如果是不及格水平，需要出现明显概念混淆、空泛套话或错误理解。
8. 不要直接说明"我是优秀/及格学生"。

请生成答案："""


def build_exam_answer_prompt(topic: Topic, initial_answer: str, exam_question: str, level: str) -> str:
    return f"""你正在模拟一名操作系统课程学生参加 AI 文字口试。请根据指定水平回答考官问题。

【原始实验题目】
{topic.question}

【学生初始答案】
{initial_answer}

【考官追问】
{exam_question}

【学生水平】
{LEVEL_DESC[level]}

【回答要求】
1. 只输出学生对该追问的回答，不要输出分析、评分或标题。
2. 答案应与学生水平一致，不能突然变得比初始答案强很多。
3. 优秀水平应具体解释机制，并尽量联系代码或实现细节。
4. 良好水平应回答主要点，但可以遗漏边界情况。
5. 及格水平应有部分正确关键词，但解释浅，可能缺少因果关系。
6. 不及格水平可以答非所问、概念混淆或明显空泛。
7. 答案长度控制在 80-250 字。

请生成回答："""


# ==================== 核心实验类 ====================

class GradingExperiment:
    def __init__(self, output_dir: str, model: str = "kimi-k2.5",
                 temperature: float = 0.7, cache_enabled: bool = True,
                 force_grading: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.temperature = temperature
        self.cache_enabled = cache_enabled
        self.force_grading = force_grading

        self._init_llm_client()
        self.samples: List[SimulatedSample] = []

        # 如果强制评分，再次尝试导入
        if force_grading and not _GRADING_AVAILABLE:
            self._force_import_grading()

    def _force_import_grading(self):
        """强制尝试导入评分模块"""
        global _GRADING_AVAILABLE, _run_council, _chairman_assess
        print("[FORCE] 尝试强制导入评分模块...")

        # 尝试直接从文件加载
        possible_paths = [
            "/home/gsk/thesis_2026-gsk/llm-council/backend/main.py",
            os.path.expanduser("~/thesis_2026-gsk/llm-council/backend/main.py"),
            "/home/gsk/thesis_2026-gsk/api/llm-council/backend/main.py",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    spec = importlib.util.spec_from_file_location("backend_main_forced", path)
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules["backend_main_forced"] = mod
                    spec.loader.exec_module(mod)
                    _run_council = getattr(mod, 'run_grading_council', None)
                    if _run_council:
                        _GRADING_AVAILABLE = True
                        print(f"[FORCE] ✓ 从 {path} 成功加载 run_grading_council")
                        return
                except Exception as e:
                    print(f"[FORCE] ✗ {path}: {e}")

    def _init_llm_client(self):
        self._callback = None
        self._openai_client = None
        self._api_key = os.getenv("MOONSHOT_API_KEY")
        self._base_url = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")

        if _create_kimi and _MessageClass:
            try:
                self._callback = _create_kimi(
                    api_key=self._api_key,
                    base_url=self._base_url,
                    model=self.model,
                    temperature=self.temperature
                )
                print(f"[INIT] LLM 客户端: create_kimi_callback")
                return
            except Exception as e:
                print(f"[WARN] create_kimi_callback 失败: {e}")

        if _HAS_OPENAI and self._api_key:
            try:
                self._openai_client = openai.OpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url
                )
                print(f"[INIT] LLM 客户端: OpenAI 直连")
            except Exception as e:
                print(f"[FATAL] OpenAI 失败: {e}")
                sys.exit(1)
        else:
            print("[FATAL] 无可用 LLM 客户端")
            sys.exit(1)

    def _cache_path(self, key: str) -> Path:
        safe = re.sub(r'[^\w\-]', '_', key)[:100]
        return self.cache_dir / f"{safe}.txt"

    def _load_cache(self, key: str) -> Optional[str]:
        if not self.cache_enabled:
            return None
        p = self._cache_path(key)
        return p.read_text(encoding="utf-8") if p.exists() else None

    def _save_cache(self, key: str, content: str):
        if not self.cache_enabled:
            return
        self._cache_path(key).write_text(content, encoding="utf-8")

    async def _llm_generate(self, prompt: str, cache_key: Optional[str] = None) -> Tuple[str, float]:
        cached = self._load_cache(cache_key) if cache_key else None
        if cached is not None:
            return cached, 0.0

        t0 = time.perf_counter()
        text = "[生成失败]"

        try:
            if self._callback and _MessageClass:
                messages = [_MessageClass(role="user", content=prompt)]
                def _call():
                    return self._callback(messages)
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(None, _call)

                if hasattr(resp, 'content'):
                    text = resp.content.strip()
                elif isinstance(resp, dict):
                    text = resp.get('content', resp.get('text', str(resp))).strip()
                elif isinstance(resp, str):
                    text = resp.strip()
                else:
                    text = str(resp).strip()

            elif self._openai_client:
                def _call():
                    return self._openai_client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                    )
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(None, _call)
                text = resp.choices[0].message.content.strip()
            else:
                raise RuntimeError("无可用 LLM 客户端")

        except Exception as e:
            print(f"    [ERROR] LLM 生成失败: {e}")
            print(f"    [TRACE] {traceback.format_exc()[:800]}")
            text = "[生成失败]"

        latency = (time.perf_counter() - t0) * 1000
        if text != "[生成失败]" and cache_key:
            self._save_cache(cache_key, text)
        return text, latency

    def generate_exam_questions(self, topic: Topic, initial_answer: str) -> Tuple[List[str], float]:
        if _get_questions is None:
            print("    [ERROR] get_questions 不可用")
            return [], 0.0

        t0 = time.perf_counter()
        try:
            agent_resp = _get_questions(topic.question, initial_answer, use_kimi=True)
            questions = self._parse_questions(agent_resp)
        except Exception as e:
            print(f"    [ERROR] 深度测试问题生成失败: {e}")
            traceback.print_exc()
            questions = []
        latency = (time.perf_counter() - t0) * 1000
        return questions, latency

    @staticmethod
    def _parse_questions(agent_resp) -> List[str]:
        content = agent_resp.content if hasattr(agent_resp, 'content') else str(agent_resp)
        lines = content.strip().split('\n')
        questions = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if any(line.startswith(p) for p in [str(i) for i in range(1, 10)]):
                for prefix in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.',
                               '1、', '2、', '3、', '4、', '5、', '6、', '7、', '8、', '9、',
                               '- ', '• ']:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                        break
                if '?' in line or '？' in line or len(line) > 10:
                    questions.append(line)
            elif ('?' in line or '？' in line) and len(line) > 15:
                questions.append(line)
        seen = set()
        uniq = []
        for q in questions:
            if q not in seen:
                seen.add(q)
                uniq.append(q)
        return uniq[:5]

    async def generate_exam_answer(self, topic: Topic, initial_answer: str,
                                    question: str, level: str, qidx: int) -> Tuple[str, float]:
        prompt = build_exam_answer_prompt(topic, initial_answer, question, level)
        key = f"ans_{topic.id}_{level}_{qidx}_{abs(hash(question)) % 9999}"
        return await self._llm_generate(prompt, key)

    # ==================== 评分（修复核心）====================
    async def score_sample(self, sample: SimulatedSample):
        """评分样本，增强诊断和错误处理"""
        global _GRADING_AVAILABLE, _run_council

        if not _GRADING_AVAILABLE:
            sample.grading_error = "评分模块未导入"
            print("    [SKIP] 评分模块不可用")
            return

        # 构建 qa_pairs
        qa_pairs = []
        for q in sample.exam_questions:
            txt = q.get("text", "")
            ans = sample.exam_answers.get(txt, "")
            if ans and ans != "[生成失败]" and len(ans) > 10:
                qa_pairs.append({"text": txt, "answer": ans})

        if not qa_pairs:
            sample.grading_error = "无有效答案"
            print(f"    [SKIP] 无有效 QA pairs")
            return

        print(f"    [GRADE] 准备评分: {len(qa_pairs)} 个 QA pairs")
        for i, qa in enumerate(qa_pairs):
            print(f"      Q{i+1}: {qa['text'][:50]}... | A: {qa['answer'][:50]}...")

        # 执行评分
        t0 = time.perf_counter()
        try:
            print(f"    [GRADE] 调用 run_grading_council...")
            details = await _run_council(qa_pairs)
            print(f"    [GRADE] 评分返回: {type(details)} | 长度: {len(details) if details else 0}")

            if not details:
                sample.grading_error = "评分返回空结果"
                print("    [WARN] 评分返回空")
                return

        except Exception as e:
            sample.grading_error = f"评分异常: {type(e).__name__}: {e}"
            print(f"    [ERROR] 评分失败: {e}")
            print(f"    [TRACE] {traceback.format_exc()[:1000]}")
            return

        scoring_ms = (time.perf_counter() - t0) * 1000

        # 解析评分结果（增强兼容性）
        parsed_scores = []
        for i, d in enumerate(details):
            try:
                score_entry = self._parse_score_detail(d)
                parsed_scores.append(score_entry)
                print(f"    [GRADE] Q{i+1} score={score_entry.get('final_score', 'N/A')} "
                      f"grade={score_entry.get('grade', 'N/A')}")
            except Exception as e:
                print(f"    [WARN] 解析评分结果 {i} 失败: {e}")
                print(f"    [DIAG] 原始类型: {type(d)}, 内容: {str(d)[:200]}")

        sample.scores = parsed_scores

        # 主席评估
        if parsed_scores:
            await self._chairman_evaluation(sample, parsed_scores)

        sample.latency["scoring_total_ms"] = scoring_ms
        sample.latency["per_question_ms"] = scoring_ms / len(qa_pairs) if qa_pairs else 0

    def _parse_score_detail(self, detail) -> Dict[str, Any]:
        """兼容解析评分详情对象（支持 dataclass / pydantic / dict）"""
        # 如果是 dict，直接提取
        if isinstance(detail, dict):
            return {
                "question_id": detail.get("question_id", str(uuid.uuid4())),
                "question_text": detail.get("question_text", ""),
                "student_answer": detail.get("student_answer", ""),
                "final_score": float(detail.get("final_score", 0)),
                "grade": detail.get("grade", "Unknown"),
                "confidence": detail.get("confidence", "中"),
                "chairman_feedback": detail.get("chairman_feedback", detail.get("response", "")),
                "teacher_scores": detail.get("teacher_scores", []),
                "consensus_stats": detail.get("consensus_stats", {}),
                "reevaluation": detail.get("reevaluation", {}),
            }

        # 如果是对象，尝试属性访问
        def _get(obj, attr, default=None):
            if hasattr(obj, attr):
                return getattr(obj, attr)
            if hasattr(obj, '__getitem__'):
                try:
                    return obj[attr]
                except:
                    pass
            return default

        return {
            "question_id": _get(detail, 'question_id', str(uuid.uuid4())),
            "question_text": _get(detail, 'question_text', ""),
            "student_answer": _get(detail, 'student_answer', ""),
            "final_score": float(_get(detail, 'final_score', 0) or 0),
            "grade": _get(detail, 'grade', "Unknown"),
            "confidence": _get(detail, 'confidence', "中"),
            "chairman_feedback": _get(detail, 'chairman_feedback') or _get(detail, 'response', ""),
            "teacher_scores": _get(detail, 'teacher_scores', []),
            "consensus_stats": _get(detail, 'consensus_stats', {}),
            "reevaluation": _get(detail, 'reevaluation', {}),
        }

    async def _chairman_evaluation(self, sample: SimulatedSample, parsed_scores: List[Dict]):
        """主席整体评估"""
        global _chairman_assess

        # 构造 exam_results
        exam_results = []
        for s in parsed_scores:
            exam_results.append({
                "question_id": s.get("question_id", ""),
                "question_text": s.get("question_text", ""),
                "student_answer": s.get("student_answer", ""),
                "stage3": {
                    "final_score": s.get("final_score", 0),
                    "grade": s.get("grade", "Unknown"),
                    "response": s.get("chairman_feedback", "")
                }
            })

        t0c = time.perf_counter()
        try:
            if _chairman_assess:
                overall = await _chairman_assess(
                    sample.original_question, sample.initial_answer, exam_results
                )
            else:
                # 如果没有 chairman_overall_assessment，构造简单结果
                scores = [s.get("final_score", 0) for s in parsed_scores]
                avg = sum(scores) / len(scores) if scores else 0
                overall = type('obj', (object,), {
                    'understanding_level': self._score_to_level(avg),
                    'confidence': 0.7,
                    'reasoning': f"平均分 {avg:.2f}",
                    'knowledge_gaps': [],
                    'recommendations': []
                })()

            sample.overall = {
                "understanding_level": getattr(overall, 'understanding_level', ""),
                "confidence": getattr(overall, 'confidence', 0),
                "reasoning": getattr(overall, 'reasoning', ""),
                "knowledge_gaps": getattr(overall, 'knowledge_gaps', []),
                "recommendations": getattr(overall, 'recommendations', []),
            }
            chair_ms = (time.perf_counter() - t0c) * 1000
            sample.latency["chairman_ms"] = chair_ms

        except Exception as e:
            print(f"    [WARN] 主席评估失败: {e}")
            sample.latency["chairman_ms"] = 0

    # ==================== 主流程 ====================
    async def run(self, topics: List[Topic], levels: List[str], per_level: int):
        total = len(topics) * len(levels) * per_level
        print(f"\n{'='*60}")
        print(f"[EXP] 实验启动")
        print(f"  题目: {len(topics)} | 等级: {len(levels)} | 每等级: {per_level} | 总计: {total}")
        print(f"  API Key: {'已设置' if self._api_key else '未设置！'}")
        print(f"  LLM: {'kimi_callback' if self._callback else 'openai_direct'}")
        print(f"  评分模块: {'可用' if _GRADING_AVAILABLE else '不可用'} (force={self.force_grading})")
        print(f"{'='*60}\n")

        done = 0
        for topic in topics:
            for level in levels:
                for i in range(per_level):
                    sid = f"{topic.id}_{level}_{i+1:02d}"
                    print(f"[{done+1}/{total}] 样本 {sid}")
                    try:
                        await self._process_one(topic, level, i, sid)
                        self._save_checkpoint()
                    except Exception as e:
                        print(f"    [FATAL] 样本处理异常: {e}")
                        traceback.print_exc()
                    done += 1

        # 统计评分成功/失败
        graded = sum(1 for s in self.samples if s.scores)
        failed = sum(1 for s in self.samples if s.grading_error)
        print(f"\n[EXP] 完成 | 总样本: {len(self.samples)} | 评分成功: {graded} | 评分失败: {failed}")
        self._analyze_and_save()

    async def _process_one(self, topic: Topic, level: str, idx: int, sid: str):
        # 1. 初始答案
        prompt = build_initial_prompt(topic, level)
        init, lat_init = await self._llm_generate(prompt, f"init_{topic.id}_{level}_{idx}")
        print(f"    [INIT] {len(init)}字 ({lat_init:.0f}ms)")
        if init == "[生成失败]" or len(init) < 20:
            print(f"    [SKIP] 初始答案无效")
            return

        # 2. 深度测试问题
        questions, lat_q = self.generate_exam_questions(topic, init)
        print(f"    [QUES] {len(questions)}个问题 ({lat_q:.0f}ms)")
        if not questions:
            print(f"    [SKIP] 无深度测试问题")
            return

        # 3. 生成回答
        answers = {}
        lat_ans = 0
        for j, q in enumerate(questions):
            ans, la = await self.generate_exam_answer(topic, init, q, level, j)
            answers[q] = ans
            lat_ans += la
            status = "OK" if ans != "[生成失败]" else "FAIL"
            print(f"    [ANS{j+1}] {status} ({la:.0f}ms)")

        # 4. 组装 & 评分
        sample = SimulatedSample(
            sample_id=sid, topic_id=topic.id, level=level,
            original_question=topic.question, initial_answer=init,
            exam_questions=[{"id": f"q{j+1}", "text": q} for j, q in enumerate(questions)],
            exam_answers=answers,
            latency={
                "initial_gen_ms": lat_init,
                "question_gen_ms": lat_q,
                "answer_gen_ms": lat_ans,
                "answer_gen_per_q_ms": lat_ans / len(questions) if questions else 0,
            }
        )

        # 关键：调用评分
        await self.score_sample(sample)

        avg_score = statistics.mean([q.get("final_score", 0) for q in sample.scores]) if sample.scores else 0
        status = "评分成功" if sample.scores else (sample.grading_error or "未知错误")
        print(f"    [RESULT] {status} | 均分:{avg_score:.2f} "
              f"主席:{sample.overall.get('understanding_level', 'N/A')}")
        self.samples.append(sample)

    def _save_checkpoint(self):
        path = self.output_dir / "checkpoint.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for s in self.samples:
                f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")

    # ==================== 分析 & 保存 ====================
    def _analyze_and_save(self):
        if not self.samples:
            print("[WARN] 无样本可分析")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 分离成功评分和失败的样本
        graded_samples = [s for s in self.samples if s.scores]
        failed_samples = [s for s in self.samples if not s.scores]

        print(f"\n[ANALYSIS] 有效评分样本: {len(graded_samples)}/{len(self.samples)}")

        if not graded_samples:
            print("[WARN] 无成功评分样本，跳过指标计算")
            # 仍保存原始数据
            with open(self.output_dir / f"exp2_{ts}_raw.jsonl", "w", encoding="utf-8") as f:
                for s in self.samples:
                    f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
            return

        # 提取数据（仅使用成功评分的样本）
        level_scores = {lvl: [] for lvl in LEVELS}
        topic_level_scores: Dict[str, Dict[str, List[float]]] = {}
        records = []

        for s in graded_samples:
            scores = [q.get("final_score", 0) for q in s.scores]
            avg = statistics.mean(scores) if scores else 0
            level_scores[s.level].append(avg)
            topic_level_scores.setdefault(s.topic_id, {lvl: [] for lvl in LEVELS})
            topic_level_scores[s.topic_id][s.level].append(avg)

            pred = self._score_to_level(avg)
            records.append({
                "sample_id": s.sample_id, "topic_id": s.topic_id,
                "true_level": s.level, "true_num": LEVEL_NUM[s.level],
                "avg_score": avg, "pred_level": pred,
                "overall_confidence": s.overall.get("confidence", 0),
                "question_count": len(s.scores),
            })

        # 描述统计
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

        # 单调性
        mono_ok = 0
        for tid, sc in topic_level_scores.items():
            means = [statistics.mean(sc[l]) if sc[l] else 0 for l in LEVELS]
            if all(means[i] > means[i+1] for i in range(3)):
                mono_ok += 1
        monotonicity = mono_ok / len(topic_level_scores) if topic_level_scores else 0

        # Spearman
        true_nums = [r["true_num"] for r in records]
        avg_scores = [r["avg_score"] for r in records]
        if _HAS_SCIPY and len(records) > 2:
            rho, pval = stats.spearmanr(true_nums, avg_scores)
        else:
            rho, pval = 0.0, 1.0

        # 分类
        correct = sum(1 for r in records if r["pred_level"] == r["true_level"])
        acc = correct / len(records) if records else 0
        adj = sum(1 for r in records if abs(LEVEL_NUM[r["pred_level"]] - r["true_num"]) <= 1)
        adj_acc = adj / len(records) if records else 0

        means = [desc[l]["mean"] for l in LEVELS]
        deltas = {"AB": means[0]-means[1], "BC": means[1]-means[2], "CD": means[2]-means[3]}

        # 一致性与重评
        consistency = {"高": 0, "中": 0, "低": 0}
        reeval_cnt = 0
        total_q = 0
        for s in graded_samples:
            for q in s.scores:
                total_q += 1
                c = q.get("confidence", "中")
                consistency[c] = consistency.get(c, 0) + 1
                rt = q.get("reevaluation", {})
                if isinstance(rt, dict) and rt.get("trigger_report", {}).get("triggered", False):
                    reeval_cnt += 1

        # 延迟
        lat_summary = {}
        for key in ["initial_gen_ms", "question_gen_ms", "answer_gen_ms",
                    "scoring_total_ms", "chairman_ms", "per_question_ms"]:
            vals = [s.latency.get(key, 0) for s in graded_samples if s.latency.get(key, 0) > 0]
            if vals:
                lat_summary[key] = {
                    "mean": statistics.mean(vals), "median": statistics.median(vals),
                    "std": statistics.stdev(vals) if len(vals) > 1 else 0,
                    "min": min(vals), "max": max(vals),
                }

        summary = {
            "timestamp": ts,
            "total_samples": len(self.samples),
            "graded_samples": len(graded_samples),
            "failed_samples": len(failed_samples),
            "total_questions_rated": total_q,
            "descriptive": desc,
            "monotonicity": {"ratio": monotonicity, "monotonic_topics": mono_ok,
                             "total_topics": len(topic_level_scores)},
            "spearman": {"rho": float(rho), "p_value": float(pval)},
            "classification": {"accuracy": acc, "adjacent_accuracy": adj_acc, "deltas": deltas},
            "consistency": {
                "high_ratio": consistency["高"] / total_q if total_q else 0,
                "medium_ratio": consistency["中"] / total_q if total_q else 0,
                "low_ratio": consistency["低"] / total_q if total_q else 0,
            },
            "reevaluation": {"trigger_rate": reeval_cnt / total_q if total_q else 0,
                           "triggered_count": reeval_cnt},
            "latency": lat_summary,
            "failed_details": [{"sample_id": s.sample_id, "error": s.grading_error} for s in failed_samples],
        }

        # 保存
        (self.output_dir / f"exp2_{ts}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with open(self.output_dir / f"exp2_{ts}_detail.jsonl", "w", encoding="utf-8") as f:
            for s in self.samples:
                f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")

        self._md_report(summary, records, ts)
        if _HAS_MPL:
            self._charts(summary, records, level_scores, lat_summary, ts)

        self._print_summary(summary)

    @staticmethod
    def _score_to_level(score: float) -> str:
        if score >= 8.5: return "A"
        if score >= 7.0: return "B"
        if score >= 5.0: return "C"
        return "D"

    def _md_report(self, summary, records, ts):
        lines = [
            "# 实验二：多LLM评分区分度与一致性评估报告",
            "",
            f"- **测试时间**: {summary['timestamp']}",
            f"- **总样本数**: {summary['total_samples']}",
            f"- **评分成功**: {summary['graded_samples']}",
            f"- **评分失败**: {summary['failed_samples']}",
            f"- **总评分题目数**: {summary['total_questions_rated']}",
            "",
            "## 一、描述性统计（按真实等级）",
            "",
            "| 等级 | N | 平均分 | 中位数 | 标准差 | 最小值 | 最大值 |",
            "|------|---:|-------:|-------:|-------:|-------:|-------:|",
        ]
        for lvl in LEVELS:
            d = summary["descriptive"].get(lvl, {})
            lines.append(f"| {lvl} | {d.get('n', 0)} | {d.get('mean', 0):.2f} | "
                        f"{d.get('median', 0):.2f} | {d.get('std', 0):.2f} | "
                        f"{d.get('min', 0):.2f} | {d.get('max', 0):.2f} |")

        lines.extend(["", "## 二、区分度指标", ""])
        lines.append(f"- **单调性**: {summary['monotonicity']['ratio']:.2%}")
        lines.append(f"- **Spearman ρ**: {summary['spearman']['rho']:.3f}")
        lines.append(f"- **严格准确率**: {summary['classification']['accuracy']:.2%}")
        lines.append(f"- **相邻准确率**: {summary['classification']['adjacent_accuracy']:.2%}")
        d = summary["classification"]["deltas"]
        lines.append(f"- **相邻分差**: ΔAB={d['AB']:.2f}, ΔBC={d['BC']:.2f}, ΔCD={d['CD']:.2f}")

        lines.extend(["", "## 三、一致性与重评", ""])
        c = summary["consistency"]
        lines.append(f"- **高一致性**: {c['high_ratio']:.2%}")
        lines.append(f"- **重评触发率**: {summary['reevaluation']['trigger_rate']:.2%}")

        lines.extend(["", "## 四、延迟统计 (ms)", ""])
        for k, v in summary["latency"].items():
            lines.append(f"- **{k}**: mean={v['mean']:.0f}, median={v['median']:.0f}")

        lines.extend(["", "## 五、失败样本", ""])
        for f in summary.get("failed_details", []):
            lines.append(f"- `{f['sample_id']}`: {f['error']}")

        (self.output_dir / f"exp2_{ts}_report.md").write_text("\n".join(lines), encoding="utf-8")

    def _charts(self, summary, records, level_scores, lat_summary, ts):
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("Grading Discrimination & Consistency", fontsize=14, fontweight='bold')

        # 1. 箱线图
        ax = axes[0, 0]
        data = [level_scores[l] for l in LEVELS]
        bp = ax.boxplot(data, labels=LEVELS, patch_artist=True)
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_ylabel("Score")
        ax.set_title("Score by Level")
        ax.set_ylim(0, 11)

        # 2. 均值
        ax = axes[0, 1]
        means = [summary["descriptive"][l]["mean"] for l in LEVELS]
        errs = [summary["descriptive"][l]["std"] for l in LEVELS]
        ax.bar(LEVELS, means, yerr=errs, capsize=5, color=colors, edgecolor='black')
        ax.set_ylabel("Mean Score")
        ax.set_title("Mean ± SD")
        ax.set_ylim(0, 11)

        # 3. 混淆矩阵
        ax = axes[0, 2]
        cm = {t: {p: 0 for p in LEVELS} for t in LEVELS}
        for r in records:
            cm[r["true_level"]][r["pred_level"]] += 1
        mat = [[cm[t][p] for p in LEVELS] for t in LEVELS]
        im = ax.imshow(mat, cmap='Blues')
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(LEVELS); ax.set_yticklabels(LEVELS)
        ax.set_title("Confusion Matrix")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, mat[i][j], ha='center', va='center', fontweight='bold')

        # 4. 题目折线
        ax = axes[1, 0]
        topic_means = {}
        for r in records:
            topic_means.setdefault(r["topic_id"], {lvl: [] for lvl in LEVELS})
            topic_means[r["topic_id"]][r["true_level"]].append(r["avg_score"])
        for tid in sorted(topic_means.keys()):
            means = [statistics.mean(topic_means[tid][l]) if topic_means[tid][l] else 0 for l in LEVELS]
            ax.plot(LEVELS, means, marker='o', label=tid)
        ax.set_ylabel("Score")
        ax.set_title("By Topic")
        ax.legend(fontsize=7)

        # 5. 延迟
        ax = axes[1, 1]
        lat_keys = ["initial_gen_ms", "answer_gen_ms", "scoring_total_ms"]
        lat_data = []
        lat_labels = []
        for k in lat_keys:
            if k in lat_summary:
                vals = [s.latency.get(k, 0) for s in self.samples if s.latency.get(k, 0) > 0]
                lat_data.append(vals)
                lat_labels.append(k.replace("_ms", ""))
        if lat_data:
            ax.boxplot(lat_data, labels=lat_labels)
            ax.set_ylabel("ms")
            ax.set_title("Latency")
            ax.set_yscale('log')

        # 6. 一致性
        ax = axes[1, 2]
        c = summary["consistency"]
        sizes = [c["high_ratio"], c["medium_ratio"], c["low_ratio"]]
        ax.pie(sizes, labels=["High", "Medium", "Low"], autopct='%1.1f%%', startangle=90)
        ax.set_title("Consistency")

        plt.tight_layout()
        plt.savefig(self.output_dir / f"exp2_{ts}_charts.png", dpi=150, bbox_inches='tight')
        plt.close()

    def _print_summary(self, summary):
        print(f"\n{'='*60}")
        print("实验结果")
        print(f"{'='*60}")
        print(f"  评分成功    : {summary['graded_samples']}/{summary['total_samples']}")
        print(f"  单调性      : {summary['monotonicity']['ratio']:.2%}")
        print(f"  Spearman ρ  : {summary['spearman']['rho']:.3f}")
        print(f"  严格准确率  : {summary['classification']['accuracy']:.2%}")
        print(f"  相邻准确率  : {summary['classification']['adjacent_accuracy']:.2%}")
        d = summary["classification"]["deltas"]
        print(f"  相邻分差    : AB={d['AB']:.2f} BC={d['BC']:.2f} CD={d['CD']:.2f}")
        print(f"  重评触发率  : {summary['reevaluation']['trigger_rate']:.2%}")
        if summary.get("failed_details"):
            print(f"  失败样本    : {len(summary['failed_details'])}个")
        print(f"{'='*60}")


# ==================== 入口 ====================

def main():
    parser = argparse.ArgumentParser(description="实验二：评分区分度自动化测试（修复版 v2）")
    parser.add_argument("--output", default="results", help="输出目录")
    parser.add_argument("--topics", nargs="+", default=[t.id for t in TOPICS],
                        choices=[t.id for t in TOPICS])
    parser.add_argument("--per-level", type=int, default=1, help="每等级样本数")
    parser.add_argument("--levels", nargs="+", default=LEVELS, choices=LEVELS)
    parser.add_argument("--model", default="kimi-k2.5")
    parser.add_argument("--temp", type=float, default=1)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--force-grading", action="store_true",
                        help="强制尝试导入评分模块（即使初始导入失败）")
    args = parser.parse_args()

    topics = [t for t in TOPICS if t.id in args.topics]
    exp = GradingExperiment(
        output_dir=args.output,
        model=args.model,
        temperature=args.temp,
        cache_enabled=not args.no_cache,
        force_grading=args.force_grading
    )
    asyncio.run(exp.run(topics, args.levels, args.per_level))


if __name__ == "__main__":
    main()