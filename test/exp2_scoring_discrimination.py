#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验：真实 OS 实验考试流程自动化评测（支持中断恢复）

流程：
  1. 加载 ch3-ch8 实验数据（os.zip / os-feat.zip / chX.txt）
  2. OSQuestionProposer 基于代码差异与实验要求生成深度测试问题
  3. LLM 按 A/B/C/D 四个水平模拟学生作答
  4. llm-council 三阶段委员会对每道题评分
  5. 主席模型给出整体评估
  6. 统计分析区分度、一致性、单调性等指标

运行方式:
    cd /home/gsk/thesis_2026-gsk/test
    python exp3_os_real_exam.py --chapters ch3 ch4 ch5 ch6 ch7 ch8 --per-level 2 --num-questions 5 --course ucore
    
中断恢复:
    直接重新运行相同命令即可，程序会自动检测 checkpoint 并跳过已完成样本
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
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ==================== 路径配置 ====================
_ROOT = "/home/gsk/thesis_2026-gsk"
_PATHS_TO_TRY = [
    f"{_ROOT}/api",
    f"{_ROOT}/questions",
    f"{_ROOT}/llm-council/backend",
    os.path.expanduser("~/thesis_2026-gsk/api"),
    os.path.expanduser("~/thesis_2026-gsk/questions"),
    os.path.expanduser("~/thesis_2026-gsk/llm-council"),
]
for p in _PATHS_TO_TRY:
    if p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)

# ==================== 模块导入 ====================
_import_errors = []

# --- OS Proposer ---
try:
    from os_proposer import (
        CodeExtractor,
        DiffAnalyzer,
        OSExperimentAnalyzer,
        QuestionProposer,
    )
    _HAS_OS_PROPOSER = True
    print("[IMPORT] ✓ os_proposer 导入成功")
except Exception as e:
    _HAS_OS_PROPOSER = False
    _import_errors.append(f"os_proposer: {e}")
    print(f"[IMPORT] ✗ os_proposer 导入失败: {e}")

# --- LLM Council 评分 ---
try:
    from backend.main import run_grading_council
    from backend.kimi import query_model
    from backend.config import CHAIRMAN_MODEL
    _HAS_COUNCIL = True
    print("[IMPORT] ✓ llm-council 评分模块导入成功")
except Exception as e:
    _HAS_COUNCIL = False
    _import_errors.append(f"backend.main: {e}")
    print(f"[IMPORT] ✗ llm-council 导入失败: {e}")

# --- LLM 客户端 ---
try:
    from question_proposer import create_kimi_callback, AgentResponse, Message as QPMessage
    _create_kimi = create_kimi_callback
    _MessageClass = QPMessage
    print("[IMPORT] ✓ question_proposer (LLM client) 导入成功")
except Exception as e:
    _create_kimi = None
    _MessageClass = None
    _import_errors.append(f"question_proposer: {e}")
    print(f"[IMPORT] ✗ question_proposer 导入失败: {e}")

# OpenAI fallback
try:
    import openai
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

# 可视化
try:
    import matplotlib
    matplotlib.use("Agg")
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
class OSExamTopic:
    chapter: str
    requirement: str
    before_code: Dict[str, str]
    after_code: Dict[str, str]
    diff_report: Any
    key_changes: List[str]


@dataclass
class SimulatedSample:
    sample_id: str
    chapter: str
    level: str
    questions: List[Dict[str, Any]] = field(default_factory=list)
    answers: Dict[str, str] = field(default_factory=dict)
    scores: List[Dict[str, Any]] = field(default_factory=list)
    overall: Dict[str, Any] = field(default_factory=dict)
    latency: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    grading_error: Optional[str] = None
    dialogues: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)


# ==================== 实验常量 ====================

LEVELS = ["A", "B", "C", "D"]
LEVEL_NUM = {"A": 4, "B": 3, "C": 2, "D": 1}

LEVEL_DESC = {
    "A": (
        "优秀。该学生真正完成过实验，能较准确解释 OS 机制和代码实现，"
        "能说明关键边界情况，能联系代码细节回答。"
    ),
    "B": (
        "良好。该学生基本完成实验，理解主要流程，但对少数底层细节、"
        "异常路径或设计取舍解释不完整。"
    ),
    "C": (
        "及格。该学生看过实验材料，也能说出一些关键词，但回答偏记忆化，"
        "缺少对代码和底层机制的深入解释。"
    ),
    "D": (
        "不及格。该学生没有真正理解实验，只能给出空泛描述，"
        "可能混淆概念，无法解释关键机制。"
    ),
}


def build_answer_prompt(
    question_text: str,
    code_snippets: List[str],
    level: str,
    chapter: str,
    key_changes: List[str],
) -> str:
    code_block = ""
    if code_snippets:
        code_block = "\n【问题关联代码片段】\n" + "\n---\n".join(code_snippets)

    changes_block = ""
    if key_changes:
        changes_block = "\n【本次实验关键变更】\n- " + "\n- ".join(key_changes[:5])

    return f"""你正在模拟一名操作系统课程学生参加 AI 口试。请根据指定水平回答考官问题。

【实验章节】
{chapter}

【考官问题】
{question_text}{code_block}{changes_block}

【学生水平】
{LEVEL_DESC[level]}

【回答要求】
1. 只输出学生对该问题的回答，不要输出分析、评分或标题。
2. 答案应与学生水平一致，不能突然变得比预期水平强很多。
3. 优秀水平应具体解释机制，并尽量联系代码或实现细节，能指出边界情况。
4. 良好水平应回答主要点，但可以遗漏边界情况或部分细节。
5. 及格水平应有部分正确关键词，但解释浅，可能缺少因果关系，代码细节模糊。
6. 不及格水平可以答非所问、概念混淆、明显空泛或只写套话。
7. 答案长度控制在 80-300 字。

请生成回答："""


def build_follow_up_prompt(
    original_question: str,
    student_answer: str,
    code_snippets: List[str],
    level: str,
    depth: int,
) -> str:
    code_ctx = ""
    if code_snippets:
        code_ctx = "\n【问题关联代码】\n" + "\n---\n".join(code_snippets)
    return f"""你是一位操作系统课程口试考官。你刚才向学生提出了一个问题，学生回答如下。

【原始问题】
{original_question}{code_ctx}

【学生回答】
{student_answer}

【当前追问深度】第 {depth + 1} 轮追问

【要求】
1. 仔细阅读学生回答，找出其中的漏洞、不严谨之处、或可以深入挖掘的点。
2. 提出一个简短的追问（1-2句话），检测学生是否真正理解而非背诵。
3. 可以要求学生：举例说明、解释原因、分析边界情况、联系代码细节、或指出潜在问题。
4. 只输出追问内容本身，不要有任何前缀、标题或分析过程。
5. 追问必须以问号（？或?）结尾。
6. 追问文字控制在 60 字以内。

请生成追问："""


def build_follow_up_answer_prompt(
    follow_up_question: str,
    original_answer: str,
    level: str,
    chapter: str,
) -> str:
    return f"""你正在模拟一名操作系统课程学生参加 AI 口试。考官针对你之前的回答进行了追问。

【实验章节】
{chapter}

【你此前对原始问题的回答】
{original_answer}

【考官追问】
{follow_up_question}

【学生水平】
{LEVEL_DESC[level]}

【回答要求】
1. 只输出你对该追问的回答，不要输出分析、评分或标题。
2. 答案应与学生水平一致，不能突然变得比预期水平强很多。
3. 优秀水平应具体解释机制，并尽量联系代码或实现细节，能指出边界情况。
4. 良好水平应回答主要点，但可以遗漏边界情况或部分细节。
5. 及格水平应有部分正确关键词，但解释浅，可能缺少因果关系，代码细节模糊。
6. 不及格水平可以答非所问、概念混淆、明显空泛或只写套话。
7. 答案长度控制在 50-200 字。

请生成回答："""


# ==================== 核心实验类 ====================

class OSRealExamExperiment:
    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        course: str = "ucore",
        model: str = "kimi-k2.6",
        temperature: float = 1,
        cache_enabled: bool = True,
        num_questions: int = 5,
        resume: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.course = course.lower()
        self.model = model
        self.temperature = temperature
        self.cache_enabled = cache_enabled
        self.num_questions = num_questions
        self.resume = resume

        self.samples: List[SimulatedSample] = []
        self._completed_ids: set = set()
        self._init_llm_client()

        if not _HAS_OS_PROPOSER:
            raise RuntimeError("os_proposer 模块不可用，无法生成 OS 实验问题")
        if not _HAS_COUNCIL:
            raise RuntimeError("llm-council 评分模块不可用")

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
                    temperature=self.temperature,
                )
                print(f"[INIT] LLM 客户端: create_kimi_callback")
                return
            except Exception as e:
                print(f"[WARN] create_kimi_callback 失败: {e}")

        if _HAS_OPENAI and self._api_key:
            try:
                self._openai_client = openai.OpenAI(
                    api_key=self._api_key, base_url=self._base_url
                )
                print(f"[INIT] LLM 客户端: OpenAI 直连")
            except Exception as e:
                print(f"[FATAL] OpenAI 失败: {e}")
                sys.exit(1)
        else:
            print("[FATAL] 无可用 LLM 客户端")
            sys.exit(1)

    # ==================== 中断恢复核心逻辑 ====================

    @property
    def _checkpoint_path(self) -> Path:
        return self.output_dir / "checkpoint.jsonl"

    @property
    def _progress_path(self) -> Path:
        return self.output_dir / "progress.json"

    def _load_checkpoint(self) -> int:
        if not self.resume:
            print("[RESUME] 中断恢复已禁用，将从头开始")
            return 0

        if not self._checkpoint_path.exists():
            print("[RESUME] 未找到 checkpoint，从头开始")
            return 0

        restored = 0
        print(f"[RESUME] 正在加载 checkpoint: {self._checkpoint_path}")

        with open(self._checkpoint_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    sample = self._dict_to_sample(data)
                    self.samples.append(sample)
                    self._completed_ids.add(sample.sample_id)
                    restored += 1
                except Exception as e:
                    print(f"[RESUME] [WARN] 第 {line_num} 行解析失败: {e}")
                    continue

        if self._progress_path.exists():
            try:
                progress = json.loads(self._progress_path.read_text(encoding="utf-8"))
                print(f"[RESUME] 上次运行时间: {progress.get('timestamp', 'unknown')}")
                print(f"[RESUME] 上次完成: {progress.get('completed', 0)}/{progress.get('total', '?')} 样本")
            except Exception:
                pass

        print(f"[RESUME] ✓ 成功恢复 {restored} 个样本")
        return restored

    def _save_checkpoint(self, sample: Optional[SimulatedSample] = None):
        if sample is not None:
            with open(self._checkpoint_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
            self._completed_ids.add(sample.sample_id)

        progress = {
            "timestamp": datetime.now().isoformat(),
            "completed": len(self._completed_ids),
            "total": getattr(self, "_total_planned", 0),
            "completed_ids": sorted(list(self._completed_ids)),
        }
        self._progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _dict_to_sample(data: Dict) -> SimulatedSample:
        return SimulatedSample(
            sample_id=data.get("sample_id", "unknown"),
            chapter=data.get("chapter", ""),
            level=data.get("level", ""),
            questions=data.get("questions", []),
            answers=data.get("answers", {}),
            scores=data.get("scores", []),
            overall=data.get("overall", {}),
            latency=data.get("latency", {}),
            metadata=data.get("metadata", {}),
            grading_error=data.get("grading_error"),
            dialogues=data.get("dialogues", {}),
        )

    def _is_completed(self, sid: str) -> bool:
        return sid in self._completed_ids

    # ---------- 数据加载 ----------

    def load_chapter_data(self, chapter: str) -> OSExamTopic:
        m = re.match(r"ch(\d+)", chapter.lower())
        if not m:
            raise ValueError(f"章节名称格式错误，应为 chX: {chapter}")
        ch_num = int(m.group(1))
        if ch_num < 3:
            raise ValueError(f"章节 {chapter} 无前序章节代码，至少从 ch3 开始")

        data_dir = self.data_dir

        req_path = data_dir / f"{chapter}.txt"
        if not req_path.exists():
            raise FileNotFoundError(f"缺少实验要求文件: {req_path}")
        requirement = req_path.read_text(encoding="utf-8").strip()

        before_zip = data_dir / f"ch{ch_num - 1}.zip"
        after_zip = data_dir / f"{chapter}.zip"
        if not before_zip.exists():
            raise FileNotFoundError(f"缺少修改前代码 (前序章节): {before_zip}")
        if not after_zip.exists():
            raise FileNotFoundError(f"缺少修改后代码 (当前章节): {after_zip}")

        before_bytes = before_zip.read_bytes()
        after_bytes = after_zip.read_bytes()

        before_code = CodeExtractor.extract_from_zip(before_bytes)
        after_code = CodeExtractor.extract_from_zip(after_bytes)

        if not before_code and not after_code:
            raise ValueError(f"{chapter}: 未能从 zip 中提取任何代码文件")

        diff_report = DiffAnalyzer.analyze(before_code, after_code)
        key_changes = OSExperimentAnalyzer.identify_key_changes(diff_report, after_code)

        print(
            f"[DATA] {chapter}: 前序=ch{ch_num-1}.zip, 当前={chapter}.zip, "
            f"修改前{len(before_code)}个文件, 修改后{len(after_code)}个文件, "
            f"新增{len(diff_report.added_files)}个, 删除{len(diff_report.deleted_files)}个, "
            f"修改{len(diff_report.modified_files)}个, 关键变更{len(key_changes)}项"
        )

        return OSExamTopic(
            chapter=chapter,
            requirement=requirement,
            before_code=before_code,
            after_code=after_code,
            diff_report=diff_report,
            key_changes=key_changes,
        )

    # ---------- 问题生成 ----------

    async def generate_questions(self, topic: OSExamTopic) -> List[Dict[str, Any]]:
        proposer = QuestionProposer(os_type=self.course)
        loop = asyncio.get_event_loop()

        question_set = await loop.run_in_executor(
            None,
            lambda: proposer.generate_questions(
                experiment_requirement=topic.requirement,
                before_code=topic.before_code,
                after_code=topic.after_code,
                diff_report=topic.diff_report,
                num_questions=self.num_questions,
            ),
        )

        questions = []
        for q in question_set.questions:
            item = {
                "id": str(getattr(q, "id", uuid.uuid4())),
                "category": getattr(q, "category", "general"),
                "text": getattr(q, "question", str(q)),
                "code_snippets": getattr(q, "code_snippets", []),
                "diff_context": getattr(q, "diff_context", ""),
            }
            questions.append(item)

        print(f"[QUES] {topic.chapter}: 生成 {len(questions)} 个问题")
        for i, q in enumerate(questions, 1):
            print(f"       Q{i}[{q['category']}]: {q['text'][:60]}...")
        return questions

    # ---------- LLM 生成 ----------

    def _cache_path(self, key: str) -> Path:
        safe = re.sub(r"[^\w\-]", "_", key)[:100]
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

                if hasattr(resp, "content"):
                    text = resp.content.strip()
                elif isinstance(resp, dict):
                    text = resp.get("content", resp.get("text", str(resp))).strip()
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

    async def generate_answer(
        self,
        question: Dict[str, Any],
        level: str,
        topic: OSExamTopic,
        qidx: int,
    ) -> Tuple[str, float]:
        prompt = build_answer_prompt(
            question_text=question["text"],
            code_snippets=question.get("code_snippets", []),
            level=level,
            chapter=topic.chapter,
            key_changes=topic.key_changes,
        )
        key = f"ans_{topic.chapter}_{level}_{qidx}_{abs(hash(question['text'])) % 9999}"
        return await self._llm_generate(prompt, key)

    # ---------- 多轮对话模拟（口试式追问） ----------
    async def _simulate_dialogue(
        self,
        question: Dict[str, Any],
        level: str,
        topic: OSExamTopic,
        qidx: int,
        max_depth: int = 2,
    ) -> Tuple[str, List[Dict[str, str]], float]:
        dialogue: List[Dict[str, str]] = []
        student_utterances: List[str] = []
        total_latency = 0.0

        current_question = question["text"]
        code_snippets = question.get("code_snippets", [])

        ans, la = await self.generate_answer(question, level, topic, qidx)
        total_latency += la
        dialogue.append({"role": "考官", "content": current_question})
        dialogue.append({"role": "学生", "content": ans})
        student_utterances.append(ans)

        for depth in range(max_depth):
            fu_prompt = build_follow_up_prompt(
                original_question=current_question,
                student_answer=ans,
                code_snippets=code_snippets,
                level=level,
                depth=depth,
            )
            fu_key = f"fu_q_{topic.chapter}_{level}_{qidx}_d{depth}_{abs(hash(current_question)) % 9999}"
            follow_up, la_fu = await self._llm_generate(fu_prompt, cache_key=fu_key)
            total_latency += la_fu

            if follow_up == "[生成失败]" or not follow_up.strip():
                break
            if "?" not in follow_up and "？" not in follow_up:
                break

            dialogue.append({"role": "考官", "content": follow_up})

            fua_prompt = build_follow_up_answer_prompt(
                follow_up_question=follow_up,
                original_answer=student_utterances[0],
                level=level,
                chapter=topic.chapter,
            )
            fua_key = f"fu_a_{topic.chapter}_{level}_{qidx}_d{depth}_{abs(hash(follow_up)) % 9999}"
            follow_up_ans, la_fua = await self._llm_generate(fua_prompt, cache_key=fua_key)
            total_latency += la_fua

            if follow_up_ans == "[生成失败]":
                break

            dialogue.append({"role": "学生", "content": follow_up_ans})
            student_utterances.append(follow_up_ans)
            ans = follow_up_ans
            current_question = follow_up

        combined = "【初始回答】\n" + student_utterances[0]
        for i, u in enumerate(student_utterances[1:], 1):
            combined += f"\n\n【追问{i}回答】\n{u}"

        self._save_dialogue_to_file(
            sample_id=f"{topic.chapter}_{level}_Q{qidx}",
            question_id=question.get("id", "unknown"),
            chapter=topic.chapter,
            level=level,
            dialogue=dialogue,
            combined_answer=combined,
        )

        return combined, dialogue, total_latency

    # ---------- 评分 ----------

    async def score_sample(self, sample: SimulatedSample, topic: OSExamTopic):
        if not _HAS_COUNCIL:
            sample.grading_error = "llm-council 未导入"
            print("    [SKIP] 评分模块不可用")
            return

        qa_pairs = []
        for q in sample.questions:
            qid = q["id"]
            ans = sample.answers.get(qid, "")
            if ans and ans != "[生成失败]" and len(ans) > 10:
                qa_pairs.append({"text": q["text"], "answer": ans})

        if not qa_pairs:
            sample.grading_error = "无有效答案"
            print(f"    [SKIP] 无有效 QA pairs")
            return

        print(f"    [GRADE] 准备评分: {len(qa_pairs)} 个 QA pairs")

        t0 = time.perf_counter()
        parsed_scores = []
        exam_results = []

        for i, pair in enumerate(qa_pairs):
            try:
                print(f"    [GRADE] Q{i+1}: {pair['text'][:50]}...")
                stage1_results, stage2_results, stage3_result, metadata = await run_grading_council(
                    pair["text"], pair["answer"]
                )

                score_entry = {
                    "question_id": str(uuid.uuid4()),
                    "question_text": pair["text"],
                    "student_answer": pair["answer"],
                    "final_score": float(stage3_result.get("final_score", 0)),
                    "grade": stage3_result.get("grade", "Unknown"),
                    "confidence": stage3_result.get("confidence", "中"),
                    "chairman_feedback": stage3_result.get("response", ""),
                    "teacher_scores": [
                        {"model": r["model"], "score": r.get("score")}
                        for r in stage1_results
                    ],
                    "consensus_stats": metadata.get("consensus_stats", {}),
                    "reevaluation": {
                        "trigger_report": metadata.get("reevaluation_triggers", {}),
                        "results": metadata.get("reevaluation_results", []),
                    },
                }
                parsed_scores.append(score_entry)

                exam_results.append({
                    "question_id": score_entry["question_id"],
                    "question_text": pair["text"],
                    "student_answer": pair["answer"],
                    "stage3": {
                        "final_score": score_entry["final_score"],
                        "grade": score_entry["grade"],
                        "response": score_entry["chairman_feedback"],
                    },
                })

                print(
                    f"    [GRADE] Q{i+1} score={score_entry['final_score']} "
                    f"grade={score_entry['grade']}"
                )

            except Exception as e:
                print(f"    [ERROR] 评分失败 Q{i+1}: {e}")
                print(f"    [TRACE] {traceback.format_exc()[:600]}")

        scoring_ms = (time.perf_counter() - t0) * 1000
        sample.scores = parsed_scores
        sample.latency["scoring_total_ms"] = scoring_ms
        sample.latency["per_question_ms"] = scoring_ms / len(qa_pairs) if qa_pairs else 0

        if parsed_scores:
            await self._chairman_evaluation(sample, exam_results, topic)

    async def _chairman_evaluation(
        self,
        sample: SimulatedSample,
        exam_results: List[Dict],
        topic: OSExamTopic,
    ):
        scores = [e["stage3"]["final_score"] for e in exam_results if e["stage3"]["final_score"] is not None]
        avg_score = sum(scores) / len(scores) if scores else 0

        exam_summary = []
        for i, result in enumerate(exam_results, 1):
            stage3 = result.get("stage3", {})
            exam_summary.append({
                "question_id": f"考官问题{i}",
                "question_text": result.get("question_text", "")[:100],
                "final_score": stage3.get("final_score"),
                "grade": stage3.get("grade"),
                "key_feedback": stage3.get("response", "")[:300],
            })

        assessment_prompt = f"""你是一位资深的教育评估委员会主席。你已经收到了评估委员会对学生在多个深度测试问题上的表现评分。

【背景信息】
实验章节：{topic.chapter}
实验要求摘要：{topic.requirement[:200]}...

【各深度测试问题的评分结果】
{json.dumps(exam_summary, ensure_ascii=False, indent=2)}

评分统计：
- 参与评分的问题数：{len(exam_summary)}
- 分数分布：{scores}
- 平均分：{avg_score:.1f}/10

【你的任务】
作为委员会主席，请综合以上所有考官问题的评分结果，给出以下评估：

1. **整体理解程度判定**：基于学生在深度测试问题上的综合表现，判定其对原始知识点的真实理解水平。
2. **知识漏洞识别**：指出学生在哪些具体概念、原理或应用层面存在不足。
3. **学习建议**：针对发现的问题，给出3-5条具体、可操作的学习建议。
4. **置信度评估**：给出你对以上评估的置信度（0-1之间的小数）。

【输出格式】
请严格按照以下JSON格式输出（不要包含markdown代码块标记）：

{{
  "understanding_level": "你的判定结论",
  "confidence": 0.85,
  "reasoning": "详细的评估理由分析...",
  "knowledge_gaps": ["漏洞1", "漏洞2", "漏洞3"],
  "recommendations": ["建议1", "建议2", "建议3"]
}}

请给出你的专业裁定："""

        t0c = time.perf_counter()
        try:
            messages = [{"role": "user", "content": assessment_prompt}]
            response = await query_model(CHAIRMAN_MODEL, messages)
            content = response.get("content", "") if response else ""
            clean_content = content.replace("```json", "").replace("```", "").strip()
            assessment_data = json.loads(clean_content)

            sample.overall = {
                "understanding_level": assessment_data.get("understanding_level", "评估失败"),
                "confidence": float(assessment_data.get("confidence", 0)),
                "reasoning": assessment_data.get("reasoning", ""),
                "knowledge_gaps": assessment_data.get("knowledge_gaps", []),
                "recommendations": assessment_data.get("recommendations", []),
            }
            sample.latency["chairman_ms"] = (time.perf_counter() - t0c) * 1000
            print(
                f"    [CHAIR] 整体评估: {sample.overall['understanding_level']} "
                f"(置信度 {sample.overall['confidence']:.2f})"
            )
        except Exception as e:
            print(f"    [WARN] 主席评估失败: {e}")
            sample.latency["chairman_ms"] = 0

    # ---------- 主流程 ----------

    async def run(self, chapters: List[str], levels: List[str], per_level: int):
        total = len(chapters) * len(levels) * per_level
        self._total_planned = total
        
        restored = self._load_checkpoint()
        remaining = total - restored
        
        print(f"{'='*60}")
        print(f"[EXP] 真实 OS 实验考试评测启动")
        print(f"  章节: {chapters}")
        print(f"  等级: {levels}")
        print(f"  每等级每章节: {per_level}")
        print(f"  每章节问题数: {self.num_questions}")
        print(f"  总样本数: {total}")
        print(f"  已恢复: {restored} | 待处理: {remaining}")
        print(f"  API Key: {'已设置' if self._api_key else '未设置！'}")
        print(f"  LLM: {'kimi_callback' if self._callback else 'openai_direct'}")
        print(f"  课程: {self.course}")
        print(f"  恢复模式: {'启用' if self.resume else '禁用'}")
        print(f"{'='*60}")

        done = restored
        skipped = 0
        
        for chapter in chapters:
            # [修复] 章节级预检查：如果该章节所有样本都已完成，直接跳过整个章节
            chapter_total = len(levels) * per_level
            chapter_completed = sum(
                1 for lv in levels for idx in range(per_level)
                if self._is_completed(f"{chapter}_{lv}_{idx+1:02d}")
            )
            if chapter_completed == chapter_total:
                print(f"[CHAPTER] ====== {chapter} ====== [全部 {chapter_total} 个样本已完成，跳过]")
                done += chapter_total
                continue
            
            print(f"[CHAPTER] ====== {chapter} ======")
            try:
                topic = self.load_chapter_data(chapter)
            except Exception as e:
                print(f"[SKIP] {chapter} 数据加载失败: {e}")
                continue

            questions = await self.generate_questions(topic)
            if not questions:
                print(f"[SKIP] {chapter} 未能生成问题")
                continue

            for level in levels:
                for i in range(per_level):
                    sid = f"{chapter}_{level}_{i+1:02d}"
                    
                    if self._is_completed(sid):
                        print(f"[{done+1}/{total}] 样本 {sid} [已存在，跳过]")
                        skipped += 1
                        done += 1
                        continue
                    
                    print(f"[{done+1}/{total}] 样本 {sid}")
                    try:
                        await self._process_one(topic, questions, level, i, sid)
                        done += 1
                    except Exception as e:
                        print(f"    [FATAL] 样本处理异常: {e}")
                        traceback.print_exc()
                        failed_sample = SimulatedSample(
                            sample_id=sid,
                            chapter=chapter,
                            level=level,
                            grading_error=str(e),
                        )
                        self.samples.append(failed_sample)
                        self._save_checkpoint(failed_sample)
                        done += 1

        graded = sum(1 for s in self.samples if s.scores)
        failed = sum(1 for s in self.samples if s.grading_error)
        print(f"[EXP] 完成 | 总样本: {len(self.samples)} | 评分成功: {graded} | 评分失败: {failed} | 跳过: {skipped}")
        self._analyze_and_save()

    async def _process_one(
        self,
        topic: OSExamTopic,
        questions: List[Dict],
        level: str,
        idx: int,
        sid: str,
    ):
        t_start = time.perf_counter()

        answers = {}
        dialogues = {}
        lat_ans = 0
        for j, q in enumerate(questions):
            combined, dialogue, la = await self._simulate_dialogue(q, level, topic, j, max_depth=2)
            answers[q["id"]] = combined
            dialogues[q["id"]] = dialogue
            lat_ans += la
            status = "OK" if not combined.startswith("[生成失败]") else "FAIL"
            rounds = len(dialogue) // 2
            print(f"    [DIALOGUE{j+1}] {status} ({la:.0f}ms, {rounds}轮对话)")

        sample = SimulatedSample(
            sample_id=sid,
            chapter=topic.chapter,
            level=level,
            questions=questions,
            answers=answers,
            dialogues=dialogues,
            latency={
                "answer_gen_ms": lat_ans,
                "answer_gen_per_q_ms": lat_ans / len(questions) if questions else 0,
                "total_ms": (time.perf_counter() - t_start) * 1000,
            },
            metadata={
                "requirement": topic.requirement[:200],
                "key_changes": topic.key_changes,
                "num_files_before": len(topic.before_code),
                "num_files_after": len(topic.after_code),
                "dialogues": dialogues,
            },
        )

        await self.score_sample(sample, topic)

        avg_score = statistics.mean([q.get("final_score", 0) for q in sample.scores]) if sample.scores else 0
        status = "评分成功" if sample.scores else (sample.grading_error or "未知错误")
        print(
            f"    [RESULT] {status} | 均分:{avg_score:.2f} "
            f"主席:{sample.overall.get('understanding_level', 'N/A')}"
        )
        self.samples.append(sample)
        self._save_checkpoint(sample)

    # ---------- 分析 & 保存 ----------

    def _analyze_and_save(self):
        if not self.samples:
            print("[WARN] 无样本可分析")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        graded_samples = [s for s in self.samples if s.scores]
        failed_samples = [s for s in self.samples if not s.scores]

        print(f"[ANALYSIS] 有效评分样本: {len(graded_samples)}/{len(self.samples)}")

        if not graded_samples:
            with open(self.output_dir / f"exp3_{ts}_raw.jsonl", "w", encoding="utf-8") as f:
                for s in self.samples:
                    f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
            return

        level_scores = {lvl: [] for lvl in LEVELS}
        chapter_level_scores: Dict[str, Dict[str, List[float]]] = {}
        records = []

        for s in graded_samples:
            scores = [q.get("final_score", 0) for q in s.scores]
            avg = statistics.mean(scores) if scores else 0
            level_scores[s.level].append(avg)
            chapter_level_scores.setdefault(s.chapter, {lvl: [] for lvl in LEVELS})
            chapter_level_scores[s.chapter][s.level].append(avg)

            pred = self._score_to_level(avg)
            records.append({
                "sample_id": s.sample_id,
                "chapter": s.chapter,
                "true_level": s.level,
                "true_num": LEVEL_NUM[s.level],
                "avg_score": avg,
                "pred_level": pred,
                "overall_confidence": s.overall.get("confidence", 0),
                "question_count": len(s.scores),
            })

        desc = {}
        for lvl in LEVELS:
            vals = level_scores[lvl]
            if vals:
                desc[lvl] = {
                    "n": len(vals),
                    "mean": statistics.mean(vals),
                    "median": statistics.median(vals),
                    "std": statistics.stdev(vals) if len(vals) > 1 else 0,
                    "min": min(vals),
                    "max": max(vals),
                }

        mono_ok = 0
        for ch, sc in chapter_level_scores.items():
            means = [statistics.mean(sc[l]) if sc[l] else 0 for l in LEVELS]
            if all(means[i] > means[i + 1] for i in range(3)):
                mono_ok += 1
        monotonicity = mono_ok / len(chapter_level_scores) if chapter_level_scores else 0

        true_nums = [r["true_num"] for r in records]
        avg_scores = [r["avg_score"] for r in records]
        if _HAS_SCIPY and len(records) > 2:
            rho, pval = stats.spearmanr(true_nums, avg_scores)
        else:
            rho, pval = 0.0, 1.0

        correct = sum(1 for r in records if r["pred_level"] == r["true_level"])
        acc = correct / len(records) if records else 0
        adj = sum(1 for r in records if abs(LEVEL_NUM[r["pred_level"]] - r["true_num"]) <= 1)
        adj_acc = adj / len(records) if records else 0

        means = [desc[l]["mean"] for l in LEVELS]
        deltas = {"AB": means[0] - means[1], "BC": means[1] - means[2], "CD": means[2] - means[3]}

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

        lat_summary = {}
        for key in ["answer_gen_ms", "scoring_total_ms", "chairman_ms", "per_question_ms"]:
            vals = [s.latency.get(key, 0) for s in graded_samples if s.latency.get(key, 0) > 0]
            if vals:
                lat_summary[key] = {
                    "mean": statistics.mean(vals),
                    "median": statistics.median(vals),
                    "std": statistics.stdev(vals) if len(vals) > 1 else 0,
                    "min": min(vals),
                    "max": max(vals),
                }

        summary = {
            "timestamp": ts,
            "total_samples": len(self.samples),
            "graded_samples": len(graded_samples),
            "failed_samples": len(failed_samples),
            "total_questions_rated": total_q,
            "descriptive": desc,
            "monotonicity": {
                "ratio": monotonicity,
                "monotonic_chapters": mono_ok,
                "total_chapters": len(chapter_level_scores),
            },
            "spearman": {"rho": float(rho), "p_value": float(pval)},
            "classification": {
                "accuracy": acc,
                "adjacent_accuracy": adj_acc,
                "deltas": deltas,
            },
            "consistency": {
                "high_ratio": consistency["高"] / total_q if total_q else 0,
                "medium_ratio": consistency["中"] / total_q if total_q else 0,
                "low_ratio": consistency["低"] / total_q if total_q else 0,
            },
            "reevaluation": {
                "trigger_rate": reeval_cnt / total_q if total_q else 0,
                "triggered_count": reeval_cnt,
            },
            "latency": lat_summary,
            "failed_details": [
                {"sample_id": s.sample_id, "error": s.grading_error}
                for s in failed_samples
            ],
        }

        (self.output_dir / f"exp3_{ts}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with open(self.output_dir / f"exp3_{ts}_detail.jsonl", "w", encoding="utf-8") as f:
            for s in self.samples:
                f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")   

        self._md_report(summary, records, ts)
        if _HAS_MPL:
            self._charts(summary, records, level_scores, chapter_level_scores, lat_summary, ts)

        self._print_summary(summary)

    @staticmethod
    def _score_to_level(score: float) -> str:
        if score >= 8.5:
            return "A"
        if score >= 7.0:
            return "B"
        if score >= 5.0:
            return "C"
        return "D"

    def _md_report(self, summary, records, ts):
        lines = [
            "# 实验三：真实 OS 实验考试流程自动化评测报告",
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
            lines.append(
                f"| {lvl} | {d.get('n', 0)} | {d.get('mean', 0):.2f} | "
                f"{d.get('median', 0):.2f} | {d.get('std', 0):.2f} | "
                f"{d.get('min', 0):.2f} | {d.get('max', 0):.2f} |"
            )

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

        (self.output_dir / f"exp3_{ts}_report.md").write_text("".join(lines), encoding="utf-8")

    def _charts(self, summary, records, level_scores, chapter_level_scores, lat_summary, ts):
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("OS Real Exam: Grading Discrimination & Consistency", fontsize=14, fontweight="bold")

        ax = axes[0, 0]
        data = [level_scores[l] for l in LEVELS]
        bp = ax.boxplot(data, labels=LEVELS, patch_artist=True)
        colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax.set_ylabel("Score")
        ax.set_title("Score by Level")
        ax.set_ylim(0, 11)

        ax = axes[0, 1]
        means = [summary["descriptive"][l]["mean"] for l in LEVELS]
        errs = [summary["descriptive"][l]["std"] for l in LEVELS]
        ax.bar(LEVELS, means, yerr=errs, capsize=5, color=colors, edgecolor="black")
        ax.set_ylabel("Mean Score")
        ax.set_title("Mean ± SD")
        ax.set_ylim(0, 11)

        ax = axes[0, 2]
        cm = {t: {p: 0 for p in LEVELS} for t in LEVELS}
        for r in records:
            cm[r["true_level"]][r["pred_level"]] += 1
        mat = [[cm[t][p] for p in LEVELS] for t in LEVELS]
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(LEVELS)
        ax.set_yticklabels(LEVELS)
        ax.set_title("Confusion Matrix")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, mat[i][j], ha="center", va="center", fontweight="bold")

        ax = axes[1, 0]
        for ch in sorted(chapter_level_scores.keys()):
            means = [statistics.mean(chapter_level_scores[ch][l]) if chapter_level_scores[ch][l] else 0 for l in LEVELS]
            ax.plot(LEVELS, means, marker="o", label=ch)
        ax.set_ylabel("Score")
        ax.set_title("By Chapter")
        ax.legend(fontsize=7)

        ax = axes[1, 1]
        lat_keys = ["answer_gen_ms", "scoring_total_ms"]
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
            ax.set_yscale("log")

        ax = axes[1, 2]
        c = summary["consistency"]
        sizes = [c["high_ratio"], c["medium_ratio"], c["low_ratio"]]
        ax.pie(sizes, labels=["High", "Medium", "Low"], autopct="%1.1f%%", startangle=90)
        ax.set_title("Consistency")

        plt.tight_layout()
        plt.savefig(self.output_dir / f"exp3_{ts}_charts.png", dpi=150, bbox_inches="tight")
        plt.close()

    def _save_dialogue_to_file(
        self,
        sample_id: str,
        question_id: str,
        chapter: str,
        level: str,
        dialogue: List[Dict[str, str]],
        combined_answer: str,
    ):
        debug_dir = self.output_dir / "dialogues"
        debug_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = debug_dir / f"{chapter}_{level}_{sample_id}_{ts}.txt"

        lines = [
            f"=" * 70,
            f"对话记录 | 章节: {chapter} | 水平: {level} | 样本: {sample_id}",
            f"问题ID: {question_id}",
            f"=" * 70,
            "",
        ]

        for i, turn in enumerate(dialogue, 1):
            role = turn["role"]
            content = turn["content"]
            if role == "考官":
                lines.append(f"【考官 · 第{i}轮】")
                lines.append("-" * 50)
            else:
                lines.append(f"【学生 · 第{i}轮】")
                lines.append("-" * 50)
            lines.append(content)
            lines.append("")

        lines.extend([
            "=" * 70,
            "【合并后的综合答案（用于评分）】",
            "=" * 70,
            "",
            combined_answer,
            "",
            "=" * 70,
        ])

        filename.write_text("\n".join(lines), encoding="utf-8")
        print(f"    [DIALOGUE_SAVE] 已保存对话: {filename}")

    def _print_summary(self, summary):
        print(f"{'='*60}")
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
    parser = argparse.ArgumentParser(description="实验三：真实 OS 实验考试流程自动化评测（支持中断恢复）")
    parser.add_argument(
        "--data-dir",
        default="/home/gsk/thesis_2026-gsk/test/data",
        help="实验数据根目录（含 ch2.zip-ch8.zip 及 ch3.txt-ch8.txt）",
    )
    parser.add_argument("--output", default="results/exp3_os_exam", help="输出目录")
    parser.add_argument(
        "--chapters",
        nargs="+",
        default=[f"ch{i}" for i in range(3, 9)],
        help="要测试的章节文件夹名",
    )
    parser.add_argument(
        "--per-level", type=int, default=2, help="每章节每等级生成样本数"
    )
    parser.add_argument("--levels", nargs="+", default=LEVELS, choices=LEVELS)
    parser.add_argument("--num-questions", type=int, default=5, help="每章节生成问题数")
    parser.add_argument("--course", default="ucore", choices=["ucore", "rcore"])
    parser.add_argument("--model", default="kimi-k2.6")
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--no-resume", 
        action="store_true", 
        help="禁用中断恢复，从头开始运行（默认启用恢复）"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="重置：删除 checkpoint 和进度文件，从头开始"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    
    if args.reset:
        checkpoint = output_dir / "checkpoint.jsonl"
        progress = output_dir / "progress.json"
        if checkpoint.exists():
            checkpoint.unlink()
            print(f"[RESET] 已删除 {checkpoint}")
        if progress.exists():
            progress.unlink()
            print(f"[RESET] 已删除 {progress}")
        print("[RESET] 进度已重置，将从头开始")

    exp = OSRealExamExperiment(
        data_dir=args.data_dir,
        output_dir=args.output,
        course=args.course,
        model=args.model,
        temperature=args.temp,
        cache_enabled=not args.no_cache,
        num_questions=args.num_questions,
        resume=not args.no_resume,
    )
    asyncio.run(exp.run(args.chapters, args.levels, args.per_level))


if __name__ == "__main__":
    main()