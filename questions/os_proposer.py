"""
os_proposer.py - 操作系统课程实验代码提问器
改进：RAG 知识库工具化（Tool-based RAG）+ 问题附带代码片段
"""

import os
import re
import uuid
import zipfile
import tempfile
import difflib
import logging
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

from question_proposer import create_kimi_callback, create_mock_callback, Agent, AgentResponse

# ==================== 用户 KnowledgeBase 导入（保持原有逻辑）====================
_USER_KB_CLASS = None
_USER_KB_PATH = None

for _path in [
    os.path.expanduser("/home/gsk/thesis_2026-gsk/chroma"),
    os.path.expanduser("/home/gsk/thesis_2026-gsk/questions"),
    os.path.expanduser("/home/gsk"),
    os.path.expanduser("/home/gsk/rCore-Tutorial-Guide-2025S"),
    os.path.expanduser("/home/gsk/ucore-tutorial-2025s"),
    os.path.dirname(os.path.abspath(__file__)),
]:
    if not os.path.isdir(_path):
        continue
    if _path not in sys.path:
        sys.path.insert(0, _path)
    try:
        from database import KnowledgeBase as _KB
        _USER_KB_CLASS = _KB
        _USER_KB_PATH = _path
        logger.info(f"[DIAG-OS-KB] Imported user's KnowledgeBase from: {_path}")
        break
    except ImportError as _e:
        logger.debug(f"[DIAG-OS-KB] Failed to import from {_path}: {_e}")
        continue

if _USER_KB_CLASS is None:
    logger.warning("[DIAG-OS-KB] User's KnowledgeBase not found. RAG tools will be disabled.")


# ==================== 数据模型 ====================

@dataclass
class CodeFile:
    path: str
    content: str
    language: str
    line_count: int

    def get_snippet(self, start_line: int, end_line: int) -> str:
        lines = self.content.split("\n")
        return "\n".join(lines[start_line-1:end_line])


@dataclass
class FileDiff:
    file_path: str
    status: str
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    unified_diff: str = ""
    changed_lines_old: List[int] = field(default_factory=list)
    changed_lines_new: List[int] = field(default_factory=list)


@dataclass
class CodeDiffReport:
    added_files: List[str] = field(default_factory=list)
    deleted_files: List[str] = field(default_factory=list)
    modified_files: List[FileDiff] = field(default_factory=list)
    unchanged_files: List[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class ProposedQuestion:
    id: int
    category: str
    question: str
    target_file: Optional[str] = None
    target_lines: Optional[Tuple[int, int]] = None
    rationale: str = ""
    code_snippets: List[str] = field(default_factory=list)  # ← 新增：问题附带的代码片段


@dataclass
class QuestionSet:
    experiment_title: str
    questions: List[ProposedQuestion]
    summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== RAG 工具集 ====================

@dataclass
class ToolCall:
    tool_name: str
    arguments: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool_name, "arguments": self.arguments}


class RAGToolKit:
    COLLECTION_MAP = {
        "rcore": "rcore_2025s",
        "ucore": "ucore_2025s",
    }
    
    def __init__(
        self,
        os_type: str = "rcore",
        persist_dir: str = "/home/gsk/chroma",
        default_top_k: int = 5
    ):
        self.os_type = os_type.lower()
        self.persist_dir = persist_dir
        self.default_top_k = default_top_k
        self.collection_name = self.COLLECTION_MAP.get(
            self.os_type, f"{self.os_type}_2025s"
        )
        self._kb = None
        self._initialized = False
        
        if _USER_KB_CLASS is None:
            logger.warning("[RAG-TOOLS] KnowledgeBase unavailable, all tools will return empty")
            return
            
        try:
            self._kb = _USER_KB_CLASS(self.persist_dir, self.collection_name)
            self._initialized = True
            logger.info(f"[RAG-TOOLS] Initialized for collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"[RAG-TOOLS] Failed to init: {e}")
    
    def search_course_material(self, query: str, n_results: int = 5) -> str:
        if not self._initialized:
            return "[知识库未初始化，无法检索]"
        
        try:
            raw_results = self._kb.query(query, n_results=n_results)
            if not raw_results:
                return f"[未找到与 '{query}' 相关的课程资料]"
            
            lines = [f"\n--- 课程资料检索结果: '{query}' ---"]
            for i, item in enumerate(raw_results, 1):
                content = item.get("content", "")
                source = item.get("metadata", {}).get("source", "unknown")
                score = item.get("relevance_score", 0.0)
                lines.append(f"\n[{i}] 来源: {source} (相关度: {score:.3f})")
                lines.append(content[:600])
                if len(content) > 600:
                    lines.append("... (已截断)")
            lines.append("--- 检索结束 ---\n")
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[RAG-TOOLS] search_course_material failed: {e}")
            return f"[检索出错: {e}]"
    
    def search_os_concept(self, concept: str, detail_level: str = "standard") -> str:
        query = f"{self.os_type} {concept} 原理 实现"
        n_results = {"brief": 2, "standard": 4, "detailed": 6}.get(detail_level, 4)
        return self.search_course_material(query, n_results)
    
    def search_by_lab(self, lab_name: str) -> str:
        query = f"实验 {lab_name} 要求 步骤"
        return self.search_course_material(query, n_results=5)
    
    def search_by_code_symbol(self, symbol_name: str, context: str = "") -> str:
        query = f"{symbol_name} {context}".strip()
        return self.search_course_material(query, n_results=4)
    
    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        tool_map = {
            "search_course_material": self.search_course_material,
            "search_os_concept": self.search_os_concept,
            "search_by_lab": self.search_by_lab,
            "search_by_code_symbol": self.search_by_code_symbol,
        }
        tool_func = tool_map.get(tool_name)
        if not tool_func:
            return f"[错误: 未知工具 '{tool_name}']"
        try:
            return tool_func(**arguments)
        except Exception as e:
            return f"[工具执行错误: {e}]"
    
    @property
    def tools_description(self) -> str:
        return """你可以使用以下工具来检索课程知识库，以帮助你提出更精准的问题：

【工具1】search_course_material
参数:
  - query: str (搜索查询，应该是具体的技术概念或实验主题)
  - n_results: int (可选，返回结果数量，默认5)
用途: 通用课程资料检索，适用于大多数情况。

【工具2】search_os_concept
参数:
  - concept: str (概念名称，如"页表"、"进程切换"、"信号量")
  - detail_level: str (可选，"brief"/"standard"/"detailed"，默认"standard")
用途: 检索特定 OS 概念的详细解释。

【工具3】search_by_lab
参数:
  - lab_name: str (实验名称，如"lab4"或"虚拟内存")
用途: 按实验章节检索教学目标和要求。

【工具4】search_by_code_symbol
参数:
  - symbol_name: str (代码中的符号名，如函数名、结构体名)
  - context: str (可选，该符号出现的上下文描述)
用途: 按代码符号检索课程中的定义和说明。

使用格式（必须严格遵循）：
<tool_calls>
[
  {"tool": "search_os_concept", "arguments": {"concept": "页表", "detail_level": "detailed"}},
  {"tool": "search_by_code_symbol", "arguments": {"symbol_name": "PageTable", "context": "内存管理"}}
]
</tool_calls>

如果不需要检索，输出空列表即可:
<tool_calls>[]</tool_calls>"""


# ==================== 代码处理工具 ====================

class CodeExtractor:
    CODE_EXTENSIONS = {
        '.c': 'c', '.h': 'c',
        '.cpp': 'cpp', '.cc': 'cpp', '.hpp': 'cpp',
        '.rs': 'rust',
        '.py': 'python',
        '.go': 'go',
        '.java': 'java',
        '.s': 'asm', '.S': 'asm',
        '.ld': 'linker',
        '.mk': 'makefile', 'Makefile': 'makefile',
        '.sh': 'shell',
        '.md': 'markdown',
        '.txt': 'text'
    }
    SKIP_DIRS = {'target', 'build', '.git', '__pycache__', '.vscode', '.idea', 'node_modules', 'out'}

    @classmethod
    def extract_from_zip(cls, zip_bytes: bytes, temp_dir: Optional[str] = None) -> Dict[str, CodeFile]:
        code_files: Dict[str, CodeFile] = {}
        with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdir:
            zip_path = os.path.join(tmpdir, "code.zip")
            with open(zip_path, 'wb') as fh:
                fh.write(zip_bytes)
            extract_dir = os.path.join(tmpdir, "extracted")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
            for root, dirs, files in os.walk(extract_dir):
                dirs[:] = [d for d in dirs if d not in cls.SKIP_DIRS]
                for filename in files:
                    ext = os.path.splitext(filename)[1]
                    if ext not in cls.CODE_EXTENSIONS and filename not in cls.CODE_EXTENSIONS:
                        continue
                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_path, extract_dir)
                    try:
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as fh:
                            content = fh.read()
                    except Exception:
                        continue
                    language = cls.CODE_EXTENSIONS.get(ext) or cls.CODE_EXTENSIONS.get(filename, 'unknown')
                    code_files[rel_path] = CodeFile(
                        path=rel_path, content=content, language=language,
                        line_count=len(content.split("\n"))
                    )
        return code_files


class DiffAnalyzer:
    @classmethod
    def analyze(cls, before: Dict[str, CodeFile], after: Dict[str, CodeFile]) -> CodeDiffReport:
        report = CodeDiffReport()
        before_paths = set(before.keys())
        after_paths = set(after.keys())
        report.added_files = sorted(list(after_paths - before_paths))
        report.deleted_files = sorted(list(before_paths - after_paths))
        common_paths = sorted(list(before_paths & after_paths))
        for path in common_paths:
            old_file = before[path]
            new_file = after[path]
            if old_file.content == new_file.content:
                report.unchanged_files.append(path)
            else:
                diff = cls._compute_file_diff(old_file, new_file)
                report.modified_files.append(diff)
        report.summary = cls._generate_summary(report)
        return report

    @classmethod
    def _compute_file_diff(cls, old: CodeFile, new: CodeFile) -> FileDiff:
        old_lines = old.content.split("\n")
        new_lines = new.content.split("\n")
        diff = difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{old.path}", tofile=f"b/{new.path}", lineterm=''
        )
        unified = "\n".join(diff)
        changed_old, changed_new = cls._extract_changed_lines(unified)
        return FileDiff(
            file_path=old.path, status="modified",
            old_content=old.content, new_content=new.content,
            unified_diff=unified,
            changed_lines_old=changed_old, changed_lines_new=changed_new
        )

    @classmethod
    def _extract_changed_lines(cls, unified_diff: str) -> Tuple[List[int], List[int]]:
        old_lines, new_lines = [], []
        old_idx, new_idx = -1, -1
        for line in unified_diff.split("\n"):
            if line.startswith("@@"):
                match = re.match(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
                if match:
                    old_idx, new_idx = int(match.group(1)), int(match.group(2))
            elif line.startswith('-') and not line.startswith('---'):
                if old_idx > 0:
                    old_lines.append(old_idx)
                    old_idx += 1
            elif line.startswith('+') and not line.startswith('+++'):
                if new_idx > 0:
                    new_lines.append(new_idx)
                    new_idx += 1
            elif not line.startswith('\\') and not line.startswith('@@'):
                if old_idx > 0:
                    old_idx += 1
                if new_idx > 0:
                    new_idx += 1
        return old_lines, new_lines

    @classmethod
    def _generate_summary(cls, report: CodeDiffReport) -> str:
        lines = [
            f"新增文件: {len(report.added_files)}个",
            f"删除文件: {len(report.deleted_files)}个",
            f"修改文件: {len(report.modified_files)}个",
            f"未变文件: {len(report.unchanged_files)}个",
        ]
        if report.modified_files:
            lines.append("\n主要修改:")
            for diff in report.modified_files[:5]:
                lines.append(f"  - {diff.file_path} (变更行: {len(diff.changed_lines_new)}行)")
        return "\n".join(lines)


class OSExperimentAnalyzer:
    OS_KEYWORDS = {
        'memory': ['malloc', 'free', 'page', 'segment', 'heap', 'stack', 'mmap', 'vm', 'paging', 'tlb'],
        'process': ['fork', 'exec', 'wait', 'pid', 'process', 'thread', 'pthread', 'schedule', 'pcb'],
        'sync': ['mutex', 'semaphore', 'lock', 'atomic', 'barrier', 'condition', 'spinlock', 'rcu', 'rwlock'],
        'fs': ['inode', 'dentry', 'file', 'open', 'read', 'write', 'mount', 'fs', 'vfs', 'superblock'],
        'interrupt': ['irq', 'interrupt', 'handler', 'trap', 'syscall', 'context', 'idt', 'gdt'],
        'boot': ['boot', 'grub', 'mbr', 'uefi', 'loader', 'kernel', 'multiboot'],
    }

    @classmethod
    def identify_key_changes(cls, diff_report: CodeDiffReport, after_code: Dict[str, CodeFile]) -> List[str]:
        changes = []
        for diff in diff_report.modified_files:
            content = diff.new_content or ""
            for category, keywords in cls.OS_KEYWORDS.items():
                if any(kw in content.lower() for kw in keywords):
                    changes.append(f"[{category.upper()}] 文件 {diff.file_path} 涉及{category}相关修改")
                    break
            func_changes = cls._detect_function_changes(diff)
            changes.extend(func_changes)
        for path in diff_report.added_files:
            if path in after_code:
                file_info = after_code[path]
                changes.append(f"[ADD] 新增文件 {path} ({file_info.line_count}行, {file_info.language})")
        return changes

    @classmethod
    def _detect_function_changes(cls, diff: FileDiff) -> List[str]:
        changes = []
        func_pattern = r'^(?:static\s+)?(?:inline\s+)?(?:\w+\s+)+(\w+)\s*\([^)]*\)\s*\{'
        if diff.new_content:
            new_funcs = set(re.findall(func_pattern, diff.new_content, re.MULTILINE))
            if diff.old_content:
                old_funcs = set(re.findall(func_pattern, diff.old_content, re.MULTILINE))
                for func in new_funcs - old_funcs:
                    changes.append(f"[FUNC] {diff.file_path} 新增函数: {func}")
                for func in old_funcs - new_funcs:
                    changes.append(f"[FUNC] {diff.file_path} 删除函数: {func}")
        return changes


# ==================== 核心：问题生成器 ====================

class QuestionProposer:
    MAX_CONTEXT_LENGTH = 12000

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.moonshot.cn/v1",
        model: str = "kimi-k2.5",
        temperature: float = 1,
        use_kimi: bool = True,
        os_type: str = "rcore",
        persist_dir: str = "/home/gsk/chroma",
        save_dir: str = "/home/gsk/thesis_2026-gsk/questions/results"
    ):
        self.api_key = api_key or os.getenv("MOONSHOT_API_KEY")
        self.use_kimi = use_kimi and bool(self.api_key)
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.os_type = os_type
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.rag_tools = RAGToolKit(os_type=os_type, persist_dir=persist_dir)
        self.rag_enabled = self.rag_tools._initialized
        
        logger.info(f"[DIAG-OS] Questions will be saved to: {self.save_dir}")
        logger.info(f"[DIAG-OS] RAG tools enabled: {self.rag_enabled}")

        if self.use_kimi:
            try:
                self.model_callback = create_kimi_callback(
                    api_key=self.api_key, base_url=base_url,
                    model=model, temperature=temperature
                )
                logger.info(f"[DIAG-OS] Kimi callback created")
            except ValueError as e:
                logger.error(f"[DIAG-OS] Failed to create kimi callback: {e}")
                self.model_callback = create_mock_callback()
                self.use_kimi = False
        else:
            self.model_callback = create_mock_callback()

    # ── Stage 1: 分析代码，输出检索计划 ──
    
    def _analyze_and_plan_retrieval(
        self,
        experiment_requirement: str,
        diff_report: CodeDiffReport,
        after_code: Dict[str, CodeFile],
        key_changes: List[str]
    ) -> List[ToolCall]:
        if not self.rag_enabled:
            logger.info("[STAGE-1] RAG disabled, skipping retrieval planning")
            return []
        
        system_prompt = """你是一位资深的操作系统课程分析专家。你的任务是分析学生的代码变更，判断为了提出高质量的面试问题，需要从课程知识库中检索哪些内容。

你必须遵循以下原则：
1. 聚焦变更：只关注代码中被修改的部分，判断涉及了哪些 OS 概念
2. 精准查询：每个检索请求应该针对一个具体概念，避免过于宽泛
3. 适度检索：通常 2-5 个检索请求即可，不要过度检索
4. 选择合适工具：根据需求选择最匹配的工具

输出格式要求：
请在分析之后，严格按以下格式输出工具调用计划：

<analysis>
你的分析过程（简要说明代码涉及了哪些 OS 概念，为什么需要检索）
</analysis>

<tool_calls>
[
  {"tool": "工具名", "arguments": {"参数名": "参数值"}},
  ...
]
</tool_calls>

如果认为不需要检索任何内容，输出空列表：
<tool_calls>[]</tool_calls>

注意：tool_calls 必须是合法的 JSON 数组格式。"""

        context_lines = [
            f"实验要求: {experiment_requirement}",
            "-" * 40,
            f"\n代码变更摘要:\n{diff_report.summary}",
            f"\n关键变更点:",
        ]
        for change in key_changes[:10]:
            context_lines.append(f"  - {change}")
        
        context_lines.append(f"\n修改文件详情（前2个）:")
        for diff in diff_report.modified_files[:2]:
            context_lines.append(f"\n--- {diff.file_path} ---")
            diff_lines = diff.unified_diff.split("\n")
            context_lines.extend(diff_lines[:25])
            if len(diff_lines) > 25:
                context_lines.append(f"... ({len(diff_lines) - 25} more lines)")
        
        context_lines.append(f"\n\n{self.rag_tools.tools_description}")
        context_lines.append("\n请基于以上代码变更，输出你的分析和检索计划。")
        
        context = "\n".join(context_lines)
        if len(context) > self.MAX_CONTEXT_LENGTH:
            context = context[:self.MAX_CONTEXT_LENGTH] + "\n\n... (truncated)\n"
        
        agent = Agent(system_prompt=system_prompt, model_callback=self.model_callback)
        logger.info("[STAGE-1] Running retrieval planning agent...")
        response = agent.run(context)
        
        tool_calls = self._parse_tool_calls(response.content)
        logger.info(f"[STAGE-1] Agent requested {len(tool_calls)} tool calls")
        for tc in tool_calls:
            logger.info(f"  - {tc.tool_name}: {tc.arguments}")
        
        return tool_calls
    
    def _parse_tool_calls(self, content: str) -> List[ToolCall]:
        tool_calls = []
        
        match = re.search(r'<tool_calls>\s*(\[.*?\])\s*</tool_calls>', content, re.DOTALL)
        if not match:
            match = re.search(r'(\[\s*\{.*?\}\s*\])', content, re.DOTALL)
        
        if match:
            try:
                raw_calls = json.loads(match.group(1))
                for call in raw_calls:
                    if isinstance(call, dict) and "tool" in call:
                        tool_calls.append(ToolCall(
                            tool_name=call["tool"],
                            arguments=call.get("arguments", {})
                        ))
            except json.JSONDecodeError as e:
                logger.warning(f"[STAGE-1] Failed to parse tool calls JSON: {e}")
        
        return tool_calls

    # ── Stage 2: 执行检索 ──
    
    def _execute_retrieval(self, tool_calls: List[ToolCall]) -> str:
        if not tool_calls:
            return ""
        
        results = []
        for i, call in enumerate(tool_calls, 1):
            logger.info(f"[STAGE-2] Executing tool {i}/{len(tool_calls)}: {call.tool_name}")
            result = self.rag_tools.execute(call.tool_name, call.arguments)
            results.append(f"\n【检索 {i}】工具: {call.tool_name}\n参数: {call.arguments}\n结果:\n{result}")
        
        return "\n".join(results)

    # ── Stage 3: 生成问题 ──
    
    def generate_questions(
        self,
        experiment_requirement: str,
        before_code: Dict[str, CodeFile],
        after_code: Dict[str, CodeFile],
        diff_report: CodeDiffReport,
        num_questions: int = 5
    ) -> QuestionSet:
        key_changes = OSExperimentAnalyzer.identify_key_changes(diff_report, after_code)
        
        tool_calls = self._analyze_and_plan_retrieval(
            experiment_requirement, diff_report, after_code, key_changes
        )
        
        retrieval_results = self._execute_retrieval(tool_calls)
        if retrieval_results:
            logger.info(f"[STAGE-2] Total retrieval results length: {len(retrieval_results)}")
        
        system_prompt = self._build_system_prompt(num_questions)
        context = self._build_final_context(
            experiment_requirement, before_code, after_code,
            diff_report, key_changes, retrieval_results
        )
        
        original_len = len(context)
        if len(context) > self.MAX_CONTEXT_LENGTH:
            logger.warning(f"[STAGE-3] Context too long ({len(context)}), truncating")
            context = context[:self.MAX_CONTEXT_LENGTH] + "\n\n... (content truncated)\n"
        logger.info(f"[STAGE-3] Final context length: {len(context)} (original: {original_len})")
        
        agent = Agent(system_prompt=system_prompt, model_callback=self.model_callback)
        logger.info("[STAGE-3] Running question generation agent...")
        agent_response = agent.run(context)
        
        logger.info(f"[STAGE-3] Raw response length: {len(agent_response.content)}")
        
        questions = self._parse_questions(agent_response.content)
        logger.info(f"[STAGE-3] Parsed {len(questions)} questions")
        
        if not questions and not self.use_kimi:
            questions = [
                ProposedQuestion(id=1, category="concept", question="请解释实验的核心OS原理。"),
                ProposedQuestion(id=2, category="implementation", question="请说明代码中关键函数的设计思路。"),
            ]
        
        question_set = QuestionSet(
            experiment_title=experiment_requirement[:50] + "..." if len(experiment_requirement) > 50 else experiment_requirement,
            questions=questions,
            summary=f"基于{len(diff_report.modified_files)}个修改文件和{len(diff_report.added_files)}个新增文件生成{len(questions)}个问题（OS类型: {self.os_type}, RAG检索: {len(tool_calls)}次）",
            metadata={
                "diff_summary": diff_report.summary,
                "total_files": len(before_code) + len(after_code),
                "raw_response": agent_response.content,
                "context_length": len(context),
                "use_kimi": self.use_kimi,
                "os_type": self.os_type,
                "rag_enabled": self.rag_enabled,
                "rag_tool_calls": [tc.to_dict() for tc in tool_calls],
                "rag_results_length": len(retrieval_results),
            }
        )
        
        self._save_question_set(question_set, experiment_requirement, diff_report)
        return question_set

    # ── 修改1: system prompt 加入代码片段要求 ──
    
    def _build_system_prompt(self, num_questions: int) -> str:
        return f"""你是一位资深的操作系统课程助教和面试官。你的任务是根据学生的实验代码修改情况和课程知识库资料，设计针对性的面试问题。

你的提问原则：
1. 深度优先于广度：针对关键修改点深入追问，而非泛泛而谈
2. 原理结合实践：问题应考察学生对OS原理的理解以及代码实现细节
3. 区分度：问题应能区分"真正理解"和"照搬代码"的学生
4. 聚焦变更：重点关注学生修改的部分，而非未修改的代码
5. 循序渐进：从具体实现到设计决策，再到潜在问题
6. 结合课程知识：参考提供的课程教材内容，确保问题与课程教学一致
7. 精准引用：如果课程资料中有明确的相关内容，请在问题中体现

【代码提问格式 - 重要】
每个问题必须附带与问题直接相关的代码片段，供答题者参考分析。代码片段必须使用以下标记包裹：
<code>
[代码内容]
</code>

代码标记规则：
- 代码块要简洁，通常不超过15行，展示关键逻辑即可
- 可以包含行内注释（以#或//开头）
- 如果是多段代码，每段用独立的<code>...</code>包裹
- 提问时要明确指出让学生分析代码的哪个方面（复杂度/bug/优化/原理等）
- 代码必须直接来自学生实验代码的修改部分，不要编造不存在的代码

例如：
"请看这段代码实现：<code>
fn page_table_walk(vpn: VirtPageNum, root: usize) -> Option<PageTableEntry> {{
    let idxs = vpn.indexes();
    let mut ppn = PhysPageNum(root);
    for i in 0..3 {{
        let pte = &mut ppn.get_pte_array()[idxs[i]];
        if !pte.is_valid() {{
            return None;
        }}
        ppn = pte.ppn();
    }}
    Some(&mut ppn.get_pte_array()[idxs[2]])
}}
</code>
这段页表遍历代码中，如果某级页表项无效时直接返回None，这种处理方式在OS中是否合理？请说明理由。"

问题类型包括：
- concept: 概念理解（如页表机制、调度策略）
- implementation: 实现细节（如某段代码的具体逻辑）
- debugging: 调试能力（如边界情况、错误处理）
- optimization: 优化思考（如性能、资源管理）
- understanding: 整体理解（如设计决策、架构选择）

请用中文提问，技术术语可保留英文。每个问题必须具体、明确，避免模糊表述。

请生成{num_questions}个问题，按以下格式输出：

## 问题1 [类型]
问题内容...

**出题理由**: ...

---

## 问题2 [类型]
...

（以此类推）"""

    def _build_final_context(
        self,
        requirement: str,
        before: Dict[str, CodeFile],
        after: Dict[str, CodeFile],
        diff: CodeDiffReport,
        key_changes: List[str],
        retrieval_results: str
    ) -> str:
        lines: List[str] = []
        lines.append("实验要求:")
        lines.append(requirement)
        lines.append("=" * 40)
        
        if retrieval_results:
            lines.append("\n【课程知识库检索结果】")
            lines.append("以下是从课程知识库中检索到的相关资料，请在提问时充分参考：")
            lines.append(retrieval_results)
            lines.append("=" * 40)
        
        lines.append("\n代码变更摘要:")
        lines.append(diff.summary)
        
        if key_changes:
            lines.append("\n关键变更点:")
            for change in key_changes[:10]:
                lines.append(f"  {change}")
        
        lines.append("\n详细代码变更:")
        for file_diff in diff.modified_files[:2]:
            lines.append(f"\n--- {file_diff.file_path} ---")
            diff_lines = file_diff.unified_diff.split("\n")
            preview_lines = diff_lines[:30]
            lines.extend(preview_lines)
            if len(diff_lines) > 30:
                lines.append(f"... ({len(diff_lines) - 30} more lines)")
        
        if diff.added_files:
            lines.append("\n新增文件:")
            for path in diff.added_files[:2]:
                if path in after:
                    file_info = after[path]
                    lines.append(f"\n--- {path} ({file_info.line_count}行) ---")
                    content_lines = file_info.content.split("\n")
                    preview = "\n".join(content_lines[:15])
                    lines.append(preview)
                    if len(content_lines) > 15:
                        lines.append(f"... ({len(content_lines)-15} more lines)")
        
        lines.append("\n")
        lines.append("请基于以上代码变更和课程知识库资料，生成面试问题。")
        lines.append("要求：每个问题必须附带相关的代码片段（用<code>标记），问题必须紧密结合代码修改内容。")
        
        return "\n".join(lines)

    # ── 修改2: 问题解析提取 code_snippets ──
    
    def _parse_questions(self, content: str) -> List[ProposedQuestion]:
        questions: List[ProposedQuestion] = []
        if "[Kimi API 错误]" in content or "[API错误]" in content:
            logger.error(f"[DIAG-OS] LLM returned error: {content[:200]}")
            return questions
        
        # 先按问题分割
        pattern = r'##\s*问题\s*(\d+)\s*\[([^\]]+)\]\s*\n(.*?)\n(?=##\s*问题|\Z)'
        matches = re.findall(pattern, content, re.DOTALL)
        logger.info(f"[DIAG-OS] Regex matched {len(matches)} questions")
        
        for idx, (qid, category, qcontent) in enumerate(matches, 1):
            # 提取代码片段
            code_snippets = []
            code_pattern = r'<code>(.*?)</code>'
            code_matches = re.findall(code_pattern, qcontent, re.DOTALL)
            for snippet in code_matches:
                code_snippets.append(snippet.strip())
            
            # 移除代码标记后的纯文本
            clean_content = re.sub(code_pattern, '', qcontent, flags=re.DOTALL).strip()
            
            rationale_match = re.search(r'\*\*出题理由\*\*[:：]\s*(.*?)\n(?=---|\Z)', clean_content, re.DOTALL)
            rationale = rationale_match.group(1).strip() if rationale_match else ""
            question_text = re.sub(r'\*\*出题理由\*\*[:：].*', '', clean_content, flags=re.DOTALL).strip()
            question_text = re.sub(r'^[-*]\s*', '', question_text, flags=re.MULTILINE)
            
            if question_text and len(question_text) > 5:
                questions.append(ProposedQuestion(
                    id=idx,
                    category=category.strip().lower(),
                    question=question_text,
                    rationale=rationale,
                    code_snippets=code_snippets
                ))
        
        if not questions:
            logger.warning("[DIAG-OS] Primary regex failed, trying fallback parsing")
            questions = self._fallback_parse(content)
        return questions

    def _fallback_parse(self, content: str) -> List[ProposedQuestion]:
        questions: List[ProposedQuestion] = []
        lines = content.strip().split("\n")
        current_q = None
        qid = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            match = re.match(r'^(?:##\s*)?(?:问题)?\s*(\d+)[\.、]?\s*(?:\[[^\]]+\])?\s*(.*)', line)
            if match:
                if current_q:
                    questions.append(current_q)
                qid += 1
                text = match.group(2).strip()
                current_q = ProposedQuestion(id=qid, category="general", question=text)
            elif current_q:
                # 在 fallback 中也尝试提取代码
                code_match = re.match(r'<code>(.*?)</code>', line, re.DOTALL)
                if code_match:
                    current_q.code_snippets.append(code_match.group(1).strip())
                else:
                    current_q.question += "\n" + line
        if current_q:
            questions.append(current_q)
        logger.info(f"[DIAG-OS] Fallback parsed {len(questions)} questions")
        return questions

    # ── 修改3: 保存逻辑加入 code_snippets ──
    
    def _save_question_set(
        self,
        question_set: QuestionSet,
        experiment_requirement: str,
        diff_report: CodeDiffReport
    ) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_title = re.sub(r'[^\w\-]', '_', question_set.experiment_title[:30])
        filename_base = f"{timestamp}_{self.os_type}_{safe_title}"
        
        data = {
            "generated_at": datetime.now().isoformat(),
            "os_type": self.os_type,
            "experiment_requirement": experiment_requirement,
            "experiment_title": question_set.experiment_title,
            "summary": question_set.summary,
            "total_questions": len(question_set.questions),
            "questions": [],
            "diff_summary": {
                "added_files": diff_report.added_files,
                "deleted_files": diff_report.deleted_files,
                "modified_files": [d.file_path for d in diff_report.modified_files],
                "summary": diff_report.summary
            },
            "metadata": question_set.metadata
        }
        
        for q in question_set.questions:
            data["questions"].append({
                "id": q.id,
                "category": q.category,
                "question": q.question,
                "rationale": q.rationale,
                "target_file": q.target_file,
                "target_lines": q.target_lines,
                "code_snippets": q.code_snippets  # ← 新增
            })
        
        json_path = self.save_dir / f"{filename_base}.json"
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"[DIAG-OS] Saved questions to JSON: {json_path}")
        except Exception as e:
            logger.error(f"[DIAG-OS] Failed to save JSON: {e}")
        
        md_path = self.save_dir / f"{filename_base}.md"
        try:
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# OS 实验面试问题集\n\n")
                f.write(f"**生成时间**: {data['generated_at']}\n\n")
                f.write(f"**OS 类型**: {self.os_type}\n\n")
                f.write(f"**实验要求**: {experiment_requirement}\n\n")
                f.write(f"**摘要**: {question_set.summary}\n\n")
                
                rag_calls = question_set.metadata.get("rag_tool_calls", [])
                if rag_calls:
                    f.write(f"**RAG 检索**: 执行了 {len(rag_calls)} 次工具调用\n")
                    for call in rag_calls:
                        f.write(f"  - `{call['tool']}`: {call['arguments']}\n")
                    f.write("\n")
                
                f.write("---\n\n")
                for q in question_set.questions:
                    f.write(f"## 问题 {q.id} [{q.category.upper()}]\n\n")
                    f.write(f"{q.question}\n\n")
                    
                    # ← 新增：输出代码片段
                    if q.code_snippets:
                        f.write("**参考代码**:\n\n")
                        for i, snippet in enumerate(q.code_snippets, 1):
                            f.write(f"```\n{snippet}\n```\n\n")
                    
                    if q.rationale:
                        f.write(f"**出题理由**: {q.rationale}\n\n")
                    if q.target_file:
                        f.write(f"**目标文件**: {q.target_file}\n")
                    f.write("---\n\n")
                f.write("\n## 代码变更摘要\n\n")
                f.write(f"```\n{diff_report.summary}\n```\n")
            logger.info(f"[DIAG-OS] Saved questions to Markdown: {md_path}")
        except Exception as e:
            logger.error(f"[DIAG-OS] Failed to save Markdown: {e}")


# ==================== 独立运行入口 ====================

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        pass
    else:
        import uvicorn
        from fastapi import FastAPI
        app = FastAPI()
        uvicorn.run(app, host="0.0.0.0", port=8000)