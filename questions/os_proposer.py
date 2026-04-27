"""
os_proposer.py - 操作系统课程实验代码提问器（Agent模式）

功能：
1. 接收学生修改前和修改后的代码zip压缩包
2. 解压并提取代码文件内容
3. 分析代码差异（文件增删改、关键逻辑变化）
4. 基于实验要求和代码变更，利用大模型生成深度提问

FastAPI 接口：
    POST /api/os-experiment/start   - 生成问题集
    POST /api/os-experiment/diff    - 仅分析差异
"""

import os
import re
import uuid
import zipfile
import tempfile
import difflib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field, asdict

from fastapi import FastAPI, UploadFile, File, Form, APIRouter
from fastapi.responses import JSONResponse

# ========== 完全复用 question_proposer 的 Agent 模式 ==========
from question_proposer import create_kimi_callback, create_mock_callback, Agent, AgentResponse

logger = logging.getLogger(__name__)

# ==================== 数据模型 ====================

@dataclass
class CodeFile:
    """代码文件信息"""
    path: str           # 相对路径
    content: str        # 文件内容
    language: str       # 编程语言
    line_count: int     # 行数
    
    def get_snippet(self, start_line: int, end_line: int) -> str:
        """获取代码片段"""
        lines = self.content.split('\n')
        return '\n'.join(lines[start_line-1:end_line])


@dataclass
class FileDiff:
    """单个文件的差异信息"""
    file_path: str
    status: str         # "added", "deleted", "modified", "unchanged"
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    unified_diff: str = ""
    changed_lines_old: List[int] = field(default_factory=list)
    changed_lines_new: List[int] = field(default_factory=list)


@dataclass
class CodeDiffReport:
    """代码差异报告"""
    added_files: List[str] = field(default_factory=list)
    deleted_files: List[str] = field(default_factory=list)
    modified_files: List[FileDiff] = field(default_factory=list)
    unchanged_files: List[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class ProposedQuestion:
    """生成的问题"""
    id: int
    category: str       # "concept", "implementation", "debugging", "optimization", "understanding"
    question: str
    target_file: Optional[str] = None
    target_lines: Optional[Tuple[int, int]] = None
    rationale: str = ""  # 出题理由


@dataclass
class QuestionSet:
    """问题集合"""
    experiment_title: str
    questions: List[ProposedQuestion]
    summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== 代码处理工具 ====================

class CodeExtractor:
    """代码提取器 - 从zip文件中提取代码"""
    
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
        """
        从zip字节数据中提取代码文件
        
        Args:
            zip_bytes: zip文件的二进制数据
            temp_dir: 临时目录路径（可选）
            
        Returns:
            Dict[str, CodeFile]: 文件路径到CodeFile的映射
        """
        code_files: Dict[str, CodeFile] = {}
        
        with tempfile.TemporaryDirectory(dir=temp_dir) as tmpdir:
            zip_path = os.path.join(tmpdir, "code.zip")
            with open(zip_path, 'wb') as f:
                f.write(zip_bytes)
            
            extract_dir = os.path.join(tmpdir, "extracted")
            os.makedirs(extract_dir, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
            
            # 遍历提取的文件
            for root, dirs, files in os.walk(extract_dir):
                # 跳过不需要的目录
                dirs[:] = [d for d in dirs if d not in cls.SKIP_DIRS]
                
                for filename in files:
                    ext = os.path.splitext(filename)[1]
                    if ext not in cls.CODE_EXTENSIONS and filename not in cls.CODE_EXTENSIONS:
                        continue
                    
                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_path, extract_dir)
                    
                    try:
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                    except Exception:
                        continue
                    
                    language = cls.CODE_EXTENSIONS.get(ext) or cls.CODE_EXTENSIONS.get(filename, 'unknown')
                    
                    code_files[rel_path] = CodeFile(
                        path=rel_path,
                        content=content,
                        language=language,
                        line_count=len(content.split('\n'))
                    )
        
        return code_files


class DiffAnalyzer:
    """差异分析器 - 分析修改前后的代码差异"""
    
    @classmethod
    def analyze(cls, before: Dict[str, CodeFile], after: Dict[str, CodeFile]) -> CodeDiffReport:
        """
        分析两组代码的差异
        
        Args:
            before: 修改前的代码文件
            after: 修改后的代码文件
            
        Returns:
            CodeDiffReport: 差异报告
        """
        report = CodeDiffReport()
        
        before_paths = set(before.keys())
        after_paths = set(after.keys())
        
        # 新增文件
        report.added_files = sorted(list(after_paths - before_paths))
        # 删除文件
        report.deleted_files = sorted(list(before_paths - after_paths))
        # 共同文件
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
        """计算单个文件的unified diff"""
        old_lines = old.content.split('\n')
        new_lines = new.content.split('\n')
        
        diff = difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{old.path}",
            tofile=f"b/{new.path}",
            lineterm=''
        )
        unified = '\n'.join(diff)
        
        # 提取变更行号
        changed_old, changed_new = cls._extract_changed_lines(unified)
        
        return FileDiff(
            file_path=old.path,
            status="modified",
            old_content=old.content,
            new_content=new.content,
            unified_diff=unified,
            changed_lines_old=changed_old,
            changed_lines_new=changed_new
        )
    
    @classmethod
    def _extract_changed_lines(cls, unified_diff: str) -> Tuple[List[int], List[int]]:
        """从unified diff中提取变更的行号"""
        old_lines: List[int] = []
        new_lines: List[int] = []
        
        old_idx = -1
        new_idx = -1
        
        for line in unified_diff.split('\n'):
            if line.startswith('@@'):
                # 解析 @@ -start,count +start,count @@
                match = re.match(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
                if match:
                    old_idx = int(match.group(1))
                    new_idx = int(match.group(2))
            
            elif line.startswith('-') and not line.startswith('---'):
                if old_idx > 0:
                    old_lines.append(old_idx)
                    old_idx += 1
            
            elif line.startswith('+') and not line.startswith('+++'):
                if new_idx > 0:
                    new_lines.append(new_idx)
                    new_idx += 1
            
            elif not line.startswith('\\') and not line.startswith('@@'):
                # 上下文行
                if old_idx > 0:
                    old_idx += 1
                if new_idx > 0:
                    new_idx += 1
        
        return old_lines, new_lines
    
    @classmethod
    def _generate_summary(cls, report: CodeDiffReport) -> str:
        """生成差异摘要"""
        lines = []
        lines.append(f"新增文件: {len(report.added_files)}个")
        lines.append(f"删除文件: {len(report.deleted_files)}个")
        lines.append(f"修改文件: {len(report.modified_files)}个")
        lines.append(f"未变文件: {len(report.unchanged_files)}个")
        
        if report.modified_files:
            lines.append("\n主要修改:")
            for diff in report.modified_files[:5]:  # 最多显示5个
                lines.append(f"  - {diff.file_path} (变更行: {len(diff.changed_lines_new)}行)")
        
        return '\n'.join(lines)


class OSExperimentAnalyzer:
    """操作系统实验专用分析器"""
    
    # 操作系统实验常见关注点
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
        """
        识别操作系统实验中的关键变更点
        
        Returns:
            List[str]: 关键变更描述列表
        """
        changes: List[str] = []
        
        # 分析修改的文件
        for diff in diff_report.modified_files:
            content = diff.new_content or ""
            
            # 检查是否涉及关键OS概念
            for category, keywords in cls.OS_KEYWORDS.items():
                if any(kw in content.lower() for kw in keywords):
                    changes.append(f"[{category.upper()}] 文件 {diff.file_path} 涉及{category}相关修改")
                    break
            
            # 检查关键函数变更
            func_changes = cls._detect_function_changes(diff)
            changes.extend(func_changes)
        
        # 分析新增文件
        for path in diff_report.added_files:
            if path in after_code:
                file_info = after_code[path]
                changes.append(f"[ADD] 新增文件 {path} ({file_info.line_count}行, {file_info.language})")
        
        return changes
    
    @classmethod
    def _detect_function_changes(cls, diff: FileDiff) -> List[str]:
        """检测函数级别的变更"""
        changes: List[str] = []
        
        # C/C/Rust 函数定义正则
        func_pattern = r'^(?:static\s+)?(?:inline\s+)?(?:\w+\s+)+(\w+)\s*\([^)]*\)\s*\{'
        
        if diff.new_content:
            new_funcs = set(re.findall(func_pattern, diff.new_content, re.MULTILINE))
            if diff.old_content:
                old_funcs = set(re.findall(func_pattern, diff.old_content, re.MULTILINE))
                
                added_funcs = new_funcs - old_funcs
                removed_funcs = old_funcs - new_funcs
                
                for func in added_funcs:
                    changes.append(f"[FUNC] {diff.file_path} 新增函数: {func}")
                for func in removed_funcs:
                    changes.append(f"[FUNC] {diff.file_path} 删除函数: {func}")
        
        return changes


# ==================== 大模型提问生成器（Agent模式）====================

class QuestionProposer:
    """问题生成器 - 基于代码差异生成面试问题（使用与question_proposer相同的Agent模式）"""
    
    # 上下文长度限制（字符数），防止超过模型上下文窗口
    MAX_CONTEXT_LENGTH = 8000
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.moonshot.cn/v1",
        model: str = "kimi-k2.5",
        temperature: float = 1,
        use_kimi: bool = True
    ):
        self.api_key = api_key or os.getenv("MOONSHOT_API_KEY")
        self.use_kimi = use_kimi and bool(self.api_key)
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        
        # 使用与 question_proposer 完全相同的回调创建方式
        if self.use_kimi:
            try:
                self.model_callback = create_kimi_callback(
                    api_key=self.api_key,
                    base_url=base_url,
                    model=model,
                    temperature=temperature
                )
                logger.info(f"[DIAG-OS] Kimi callback created, base_url={base_url}, model={model}")
            except ValueError as e:
                logger.error(f"[DIAG-OS] Failed to create kimi callback: {e}")
                self.model_callback = create_mock_callback()
                self.use_kimi = False
        else:
            logger.warning("[DIAG-OS] Using mock callback")
            self.model_callback = create_mock_callback()
    
    def generate_questions(
        self,
        experiment_requirement: str,
        before_code: Dict[str, CodeFile],
        after_code: Dict[str, CodeFile],
        diff_report: CodeDiffReport,
        num_questions: int = 5
    ) -> QuestionSet:
        """
        生成问题集（使用Agent模式）
        
        Args:
            experiment_requirement: 实验要求描述
            before_code: 修改前的代码
            after_code: 修改后的代码
            diff_report: 差异报告
            num_questions: 生成问题数量
            
        Returns:
            QuestionSet: 问题集合
        """
        # 构建系统提示词
        system_prompt = self._build_system_prompt(num_questions)
        
        # 构建用户输入（包含实验要求、代码差异等上下文）
        context = self._build_context(experiment_requirement, before_code, after_code, diff_report)
        
        # 【关键修复】截断过长的上下文，防止 400 Bad Request
        original_context_len = len(context)
        if len(context) > self.MAX_CONTEXT_LENGTH:
            logger.warning(f"[DIAG-OS] Context too long ({len(context)}), truncating to {self.MAX_CONTEXT_LENGTH}")
            context = context[:self.MAX_CONTEXT_LENGTH] + "\n\n... (内容已截断)\n"
        logger.info(f"[DIAG-OS] Context length: {len(context)} (original: {original_context_len})")
        
        # 创建Agent（与question_proposer完全相同的模式）
        agent = Agent(
            system_prompt=system_prompt,
            model_callback=self.model_callback
        )
        
        # 运行Agent生成问题
        logger.info("[DIAG-OS] Calling agent.run()...")
        agent_response = agent.run(context)
        
        # 【关键调试】打印原始响应
        logger.info(f"[DIAG-OS] Raw response length: {len(agent_response.content)}")
        logger.info(f"[DIAG-OS] Raw response preview: {agent_response.content[:500]}...")
        logger.info(f"[DIAG-OS] Raw response full: {agent_response.content}")
        
        # 解析问题
        questions = self._parse_questions(agent_response.content)
        logger.info(f"[DIAG-OS] Parsed {len(questions)} questions")
        
        # 如果解析失败且是模拟模式，返回模拟问题
        if not questions and not self.use_kimi:
            logger.warning("[DIAG-OS] No questions parsed in mock mode, generating fallback")
            questions = [
                ProposedQuestion(id=1, category="concept", question="请解释实验的核心OS原理。"),
                ProposedQuestion(id=2, category="implementation", question="请说明代码中关键函数的设计思路。"),
            ]
        
        return QuestionSet(
            experiment_title=experiment_requirement[:50] + "..." if len(experiment_requirement) > 50 else experiment_requirement,
            questions=questions,
            summary=f"基于{len(diff_report.modified_files)}个修改文件和{len(diff_report.added_files)}个新增文件生成{len(questions)}个问题",
            metadata={
                "diff_summary": diff_report.summary,
                "total_files": len(before_code) + len(after_code),
                "raw_response": agent_response.content,
                "context_length": len(context),
                "use_kimi": self.use_kimi
            }
        )
    
    def _build_system_prompt(self, num_questions: int) -> str:
        """构建系统提示词"""
        return f"""你是一位资深的操作系统课程助教和面试官。你的任务是根据学生的实验代码修改情况，设计针对性的面试问题。

你的提问原则：
1. 深度优先于广度：针对关键修改点深入追问，而非泛泛而谈
2. 原理结合实践：问题应考察学生对OS原理的理解以及代码实现细节
3. 区分度：问题应能区分"真正理解"和"照搬代码"的学生
4. 聚焦变更：重点关注学生修改的部分，而非未修改的代码
5. 循序渐进：从具体实现到设计决策，再到潜在问题

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
    
    def _build_context(
        self,
        requirement: str,
        before: Dict[str, CodeFile],
        after: Dict[str, CodeFile],
        diff: CodeDiffReport
    ) -> str:
        """构建给模型的上下文信息（作为Agent的用户输入）"""
        lines: List[str] = []
        lines.append("实验要求:")
        lines.append(requirement)
        lines.append("-" * 40)
        
        # 差异摘要
        lines.append("\n代码变更摘要:")
        lines.append(diff.summary)
        
        # 关键变更
        key_changes = OSExperimentAnalyzer.identify_key_changes(diff, after)
        if key_changes:
            lines.append("\n关键变更点:")
            for change in key_changes[:10]:  # 限制数量
                lines.append(f"  {change}")
        
        # 修改文件的详细diff（严格限制长度）
        lines.append("\n详细代码变更:")
        for file_diff in diff.modified_files[:2]:  # 最多2个文件
            lines.append(f"\n--- {file_diff.file_path} ---")
            diff_lines = file_diff.unified_diff.split('\n')
            # 只取前30行diff，防止过长
            preview_lines = diff_lines[:30]
            lines.extend(preview_lines)
            if len(diff_lines) > 30:
                lines.append(f"... ({len(diff_lines) - 30} more lines)")
        
        # 新增文件的内容摘要（严格限制）
        if diff.added_files:
            lines.append("\n新增文件:")
            for path in diff.added_files[:2]:
                if path in after:
                    file_info = after[path]
                    lines.append(f"\n--- {path} ({file_info.line_count}行) ---")
                    content_lines = file_info.content.split('\n')
                    preview = '\n'.join(content_lines[:15])
                    lines.append(preview)
                    if len(content_lines) > 15:
                        lines.append(f"... ({len(content_lines)-15} more lines)")
        
        lines.append("\n")
        lines.append("请基于以上信息生成面试问题。")
        
        return '\n'.join(lines)
    
    def _parse_questions(self, content: str) -> List[ProposedQuestion]:
        """解析模型返回的问题"""
        questions: List[ProposedQuestion] = []
        
        # 检查是否是错误响应
        if "[Kimi API 错误]" in content or "[API错误]" in content:
            logger.error(f"[DIAG-OS] LLM returned error: {content[:200]}")
            return questions
        
        # 按问题分割
        pattern = r'##\s*问题\s*(\d+)\s*\[([^\]]+)\]\s*\n(.*?)(?=\n##\s*问题|\Z)'
        matches = re.findall(pattern, content, re.DOTALL)
        logger.info(f"[DIAG-OS] Regex matched {len(matches)} questions")
        
        for idx, (qid, category, qcontent) in enumerate(matches, 1):
            # 提取出题理由
            rationale_match = re.search(r'\*\*出题理由\*\*[:：]\s*(.*?)(?=\n---|\Z)', qcontent, re.DOTALL)
            rationale = rationale_match.group(1).strip() if rationale_match else ""
            
            # 清理问题内容
            question_text = re.sub(r'\*\*出题理由\*\*[:：].*', '', qcontent, flags=re.DOTALL).strip()
            question_text = re.sub(r'^[-*]\s*', '', question_text, flags=re.MULTILINE)
            
            if question_text and len(question_text) > 5:
                questions.append(ProposedQuestion(
                    id=idx,
                    category=category.strip().lower(),
                    question=question_text,
                    rationale=rationale
                ))
        
        # 如果正则解析失败，尝试备用解析
        if not questions:
            logger.warning("[DIAG-OS] Primary regex failed, trying fallback parsing")
            questions = self._fallback_parse(content)
        
        return questions
    
    def _fallback_parse(self, content: str) -> List[ProposedQuestion]:
        """备用解析：当主正则失败时使用"""
        questions: List[ProposedQuestion] = []
        
        # 尝试按数字序号分割
        lines = content.strip().split('\n')
        current_q = None
        qid = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检测问题开头：数字. 或 数字、 或 ##问题
            match = re.match(r'^(?:##\s*)?(?:问题)?\s*(\d+)[\.、]?\s*(?:\[[^\]]+\])?\s*(.*)', line)
            if match:
                if current_q:
                    questions.append(current_q)
                qid += 1
                text = match.group(2).strip()
                current_q = ProposedQuestion(id=qid, category="general", question=text)
            elif current_q:
                current_q.question += "\n" + line
        
        if current_q:
            questions.append(current_q)
        
        logger.info(f"[DIAG-OS] Fallback parsed {len(questions)} questions")
        return questions


# ==================== FastAPI 路由 ====================

router = APIRouter(prefix="/api/os-experiment", tags=["os-experiment"])

proposer_instance: Optional[QuestionProposer] = None


def get_os_proposer() -> QuestionProposer:
    """获取或初始化 QuestionProposer 单例"""
    global proposer_instance
    if proposer_instance is None:
        proposer_instance = QuestionProposer()
    return proposer_instance


@router.post("/start")
async def os_experiment_start(
    experiment_requirement: str = Form(..., description="实验要求描述"),
    before_zip: UploadFile = File(..., description="修改前的代码zip"),
    after_zip: UploadFile = File(..., description="修改后的代码zip"),
    num_questions: int = Form(5, description="生成问题数量"),
    student_id: Optional[str] = Form(None, description="学生ID")
):
    """
    接收修改前后的代码zip，分析差异并生成面试问题
    
    返回格式与原有 /api/evaluation/start 兼容
    """
    try:
        logger.info("[DIAG-OS-API] /api/os-experiment/start called")
        
        # 1. 读取zip文件
        before_bytes = await before_zip.read()
        after_bytes = await after_zip.read()
        logger.info(f"[DIAG-OS-API] before={len(before_bytes)}B, after={len(after_bytes)}B")
        
        # 2. 提取代码
        before_code = CodeExtractor.extract_from_zip(before_bytes)
        after_code = CodeExtractor.extract_from_zip(after_bytes)
        logger.info(f"[DIAG-OS-API] before files={list(before_code.keys())}, after files={list(after_code.keys())}")
        
        if not before_code and not after_code:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "未能从zip中提取到任何代码文件"}
            )
        
        # 3. 分析差异
        diff_report = DiffAnalyzer.analyze(before_code, after_code)
        logger.info(f"[DIAG-OS-API] diff: {diff_report.summary.replace(chr(10), ' | ')}")
        
        # 4. 生成问题（使用Agent模式）
        proposer = get_os_proposer()
        question_set = proposer.generate_questions(
            experiment_requirement=experiment_requirement,
            before_code=before_code,
            after_code=after_code,
            diff_report=diff_report,
            num_questions=num_questions
        )
        
        logger.info(f"[DIAG-OS-API] Generated {len(question_set.questions)} questions")
        
        # 5. 转换为前端兼容格式
        return {
            "evaluation_id": f"os_{uuid.uuid4().hex[:12]}",
            "status": "ready",
            "original_question": experiment_requirement[:100],
            "exam_questions": [
                {"id": str(q.id), "text": f"[{q.category.upper()}] {q.question}"}
                for q in question_set.questions
            ],
            "question_count": len(question_set.questions),
            "generated_at": datetime.now().isoformat()
        }
        
    except ValueError as e:
        logger.error(f"[DIAG-OS-API] Config error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"配置错误: {str(e)}"}
        )
    except Exception as e:
        logger.error(f"[DIAG-OS-API] Server error: {e}")
        import traceback
        logger.error(f"[DIAG-OS-API] Traceback:\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"服务器错误: {str(e)}"}
        )


@router.post("/diff")
async def os_experiment_diff(
    before_zip: UploadFile = File(..., description="修改前的代码zip"),
    after_zip: UploadFile = File(..., description="修改后的代码zip")
):
    """
    仅分析代码差异，不生成问题
    """
    try:
        before_bytes = await before_zip.read()
        after_bytes = await after_zip.read()
        
        before_code = CodeExtractor.extract_from_zip(before_bytes)
        after_code = CodeExtractor.extract_from_zip(after_bytes)
        diff_report = DiffAnalyzer.analyze(before_code, after_code)
        key_changes = OSExperimentAnalyzer.identify_key_changes(diff_report, after_code)
        
        return {
            "success": True,
            "data": {
                "added_files": diff_report.added_files,
                "deleted_files": diff_report.deleted_files,
                "modified_files": [
                    {
                        "path": d.file_path,
                        "changed_lines_new": len(d.changed_lines_new),
                        "changed_lines_old": len(d.changed_lines_old),
                        "diff_preview": '\n'.join(d.unified_diff.split('\n')[:30])
                    }
                    for d in diff_report.modified_files
                ],
                "unchanged_files": diff_report.unchanged_files,
                "key_changes": key_changes,
                "summary": diff_report.summary
            }
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


# ==================== 应用入口 ====================

def create_app() -> FastAPI:
    """创建 FastAPI 应用（用于独立运行或作为子应用挂载）"""
    app = FastAPI(
        title="OS Experiment Question Proposer",
        description="操作系统课程实验代码提问器",
        version="1.0.0"
    )
    app.include_router(router)
    return app


# 独立运行入口
app = create_app()


# ==================== 独立测试入口 ====================

def test_proposer():
    """测试提问生成器"""
    # 模拟实验要求
    requirement = """
    实验三：内存管理
    实现一个简化的内存分配器，支持首次适应（First Fit）和最佳适应（Best Fit）算法。
    要求：
    1. 实现 malloc/free 接口
    2. 维护空闲块链表
    3. 支持内存合并
    4. 处理边界情况（如分配0字节、释放空指针等）
    """
    
    # 模拟修改前的代码
    before_code = {
        "malloc.c": CodeFile(
            path="malloc.c",
            content='''#include <stdio.h>
#include <stdlib.h>

typedef struct Block {
    size_t size;
    struct Block* next;
    int free;
} Block;

Block* free_list = NULL;

void* my_malloc(size_t size) {
    Block* curr = free_list;
    while (curr) {
        if (curr->free && curr->size >= size) {
            curr->free = 0;
            return (void*)(curr + 1);
        }
        curr = curr->next;
    }
    Block* block = sbrk(size + sizeof(Block));
    block->size = size;
    block->free = 0;
    block->next = free_list;
    free_list = block;
    return (void*)(block + 1);
}

void my_free(void* ptr) {
    if (!ptr) return;
    Block* block = (Block*)ptr - 1;
    block->free = 1;
}
''',
            language="c",
            line_count=35
        )
    }
    
    # 模拟修改后的代码
    after_code = {
        "malloc.c": CodeFile(
            path="malloc.c",
            content='''#include <stdio.h>
#include <stdlib.h>

typedef struct Block {
    size_t size;
    struct Block* next;
    struct Block* prev;
    int free;
} Block;

Block* free_list = NULL;

void* my_malloc(size_t size) {
    if (size == 0) return NULL;
    
    Block* curr = free_list;
    Block* best = NULL;
    
    while (curr) {
        if (curr->free && curr->size >= size) {
            if (!best || curr->size < best->size) {
                best = curr;
            }
        }
        curr = curr->next;
    }
    
    if (best) {
        best->free = 0;
        return (void*)(best + 1);
    }
    
    Block* block = sbrk(size + sizeof(Block));
    block->size = size;
    block->free = 0;
    block->next = free_list;
    block->prev = NULL;
    if (free_list) free_list->prev = block;
    free_list = block;
    return (void*)(block + 1);
}

void my_free(void* ptr) {
    if (!ptr) return;
    Block* block = (Block*)ptr - 1;
    block->free = 1;
    
    if (block->next && block->next->free) {
        block->size += sizeof(Block) + block->next->size;
        block->next = block->next->next;
        if (block->next) block->next->prev = block;
    }
    if (block->prev && block->prev->free) {
        block->prev->size += sizeof(Block) + block->size;
        block->prev->next = block->next;
        if (block->next) block->next->prev = block->prev;
    }
}
''',
            language="c",
            line_count=58
        ),
        "test.c": CodeFile(
            path="test.c",
            content='''#include <assert.h>
#include "malloc.h"

int main() {
    void* p1 = my_malloc(100);
    void* p2 = my_malloc(200);
    my_free(p1);
    void* p3 = my_malloc(50);
    assert(p3 == p1);
    return 0;
}
''',
            language="c",
            line_count=12
        )
    }
    
    # 分析差异
    diff = DiffAnalyzer.analyze(before_code, after_code)
    print("=" * 60)
    print("差异分析结果:")
    print(diff.summary)
    print()
    
    # 识别关键变更
    key_changes = OSExperimentAnalyzer.identify_key_changes(diff, after_code)
    print("关键变更:")
    for kc in key_changes:
        print(f"  {kc}")
    print()
    
    # 生成问题（需要API Key）
    api_key = os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        print("⚠️ 未设置 MOONSHOT_API_KEY，跳过问题生成")
        print("设置环境变量后运行: export MOONSHOT_API_KEY=your_key")
        return
    
    print("🤖 正在生成问题...")
    proposer = QuestionProposer(api_key=api_key)
    question_set = proposer.generate_questions(
        experiment_requirement=requirement,
        before_code=before_code,
        after_code=after_code,
        diff_report=diff,
        num_questions=5
    )
    
    print(f"\n{'='*60}")
    print(f"实验: {question_set.experiment_title}")
    print(f"摘要: {question_set.summary}")
    print(f"{'='*60}\n")
    
    for q in question_set.questions:
        print(f"【问题{q.id} | {q.category}】")
        print(q.question)
        if q.rationale:
            print(f"\n💡 出题理由: {q.rationale}")
        print("-" * 60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_proposer()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)