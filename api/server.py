"""FastAPI Backend for Deep Understanding Assessment System
所有评价逻辑交由LLM委员会处理，代码中无硬编码评价规则

文件路径: ~/thesis_2026-gsk/api/server.py
"""

import os
import sys
import uuid
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from voice_oral_exam import VoiceOralExamSession, oral_sessions, handle_oral_exam_ws
from oral_exam_engine import OralExamState, OralExaminer

# 添加路径以导入两个模块
sys.path.append(os.path.expanduser("/home/gsk/thesis_2026-gsk/questions"))

backend_path = os.path.expanduser("/home/gsk/thesis_2026-gsk/llm-council/backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# 模拟 backend 是一个包，创建 __init__.py 的虚拟模块
import types
backend_module = types.ModuleType("backend")
backend_module.__path__ = [backend_path]
sys.modules["backend"] = backend_module

from question_proposer import get_questions, AgentResponse

# ========== 导入 OS 实验提问器 ==========
from os_proposer import (
    CodeExtractor,
    DiffAnalyzer,
    OSExperimentAnalyzer,
    QuestionProposer as OSQuestionProposer,
)

from backend.main import (
    run_grading_council,
    stage1_teacher_scoring,
    stage2_peer_review,
    stage3_chairman_final_grade,
    calculate_scoring_consensus,
)

from backend.kimi import query_model

# 从config导入主席模型配置（根据实际路径调整）
from backend.config import CHAIRMAN_MODEL

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 数据模型 ====================

class StartEvaluationRequest(BaseModel):
    """启动评估请求"""
    original_question: str = Field(..., min_length=1, description="原始考题")
    original_answer: str = Field(..., min_length=1, description="学生提交的原始答案（仅用于生成测试问题）")
    student_id: Optional[str] = Field(default=None, description="学生标识（可选）")
    subject: Optional[str] = Field(default="general", description="学科领域")


class ExamQuestionsResponse(BaseModel):
    """返回生成的深度测试问题"""
    evaluation_id: str
    status: str = "questions_generated"
    original_question: str
    exam_questions: List[Dict[str, str]] = Field(..., description="考官生成的深度测试问题列表（带ID）")
    question_count: int
    generated_at: str


class SubmitExamAnswersRequest(BaseModel):
    """提交对考官问题的回答进行评分"""
    exam_answers: Dict[str, str] = Field(..., description="考官问题ID与回答的映射")


class QuestionScoreDetail(BaseModel):
    """单个考官问题的委员会评分结果（来自三阶段流程）"""
    question_id: str
    question_text: str
    student_answer: str
    final_score: float = Field(..., description="主席最终裁定分数 0-10")
    grade: str = Field(..., description="等级 A+/A/B+/B/C/D")
    confidence: str = Field(..., description="评分置信度 高/中/低")
    chairman_feedback: str = Field(..., description="委员会主席对该问题的详细评语")
    teacher_scores: List[Dict] = Field(..., description="各教师模型独立评分列表")
    consensus_stats: Dict[str, Any] = Field(..., description="该问题的评分一致性统计")


class OverallAssessment(BaseModel):
    """主席模型对学生整体理解程度的综合评估（由LLM生成，非代码硬编码）"""
    understanding_level: str = Field(..., description="理解程度判定（如：深入理解/部分理解等）")
    confidence: float = Field(..., description="评估置信度 0-1")
    reasoning: str = Field(..., description="评估理由（主席模型的详细分析）")
    knowledge_gaps: List[str] = Field(default=[], description="识别出的知识漏洞")
    recommendations: List[str] = Field(default=[], description="学习建议")


class FinalEvaluationResult(BaseModel):
    """最终评估结果"""
    evaluation_id: str
    status: str = "completed"
    
    # 原始内容（仅展示，无评分）
    original_question: str
    original_answer: str
    
    # 考官问题评分（三阶段委员会评分结果）
    exam_question_count: int
    exam_scores: List[QuestionScoreDetail]
    
    # 主席综合评估（由LLM委员会生成）
    overall_assessment: OverallAssessment
    
    generated_at: str


# ==================== 内存存储 ====================

class EvaluationStorage:
    def __init__(self):
        self._store: Dict[str, Dict] = {}
    
    def create(self, data: Dict) -> str:
        eval_id = str(uuid.uuid4())
        data.update({
            "evaluation_id": eval_id,
            "created_at": datetime.now().isoformat(),
            "status": "pending"
        })
        self._store[eval_id] = data
        return eval_id
    
    def get(self, eval_id: str) -> Optional[Dict]:
        return self._store.get(eval_id)
    
    def update(self, eval_id: str, data: Dict):
        if eval_id in self._store:
            self._store[eval_id].update(data)
            self._store[eval_id]["updated_at"] = datetime.now().isoformat()

storage = EvaluationStorage()


# ==================== OS 实验提问器单例 ====================

_os_proposer_instance: Optional[OSQuestionProposer] = None

def get_os_proposer() -> OSQuestionProposer:
    """获取或初始化 OS QuestionProposer 单例"""
    global _os_proposer_instance
    if _os_proposer_instance is None:
        _os_proposer_instance = OSQuestionProposer()
    return _os_proposer_instance


# ==================== FastAPI 应用 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Deep Understanding Assessment API (LLM Council Evaluation) 启动")
    yield
    logger.info("🛑 API 关闭")

app = FastAPI(
    title="深度理解评估系统（LLM委员会评价版）",
    description="所有评价逻辑由LLM委员会主席裁定，代码中无硬编码规则",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 辅助函数 ====================

async def generate_exam_questions_async(question: str, answer: str) -> AgentResponse:
    """异步包装同步的get_questions调用"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: get_questions(question, answer, use_kimi=True)
    )


def parse_exam_questions(agent_response: AgentResponse) -> List[str]:
    """从Agent响应中解析出问题列表"""
    content = agent_response.content
    lines = content.strip().split('\n')
    questions = []
    
    for line in lines:
        line = line.strip()
        if line and ('?' in line or '？' in line or 
                    any(line.startswith(str(i)) for i in range(1, 10))):
            cleaned = line
            for prefix in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', 
                          '1、', '2、', '3、', '4、', '5、', '6、', '7、', '8、', '9、',
                          '- ', '• ']:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
                    break
            if cleaned and len(cleaned) > 5:
                questions.append(cleaned)
    
    return questions if questions else [content]


async def chairman_overall_assessment(
    original_question: str,
    original_answer: str,
    exam_results: List[Dict]
) -> OverallAssessment:
    """
    由LLM委员会主席综合所有考官问题的评分，给出整体评估。
    所有评价逻辑（理解程度、知识漏洞、学习建议）均由主席模型生成。
    """
    
    # 构建所有考官问题的评分摘要
    exam_summary = []
    for i, result in enumerate(exam_results, 1):
        stage3 = result.get("stage3", {})
        exam_summary.append({
            "question_id": f"考官问题{i}",
            "question_text": result.get("question_text", "")[:100],
            "final_score": stage3.get("final_score"),
            "grade": stage3.get("grade"),
            "key_feedback": stage3.get("response", "")[:300]  # 主席对该问题的评语
        })
    
    # 计算基础统计（仅作为事实提供，不用于硬编码判断）
    scores = [e["final_score"] for e in exam_summary if e["final_score"] is not None]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    # 构建主席综合评估Prompt
    assessment_prompt = f"""你是一位资深的教育评估委员会主席。你已经收到了评估委员会对学生在多个深度测试问题上的表现评分。

【背景信息】
原始题目：{original_question}
学生原始答案摘要：{original_answer[:200]}...（注：原始答案未经独立评分，仅作参考）

【各深度测试问题的评分结果】
{json.dumps(exam_summary, ensure_ascii=False, indent=2)}

评分统计：
- 参与评分的问题数：{len(exam_summary)}
- 分数分布：{scores}
- 平均分：{avg_score:.1f}/10

【你的任务】
作为委员会主席，请综合以上所有考官问题的评分结果，给出以下评估：

1. **整体理解程度判定**：基于学生在深度测试问题上的综合表现，判定其对原始知识点的真实理解水平（例如：深入理解并能灵活运用/理解核心概念但应用有局限/仅掌握表面知识/存在明显误解/基础薄弱需要重新学习等）。请给出具体判定并解释理由。

2. **知识漏洞识别**：指出学生在哪些具体概念、原理或应用层面存在不足（基于低分问题的反馈）。

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

    messages = [{"role": "user", "content": assessment_prompt}]
    
    try:
        # 调用主席模型
        response = await query_model(CHAIRMAN_MODEL, messages)
        content = response.get('content', '') if response else ''
        
        # 尝试解析JSON
        # 清理可能的markdown代码块
        clean_content = content.replace('```json', '').replace('```', '').strip()
        assessment_data = json.loads(clean_content)
        
        return OverallAssessment(
            understanding_level=assessment_data.get("understanding_level", "评估失败"),
            confidence=float(assessment_data.get("confidence", 0)),
            reasoning=assessment_data.get("reasoning", ""),
            knowledge_gaps=assessment_data.get("knowledge_gaps", []),
            recommendations=assessment_data.get("recommendations", [])
        )
        
    except Exception as e:
        logger.error(f"主席综合评估失败: {e}, 原始响应: {content[:200]}")
        # 如果解析失败，返回一个基于原始描述的fallback（但这不是硬编码评价，只是格式错误时的保底）
        return OverallAssessment(
            understanding_level="评估生成异常",
            confidence=0.0,
            reasoning=f"评估生成过程出现错误: {str(e)}，请查看各考官问题的单独评分。",
            knowledge_gaps=[],
            recommendations=["请查看详细评分数据"]
        )


# ==================== API 端点 ====================

@app.get("/")
async def root():
    return {
        "service": "深度理解评估系统（LLM委员会全权评价）",
        "version": "3.0.0",
        "evaluation_method": "所有评价（理解程度、知识漏洞、建议）均由LLM委员会主席生成",
        "code_logic": "无硬编码评价规则，完全依赖委员会裁定",
        "endpoints": {
            "start": "POST /api/evaluation/start - 提交原始Q&A，生成深度测试",
            "submit": "POST /api/evaluation/{id}/complete - 提交考官回答，由委员会评价",
            "status": "GET /api/evaluation/{id} - 查询评估状态",
            "os_start": "POST /api/os-experiment/start - OS实验模式：上传代码zip生成问题",
            "os_diff": "POST /api/os-experiment/diff - OS实验模式：仅分析代码差异"
        }
    }


@app.post("/api/evaluation/start", response_model=ExamQuestionsResponse)
async def start_evaluation(request: StartEvaluationRequest):
    """第一阶段：提交原始问题与答案，生成考官深度测试问题（原有模式）"""
    try:
        logger.info(f"开始评估流程 - 学生: {request.student_id or 'anonymous'}")
        
        eval_id = storage.create({
            "original_question": request.original_question,
            "original_answer": request.original_answer,
            "student_id": request.student_id,
            "subject": request.subject
        })
        
        storage.update(eval_id, {"status": "generating_questions"})
        agent_resp = await generate_exam_questions_async(
            request.original_question, 
            request.original_answer
        )
        
        exam_questions = parse_exam_questions(agent_resp)
        questions_with_id = [{"id": f"q{i+1}", "text": q} for i, q in enumerate(exam_questions)]
        
        storage.update(eval_id, {
            "status": "questions_generated",
            "exam_questions": questions_with_id,
            "agent_raw_response": agent_resp.content
        })
        
        logger.info(f"评估 {eval_id}: 生成 {len(exam_questions)} 个考官问题")
        
        return ExamQuestionsResponse(
            evaluation_id=eval_id,
            status="questions_generated",
            original_question=request.original_question,
            exam_questions=questions_with_id,
            question_count=len(questions_with_id),
            generated_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"生成考官问题失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成深度测试问题失败: {str(e)}")


# ==================== OS 实验模式 API ====================

@app.post("/api/os-experiment/start", response_model=ExamQuestionsResponse)
async def os_experiment_start(
    experiment_requirement: str = Form(..., description="实验要求描述"),
    before_zip: UploadFile = File(..., description="修改前的代码zip"),
    after_zip: UploadFile = File(..., description="修改后的代码zip"),
    num_questions: int = Form(5, description="生成问题数量"),
    student_id: Optional[str] = Form(None, description="学生ID")
):
    """
    OS实验模式：接收修改前后的代码zip，分析差异并生成面试问题。
    返回格式与 /api/evaluation/start 兼容，前端可直接复用 ExamPhase。
    """
    # ========== 诊断日志：请求到达 ==========
    logger.info("=" * 60)
    logger.info("[DIAG] /api/os-experiment/start 请求到达")
    logger.info(f"[DIAG] experiment_requirement={experiment_requirement[:50]}...")
    logger.info(f"[DIAG] before_zip filename={before_zip.filename}, content_type={before_zip.content_type}")
    logger.info(f"[DIAG] after_zip filename={after_zip.filename}, content_type={after_zip.content_type}")
    logger.info(f"[DIAG] num_questions={num_questions}, student_id={student_id}")
    logger.info("=" * 60)
    
    try:
        logger.info(f"OS实验模式启动 - 学生: {student_id or 'anonymous'}")
        
        # 1. 读取zip文件
        logger.info("[DIAG] 步骤1: 读取zip字节...")
        before_bytes = await before_zip.read()
        after_bytes = await after_zip.read()
        logger.info(f"[DIAG] before_bytes={len(before_bytes)} bytes, after_bytes={len(after_bytes)} bytes")
        
        # 2. 提取代码
        logger.info("[DIAG] 步骤2: 提取代码...")
        before_code = CodeExtractor.extract_from_zip(before_bytes)
        after_code = CodeExtractor.extract_from_zip(after_bytes)
        logger.info(f"[DIAG] before_code files={list(before_code.keys())}")
        logger.info(f"[DIAG] after_code files={list(after_code.keys())}")
        
        if not before_code and not after_code:
            logger.error("[DIAG] 未能从zip中提取到任何代码文件")
            raise HTTPException(status_code=400, detail="未能从zip中提取到任何代码文件")
        
        # 3. 分析差异
        logger.info("[DIAG] 步骤3: 分析差异...")
        diff_report = DiffAnalyzer.analyze(before_code, after_code)
        logger.info(f"[DIAG] diff_report.summary={diff_report.summary.replace(chr(10), ' | ')}")
        logger.info(f"[DIAG] added={len(diff_report.added_files)}, deleted={len(diff_report.deleted_files)}, modified={len(diff_report.modified_files)}")
        
        # 4. 生成问题（异步包装同步调用）
        logger.info("[DIAG] 步骤4: 调用LLM生成问题...")
        proposer = get_os_proposer()
        logger.info(f"[DIAG] proposer model={proposer.model}, api_key 前8位={proposer.api_key[:8] if proposer.api_key else 'None'}...")
        
        loop = asyncio.get_event_loop()
        question_set = await loop.run_in_executor(
            None,
            lambda: proposer.generate_questions(
                experiment_requirement=experiment_requirement,
                before_code=before_code,
                after_code=after_code,
                diff_report=diff_report,
                num_questions=num_questions
            )
        )
        
        logger.info(f"[DIAG] question_set generated: {len(question_set.questions)} questions")
        for q in question_set.questions:
            logger.info(f"[DIAG] Q{q.id}[{q.category}]: {q.question[:60]}...")
        
        # 5. 构建与原有API兼容的问题列表
        questions_with_id = [
            {"id": str(q.id), "text": f"[{q.category.upper()}] {q.question}"}
            for q in question_set.questions
        ]
        logger.info(f"[DIAG] questions_with_id count={len(questions_with_id)}")
        
        # 6. 存入存储（复用原有评分流程）
        eval_id = storage.create({
            "original_question": experiment_requirement,
            "original_answer": f"[OS实验] 修改文件: {len(diff_report.modified_files)}个, 新增: {len(diff_report.added_files)}个",
            "student_id": student_id,
            "subject": "os_experiment",
            "exam_type": "os_experiment",
            "diff_summary": diff_report.summary,
            "key_changes": OSExperimentAnalyzer.identify_key_changes(diff_report, after_code)
        })
        
        storage.update(eval_id, {
            "status": "questions_generated",
            "exam_questions": questions_with_id,
            "diff_report": {
                "added_files": diff_report.added_files,
                "deleted_files": diff_report.deleted_files,
                "modified_files": [
                    {
                        "path": d.file_path,
                        "changed_lines_new": len(d.changed_lines_new),
                        "changed_lines_old": len(d.changed_lines_old)
                    }
                    for d in diff_report.modified_files
                ],
                "summary": diff_report.summary
            }
        })
        
        logger.info(f"OS实验 {eval_id}: 生成 {len(questions_with_id)} 个考官问题")
        
        return ExamQuestionsResponse(
            evaluation_id=eval_id,
            status="questions_generated",
            original_question=experiment_requirement[:200],
            exam_questions=questions_with_id,
            question_count=len(questions_with_id),
            generated_at=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"OS实验配置错误: {e}")
        raise HTTPException(status_code=500, detail=f"配置错误: {str(e)}")
    except Exception as e:
        logger.error(f"OS实验生成问题失败: {e}")
        import traceback
        logger.error(f"[DIAG] 完整异常堆栈:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"OS实验评估失败: {str(e)}")


@app.post("/api/os-experiment/diff")
async def os_experiment_diff(
    before_zip: UploadFile = File(..., description="修改前的代码zip"),
    after_zip: UploadFile = File(..., description="修改后的代码zip")
):
    """
    OS实验模式：仅分析代码差异，不生成问题。
    """
    logger.info("[DIAG] /api/os-experiment/diff 请求到达")
    
    try:
        before_bytes = await before_zip.read()
        after_bytes = await after_zip.read()
        logger.info(f"[DIAG] before={len(before_bytes)}B, after={len(after_bytes)}B")
        
        before_code = CodeExtractor.extract_from_zip(before_bytes)
        after_code = CodeExtractor.extract_from_zip(after_bytes)
        logger.info(f"[DIAG] before files={list(before_code.keys())}, after files={list(after_code.keys())}")
        
        diff_report = DiffAnalyzer.analyze(before_code, after_code)
        key_changes = OSExperimentAnalyzer.identify_key_changes(diff_report, after_code)
        
        logger.info(f"[DIAG] diff summary: {diff_report.summary.replace(chr(10), ' | ')}")
        
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
        logger.error(f"OS差异分析失败: {e}")
        import traceback
        logger.error(f"[DIAG] 完整异常堆栈:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"差异分析失败: {str(e)}")


# ==================== 原有评分流程（两种模式共用）====================

@app.post("/api/evaluation/{evaluation_id}/complete", response_model=FinalEvaluationResult)
async def complete_evaluation(evaluation_id: str, request: SubmitExamAnswersRequest):
    """
    第二阶段：提交考官问题回答，由LLM委员会进行完整评分。
    原有模式和OS实验模式共用此接口。
    """
    eval_data = storage.get(evaluation_id)
    if not eval_data:
        raise HTTPException(status_code=404, detail="评估会话未找到")
    
    if eval_data["status"] not in ["questions_generated"]:
        raise HTTPException(status_code=400, detail="评估状态不正确")
    
    try:
        storage.update(evaluation_id, {"status": "grading"})
        
        exam_questions = eval_data.get("exam_questions", [])
        if not exam_questions:
            raise HTTPException(status_code=400, detail="没有找到考官问题")
        
        # 并行对所有考官问题进行三阶段评分
        scoring_tasks = []
        for q in exam_questions:
            q_id = q["id"]
            if q_id in request.exam_answers:
                task = run_grading_council(q["text"], request.exam_answers[q_id])
                scoring_tasks.append({
                    "q_id": q_id,
                    "q_text": q["text"],
                    "answer": request.exam_answers[q_id],
                    "task": task
                })
        
        if not scoring_tasks:
            raise HTTPException(status_code=400, detail="没有提供任何考官问题的回答")
        
        logger.info(f"评估 {evaluation_id}: 开始对 {len(scoring_tasks)} 个考官问题进行三阶段委员会评分")
        
        # 收集所有评分结果
        exam_results = []
        exam_scores_details = []
        
        for item in scoring_tasks:
            stage1_results, stage2_results, stage3_result, metadata = await item["task"]
            
            result_data = {
                "question_id": item["q_id"],
                "question_text": item["q_text"],
                "student_answer": item["answer"],
                "stage1": stage1_results,
                "stage2": stage2_results,
                "stage3": stage3_result,
                "metadata": metadata
            }
            exam_results.append(result_data)
            
            detail = QuestionScoreDetail(
                question_id=item["q_id"],
                question_text=item["q_text"],
                student_answer=item["answer"],
                final_score=stage3_result.get("final_score", 0),
                grade=stage3_result.get("grade", "Unknown"),
                confidence=stage3_result.get("confidence", "中"),
                chairman_feedback=stage3_result.get("response", ""),
                teacher_scores=[
                    {"model": r["model"], "score": r.get("score")}
                    for r in stage1_results
                ],
                consensus_stats=metadata.get("consensus_stats", {})
            )
            exam_scores_details.append(detail)
        
        # 由主席模型进行综合评估
        logger.info(f"评估 {evaluation_id}: 调用主席模型生成综合评估")
        overall_assessment = await chairman_overall_assessment(
            original_question=eval_data["original_question"],
            original_answer=eval_data["original_answer"],
            exam_results=exam_results
        )
        
        # 保存结果
        storage.update(evaluation_id, {
            "status": "completed",
            "final_result": {
                "exam_scores": [s.model_dump() for s in exam_scores_details],
                "overall_assessment": overall_assessment.model_dump()
            }
        })
        
        logger.info(f"评估 {evaluation_id} 完成 - 主席判定: {overall_assessment.understanding_level}")
        
        return FinalEvaluationResult(
            evaluation_id=evaluation_id,
            status="completed",
            original_question=eval_data["original_question"],
            original_answer=eval_data["original_answer"],
            exam_question_count=len(exam_questions),
            exam_scores=exam_scores_details,
            overall_assessment=overall_assessment,
            generated_at=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        storage.update(evaluation_id, {"status": "error"})
        logger.error(f"完成评估失败 {evaluation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"评估流程失败: {str(e)}")


@app.get("/api/evaluation/{evaluation_id}")
async def get_evaluation_status(evaluation_id: str):
    """查询评估状态"""
    eval_data = storage.get(evaluation_id)
    if not eval_data:
        raise HTTPException(status_code=404, detail="评估会话未找到")
    
    response = {
        "evaluation_id": evaluation_id,
        "status": eval_data["status"],
        "created_at": eval_data["created_at"],
        "updated_at": eval_data.get("updated_at")
    }
    
    if eval_data["status"] == "completed":
        final = eval_data.get("final_result", {})
        assessment = final.get("overall_assessment", {})
        response["understanding_level"] = assessment.get("understanding_level")
        response["confidence"] = assessment.get("confidence")
    
    return response


# ============ 语音口试API ============

class OralExamStartRequest(BaseModel):
    original_question: str
    original_answer: str  # 学生的预习答案（用于考官准备问题）
    student_id: Optional[str] = None
    subject: Optional[str] = "general"

@app.post("/api/oral-exam/start")
async def start_oral_exam(request: OralExamStartRequest):
    """启动对话式语音口试"""
    try:
        eval_id = storage.create({
            "original_question": request.original_question,
            "original_answer": request.original_answer,
            "student_id": request.student_id,
            "subject": request.subject,
            "exam_type": "oral_dialogue",
            "status": "oral_exam_ready"
        })
        
        # 创建口试状态
        state = OralExamState(
            original_question=request.original_question,
            original_answer=request.original_answer,
            subject=request.subject
        )
        
        # 创建会话
        session = VoiceOralExamSession(eval_id, state)
        oral_sessions[eval_id] = session
        
        return {
            "evaluation_id": eval_id,
            "status": "ready",
            "websocket_url": f"ws://localhost:8000/ws/oral-exam/{eval_id}",
            "config": {
                "sample_rate": 24000,
                "language": "zh",
                "tts_voice": "zh-CN-XiaoxiaoNeural"
            },
            "instructions": {
                "start_command": "发送 {'type': 'start_exam'} 开始",
                "supported_commands": ["请重复", "解释一下", "给点提示", "下一题", "我答完了"],
                "format": "WebSocket JSON + 二进制音频"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/oral-exam/{evaluation_id}")
async def oral_exam_websocket(websocket: WebSocket, evaluation_id: str):
    """语音口试WebSocket"""
    await handle_oral_exam_ws(websocket, evaluation_id)

@app.get("/api/oral-exam/{evaluation_id}/dialogue")
async def get_dialogue_history(evaluation_id: str):
    """获取口试对话记录（供复盘）"""
    eval_data = storage.get(evaluation_id)
    if not eval_data:
        raise HTTPException(status_code=404, detail="评估不存在")
    
    return {
        "dialogue": eval_data.get("oral_record", {}).get("dialogue", []),
        "full_text": eval_data.get("oral_record", {}).get("dialogue_text", "")
    }


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, log_level="info")