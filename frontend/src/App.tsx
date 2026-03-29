/**
 * Deep Understanding Assessment Frontend
 * 对应后端: ~/thesis_2026-gsk/api/main.py
 * 
 * 技术栈: React + TypeScript + Tailwind CSS
 * 文件路径: ~/thesis_2026-gsk/frontend/src/App.tsx (或 pages/Evaluation.tsx)
 */

import React, { useState, useEffect } from 'react';

// ==================== 类型定义（对应后端Pydantic模型） ====================

interface StartEvaluationRequest {
  original_question: string;
  original_answer: string;
  student_id?: string;
  subject?: string;
}

interface ExamQuestionsResponse {
  evaluation_id: string;
  status: string;
  original_question: string;
  exam_questions: Array<{ id: string; text: string }>;
  question_count: number;
  generated_at: string;
}

interface SubmitExamAnswersRequest {
  exam_answers: Record<string, string>;
}

interface QuestionScoreDetail {
  question_id: string;
  question_text: string;
  student_answer: string;
  final_score: number;
  grade: string;
  confidence: string;
  chairman_feedback: string;
  teacher_scores: Array<{ model: string; score: number }>;
  consensus_stats: {
    agreement_level?: string;
    most_strict_teacher?: string;
    most_lenient_teacher?: string;
    [key: string]: any;
  };
}

interface OverallAssessment {
  understanding_level: string;
  confidence: number;
  reasoning: string;
  knowledge_gaps: string[];
  recommendations: string[];
}

interface FinalEvaluationResult {
  evaluation_id: string;
  status: string;
  original_question: string;
  original_answer: string;
  exam_question_count: number;
  exam_scores: QuestionScoreDetail[];
  overall_assessment: OverallAssessment;
  generated_at: string;
}

// ==================== API客户端 ====================

const API_BASE_URL = 'http://localhost:8000';

async function startEvaluation(data: StartEvaluationRequest): Promise<ExamQuestionsResponse> {
  const res = await fetch(`${API_BASE_URL}/api/evaluation/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function completeEvaluation(
  evaluationId: string, 
  data: SubmitExamAnswersRequest
): Promise<FinalEvaluationResult> {
  const res = await fetch(`${API_BASE_URL}/api/evaluation/${evaluationId}/complete`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ==================== 工具函数 ====================

const getScoreColor = (score: number): string => {
  if (score >= 9) return 'bg-green-500';
  if (score >= 7) return 'bg-blue-500';
  if (score >= 5) return 'bg-yellow-500';
  if (score >= 3) return 'bg-orange-500';
  return 'bg-red-500';
};

const getScoreTextColor = (score: number): string => {
  if (score >= 9) return 'text-green-700';
  if (score >= 7) return 'text-blue-700';
  if (score >= 5) return 'text-yellow-700';
  if (score >= 3) return 'text-orange-700';
  return 'text-red-700';
};

// ==================== 组件 ====================

// 阶段1: 提交原始问题与答案
const StartPhase: React.FC<{
  onStart: (data: StartEvaluationRequest) => void;
  loading: boolean;
}> = ({ onStart, loading }) => {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [studentId, setStudentId] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onStart({ original_question: question, original_answer: answer, student_id: studentId || undefined });
  };

  return (
    <div className="max-w-3xl mx-auto p-6 bg-white rounded-lg shadow-md">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">AI考试系统</h2>
        <p className="text-gray-600">
          使用Agent根据作业出题并自动评分
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">原始作业</label>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="例如：给定两个字符串形式输入的整数，这两个整数可能非常长，请写python代码完成一个函数，实现对两者乘积的计算并返回为字符串。"
            className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent h-24"
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">提交的结果</label>
          <textarea
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            placeholder="在此输入您对该问题的详细回答..."
            className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent h-48"
            required
          />
          <p className="text-xs text-gray-500 mt-1">
            *此答案将用于生成针对性的测试问题，但不会直接被打分
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">学生ID</label>
          <input
            type="text"
            value={studentId}
            onChange={(e) => setStudentId(e.target.value)}
            placeholder="例如：student_001"
            className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <button
          type="submit"
          disabled={loading || !question.trim() || !answer.trim()}
          className="w-full bg-blue-600 text-white py-3 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors font-medium"
        >
          {loading ? '生成测试问题中...' : '提交并生成测试问题'}
        </button>
      </form>
    </div>
  );
};

// 阶段2: 回答考官生成的深度测试问题
const ExamPhase: React.FC<{
  evaluationId: string;
  questions: ExamQuestionsResponse['exam_questions'];
  originalQuestion: string;
  onSubmit: (answers: Record<string, string>) => void;
  loading: boolean;
}> = ({ evaluationId, questions, originalQuestion, onSubmit, loading }) => {
  const [answers, setAnswers] = useState<Record<string, string>>({});

  const handleSubmit = () => {
    // 检查是否全部回答
    const unanswered = questions.filter(q => !answers[q.id]?.trim());
    if (unanswered.length > 0) {
      alert(`请回答所有问题。还剩 ${unanswered.length} 个未回答。`);
      return;
    }
    onSubmit(answers);
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="bg-blue-50 border-l-4 border-blue-500 p-4 mb-6 rounded">
        <h3 className="font-bold text-blue-900">测试阶段</h3>
        <p className="text-sm text-blue-800 mt-1">
          基于您对原始问题的回答，考官AI生成了以下{questions.length}个针对性问题。
          这些问题将测试您对答案中原理的真正理解。
        </p>
        <p className="text-xs text-blue-600 mt-2">
          原始问题: {originalQuestion.substring(0, 100)}...
        </p>
      </div>

      <div className="space-y-6">
        {questions.map((q, idx) => (
          <div key={q.id} className="bg-white rounded-lg shadow-md p-6 border border-gray-200">
            <div className="flex items-start mb-4">
              <span className="flex-shrink-0 w-8 h-8 bg-gray-800 text-white rounded-full flex items-center justify-center font-bold text-sm mr-3">
                {idx + 1}
              </span>
              <div className="flex-1">
                <h4 className="text-lg font-semibold text-gray-800 mb-2">{q.text}</h4>
                <p className="text-xs text-gray-500 mb-3">问题ID: {q.id}</p>
              </div>
            </div>
            
            <textarea
              value={answers[q.id] || ''}
              onChange={(e) => setAnswers(prev => ({ ...prev, [q.id]: e.target.value }))}
              placeholder="请详细回答此问题，展示您对相关原理的理解..."
              className="w-full p-4 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 h-32 resize-y"
            />
          </div>
        ))}
      </div>

      <div className="mt-8 flex justify-between items-center">
        <div className="text-sm text-gray-600">
          已完成: {Object.values(answers).filter(v => v.trim()).length} / {questions.length}
        </div>
        <button
          onClick={handleSubmit}
          disabled={loading}
          className="bg-green-600 text-white px-8 py-3 rounded-md hover:bg-green-700 disabled:bg-gray-400 font-medium text-lg transition-colors"
        >
          {loading ? 'Agent评分中（约需数分钟）...' : '提交答案进行Agent自动评估'}
        </button>
      </div>

      {loading && (
        <div className="mt-6 bg-gray-50 rounded-lg p-4 border border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
            <span className="text-gray-700">正在进行三阶段Agent评分...</span>
          </div>
          <div className="mt-2 text-sm text-gray-600 space-y-1 ml-8">
            <p>Stage 1: 多教师独立评分</p>
            <p>Stage 2: 同行评议与校准</p>
            <p>Stage 3: 主席综合裁定与评估</p>
          </div>
        </div>
      )}
    </div>
  );
};

// 阶段3: 展示委员会评分结果（核心组件）
const ResultPhase: React.FC<{ result: FinalEvaluationResult }> = ({ result }) => {
  const [expandedQuestion, setExpandedQuestion] = useState<string | null>(null);

  const avgScore = result.exam_scores.reduce((sum, s) => sum + s.final_score, 0) / result.exam_scores.length;

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-6">
      {/* 头部：主席综合评估 */}
      <div className="bg-gradient-to-r from-gray-900 to-gray-800 text-white rounded-xl shadow-lg p-6">
        <div className="flex justify-between items-start mb-4">
          <div>
            <h2 className="text-2xl font-bold">评估Agent最终裁定</h2>
            <p className="text-gray-300 text-sm mt-1">Evaluation ID: {result.evaluation_id}</p>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold">{avgScore.toFixed(1)}/10</div>
            <div className="text-sm text-gray-400">考官问题平均分</div>
          </div>
        </div>

        <div className="bg-white/10 rounded-lg p-4 backdrop-blur-sm">
          <div className="flex items-center mb-3">
            <span className="px-3 py-1 bg-blue-500 rounded-full text-xs font-semibold mr-3">
              主席评估
            </span>
            <span className="text-lg font-semibold">{result.overall_assessment.understanding_level}</span>
            <span className="ml-3 text-sm text-gray-300">
              (置信度: {(result.overall_assessment.confidence * 100).toFixed(0)}%)
            </span>
          </div>
          <p className="text-gray-200 leading-relaxed mb-4">
            {result.overall_assessment.reasoning}
          </p>
          
          {result.overall_assessment.knowledge_gaps.length > 0 && (
            <div className="mb-4">
              <h4 className="text-sm font-semibold text-red-300 mb-2">识别出的知识漏洞:</h4>
              <div className="flex flex-wrap gap-2">
                {result.overall_assessment.knowledge_gaps.map((gap, idx) => (
                  <span key={idx} className="px-2 py-1 bg-red-500/20 border border-red-500/30 rounded text-sm text-red-200">
                    {gap}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div>
            <h4 className="text-sm font-semibold text-green-300 mb-2">学习建议:</h4>
            <ul className="space-y-1">
              {result.overall_assessment.recommendations.map((rec, idx) => (
                <li key={idx} className="text-sm text-gray-300 flex items-start">
                  <span className="mr-2">•</span>
                  {rec}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      {/* 各考官问题详细评分 */}
      <div className="bg-white rounded-xl shadow-md border border-gray-200 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 bg-gray-50">
          <h3 className="text-lg font-bold text-gray-800">各维度详细评分（三阶段委员会流程）</h3>
          <p className="text-sm text-gray-600 mt-1">
            每个问题经过多教师独立评分 → 同行评议校准 → 主席综合裁定
          </p>
        </div>

        <div className="divide-y divide-gray-200">
          {result.exam_scores.map((score, idx) => (
            <div key={score.question_id} className="p-6 hover:bg-gray-50 transition-colors">
              <div 
                className="cursor-pointer"
                onClick={() => setExpandedQuestion(expandedQuestion === score.question_id ? null : score.question_id)}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-3">
                    <span className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center font-bold text-gray-700">
                      {idx + 1}
                    </span>
                    <h4 className="font-semibold text-gray-800">{score.question_text.substring(0, 60)}...</h4>
                  </div>
                  <div className="flex items-center space-x-4">
                    <div className={`text-2xl font-bold ${getScoreTextColor(score.final_score)}`}>
                      {score.final_score.toFixed(1)}
                    </div>
                    <span className={`px-3 py-1 rounded-full text-sm font-medium text-white ${getScoreColor(score.final_score)}`}>
                      {score.grade}
                    </span>
                    <svg 
                      className={`w-5 h-5 text-gray-400 transform transition-transform ${expandedQuestion === score.question_id ? 'rotate-180' : ''}`} 
                      fill="none" 
                      stroke="currentColor" 
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </div>
                </div>
                
                <div className="flex items-center space-x-4 text-sm text-gray-600 ml-11">
                  <span>置信度: <span className={`font-medium ${score.confidence === '高' ? 'text-green-600' : score.confidence === '中' ? 'text-yellow-600' : 'text-red-600'}`}>{score.confidence}</span></span>
                  <span>教师数: {score.teacher_scores.length}</span>
                  <span>一致性: {score.consensus_stats.agreement_level || '未知'}</span>
                </div>
              </div>

              {/* 展开详情：三阶段评分细节 */}
              {expandedQuestion === score.question_id && (
                <div className="mt-4 ml-11 pl-4 border-l-2 border-blue-200 space-y-4 animate-fade-in">
                  {/* Stage 1: 教师独立评分 */}
                  <div className="bg-blue-50 rounded-lg p-4">
                    <h5 className="font-semibold text-blue-900 mb-2 flex items-center">
                      <span className="w-6 h-6 bg-blue-600 text-white rounded text-xs flex items-center justify-center mr-2">1</span>
                      教师委员会独立评分
                    </h5>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                      {score.teacher_scores.map((teacher, tidx) => (
                        <div key={tidx} className="bg-white rounded p-2 border border-blue-200">
                          <div className="text-xs text-gray-500 truncate">{teacher.model}</div>
                          <div className={`text-lg font-bold ${getScoreTextColor(teacher.score || 0)}`}>
                            {teacher.score?.toFixed(1) || 'N/A'}/10
                          </div>
                        </div>
                      ))}
                    </div>
                    {score.consensus_stats.most_strict_teacher && (
                      <div className="mt-2 text-xs text-gray-600">
                        最严格: {score.consensus_stats.most_strict_teacher} | 
                        最宽松: {score.consensus_stats.most_lenient_teacher}
                      </div>
                    )}
                  </div>

                  {/* Stage 3: 主席评语 */}
                  <div className="bg-purple-50 rounded-lg p-4">
                    <h5 className="font-semibold text-purple-900 mb-2 flex items-center">
                      <span className="w-6 h-6 bg-purple-600 text-white rounded text-xs flex items-center justify-center mr-2">3</span>
                      委员会主席详细评语
                    </h5>
                    <div className="text-sm text-gray-800 leading-relaxed whitespace-pre-wrap bg-white p-3 rounded border border-purple-200">
                      {score.chairman_feedback}
                    </div>
                  </div>

                  {/* 学生回答预览 */}
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h5 className="font-semibold text-gray-700 mb-2">学生回答预览</h5>
                    <div className="text-sm text-gray-600 bg-white p-3 rounded border border-gray-200 max-h-32 overflow-y-auto">
                      {score.student_answer}
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* 统计摘要 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg shadow p-4 border border-gray-200">
          <h4 className="text-sm font-semibold text-gray-600 mb-2">评分分布</h4>
          <div className="space-y-2">
            {[
              { label: '优秀 (9-10)', count: result.exam_scores.filter(s => s.final_score >= 9).length, color: 'bg-green-500' },
              { label: '良好 (7-8)', count: result.exam_scores.filter(s => s.final_score >= 7 && s.final_score < 9).length, color: 'bg-blue-500' },
              { label: '中等 (5-6)', count: result.exam_scores.filter(s => s.final_score >= 5 && s.final_score < 7).length, color: 'bg-yellow-500' },
              { label: '待改进 (<5)', count: result.exam_scores.filter(s => s.final_score < 5).length, color: 'bg-red-500' },
            ].map((item, idx) => (
              <div key={idx} className="flex items-center text-sm">
                <span className="w-20 text-gray-600">{item.label}</span>
                <div className="flex-1 mx-2 bg-gray-200 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full ${item.color}`} 
                    style={{ width: `${(item.count / result.exam_scores.length) * 100}%` }}
                  ></div>
                </div>
                <span className="w-8 text-right font-medium">{item.count}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-4 border border-gray-200 md:col-span-2">
          <h4 className="text-sm font-semibold text-gray-600 mb-3">评估流程说明</h4>
          <div className="flex justify-between items-center text-xs text-gray-500">
            <div className="flex-1 text-center">
              <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-2 text-blue-600 font-bold">1</div>
              <p>多教师<br/>独立评分</p>
            </div>
            <div className="w-8 h-px bg-gray-300"></div>
            <div className="flex-1 text-center">
              <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-2 text-green-600 font-bold">2</div>
              <p>同行评议<br/>交叉校准</p>
            </div>
            <div className="w-8 h-px bg-gray-300"></div>
            <div className="flex-1 text-center">
              <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-2 text-purple-600 font-bold">3</div>
              <p>主席综合<br/>最终裁定</p>
            </div>
          </div>
        </div>
      </div>

      <div className="flex justify-center pt-6">
        <button 
          onClick={() => window.location.reload()}
          className="bg-gray-800 text-white px-6 py-3 rounded-md hover:bg-gray-900 transition-colors"
        >
          开始新的评估
        </button>
      </div>
    </div>
  );
};

// ==================== 主应用组件 ====================

const ExamEvaluationApp: React.FC = () => {
  const [phase, setPhase] = useState<'start' | 'exam' | 'result'>('start');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // 状态数据
  const [evaluationId, setEvaluationId] = useState<string>('');
  const [examQuestions, setExamQuestions] = useState<ExamQuestionsResponse | null>(null);
  const [finalResult, setFinalResult] = useState<FinalEvaluationResult | null>(null);

  const handleStart = async (data: StartEvaluationRequest) => {
    setLoading(true);
    setError(null);
    try {
      const response = await startEvaluation(data);
      setEvaluationId(response.evaluation_id);
      setExamQuestions(response);
      setPhase('exam');
    } catch (err: any) {
      setError(`启动评估失败: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitExam = async (answers: Record<string, string>) => {
    setLoading(true);
    setError(null);
    try {
      const result = await completeEvaluation(evaluationId, { exam_answers: answers });
      setFinalResult(result);
      setPhase('result');
    } catch (err: any) {
      setError(`提交评分失败: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 py-8">
      <div className="container mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">LLM委员会考试系统</h1>
          <p className="text-gray-600">基于多Agent的自动考试</p>
        </div>

        {/* Error Display */}
        {error && (
          <div className="max-w-3xl mx-auto mb-6 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
            <strong className="font-bold">错误:</strong> {error}
          </div>
        )}

        {/* Phase Routing */}
        {phase === 'start' && (
          <StartPhase onStart={handleStart} loading={loading} />
        )}

        {phase === 'exam' && examQuestions && (
          <ExamPhase
            evaluationId={evaluationId}
            questions={examQuestions.exam_questions}
            originalQuestion={examQuestions.original_question}
            onSubmit={handleSubmitExam}
            loading={loading}
          />
        )}

        {phase === 'result' && finalResult && (
          <ResultPhase result={finalResult} />
        )}
      </div>
    </div>
  );
};

export default ExamEvaluationApp;
