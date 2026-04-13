/**
 * Deep Understanding Assessment Frontend
 * 整合文字考试与语音口试双模式
 * 
 * 技术栈: React + TypeScript + Tailwind CSS
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';

// ==================== 类型定义 ====================

// 文字考试类型
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

// 语音口试类型
interface OralExamStartRequest {
  original_question: string;
  original_answer: string;
  student_id?: string;
  subject?: string;
}

interface OralExamStartResponse {
  evaluation_id: string;
  status: string;
  websocket_url: string;
  config: {
    sample_rate: number;
    language: string;
    tts_voice: string;
  };
}

interface DialogueTurn {
  role: 'examiner' | 'student';
  type?: 'repeat' | 'explanation' | 'hint' | 'follow_up' | 'new_topic' | 'answer';
  text: string;
  timestamp: string;
}

interface FinalOralResult {
  evaluation_id: string;
  understanding_level: string;
  confidence: number;
  reasoning: string;
  dialogue_text: string;
  recommendations: string[];
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

async function startOralExam(data: OralExamStartRequest): Promise<OralExamStartResponse> {
  const res = await fetch(`${API_BASE_URL}/api/oral-exam/start`, {
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

// ==================== 文字考试组件 ====================

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
        <h2 className="text-2xl font-bold text-gray-800 mb-2">AI文字考试</h2>
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

const ExamPhase: React.FC<{
  evaluationId: string;
  questions: ExamQuestionsResponse['exam_questions'];
  originalQuestion: string;
  onSubmit: (answers: Record<string, string>) => void;
  loading: boolean;
}> = ({ evaluationId, questions, originalQuestion, onSubmit, loading }) => {
  const [answers, setAnswers] = useState<Record<string, string>>({});

  const handleSubmit = () => {
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

              {expandedQuestion === score.question_id && (
                <div className="mt-4 ml-11 pl-4 border-l-2 border-blue-200 space-y-4 animate-fade-in">
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

                  <div className="bg-purple-50 rounded-lg p-4">
                    <h5 className="font-semibold text-purple-900 mb-2 flex items-center">
                      <span className="w-6 h-6 bg-purple-600 text-white rounded text-xs flex items-center justify-center mr-2">3</span>
                      委员会主席详细评语
                    </h5>
                    <div className="text-sm text-gray-800 leading-relaxed whitespace-pre-wrap bg-white p-3 rounded border border-purple-200">
                      {score.chairman_feedback}
                    </div>
                  </div>

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

// ==================== 语音口试组件 ====================

type OralExamPhase = 'prepare' | 'connecting' | 'examining' | 'grading' | 'result';

const OralExamination: React.FC = () => {
  const [phase, setPhase] = useState<OralExamPhase>('prepare');
  const [originalQuestion, setOriginalQuestion] = useState('');
  const [originalAnswer, setOriginalAnswer] = useState('');
  const [studentId, setStudentId] = useState('');
  
  const [evalId, setEvalId] = useState<string>('');
  const [connectionStatus, setConnectionStatus] = useState<'idle' | 'connecting' | 'connected' | 'error'>('idle');
  
  const [isExaminerSpeaking, setIsExaminerSpeaking] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [dialogueHistory, setDialogueHistory] = useState<DialogueTurn[]>([]);
  const [lastTranscript, setLastTranscript] = useState('');
  const [currentHint, setCurrentHint] = useState<string>('');
  
  const [finalResult, setFinalResult] = useState<FinalOralResult | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioQueueRef = useRef<Blob[]>([]);
  const isPlayingRef = useRef(false);

  const playAudio = useCallback(async (audioBlob: Blob) => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
    
    const arrayBuffer = await audioBlob.arrayBuffer();
    const audioBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);
    const source = audioContextRef.current.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContextRef.current.destination);
    
    setIsExaminerSpeaking(true);
    source.onended = () => {
      setIsExaminerSpeaking(false);
      isPlayingRef.current = false;
      if (audioQueueRef.current.length > 0) {
        const next = audioQueueRef.current.shift();
        if (next) playAudio(next);
      }
    };
    
    source.start(0);
    isPlayingRef.current = true;
  }, []);

  const connectWebSocket = useCallback((url: string, id: string) => {
    setConnectionStatus('connecting');
    const ws = new WebSocket(url);
    wsRef.current = ws;
    
    ws.onopen = () => {
      setConnectionStatus('connected');
      setPhase('examining');
      ws.send(JSON.stringify({ type: 'start_exam' }));
      setCurrentHint('考官正在准备第一个问题...');
    };
    
    ws.onmessage = async (event) => {
      if (event.data instanceof Blob) {
        if (isPlayingRef.current) {
          audioQueueRef.current.push(event.data);
        } else {
          await playAudio(event.data);
        }
      } else {
        const msg = JSON.parse(event.data);
        handleWebSocketMessage(msg);
      }
    };
    
    ws.onerror = () => {
      setConnectionStatus('error');
    };
    
    ws.onclose = () => {
      setConnectionStatus('idle');
    };
  }, [playAudio]);

  const handleWebSocketMessage = (msg: any) => {
    switch (msg.type) {
      case 'listening':
        setIsProcessing(false);
        setCurrentHint('考官正在等待你的回答...');
        break;
        
      case 'processing':
        setIsProcessing(true);
        setCurrentHint('正在识别语音...');
        break;
        
      case 'transcription':
        setLastTranscript(msg.text);
        setDialogueHistory(prev => [...prev, {
          role: 'student',
          text: msg.text,
          timestamp: new Date().toISOString(),
          type: 'answer'
        }]);
        break;
        
      case 'examiner_response':
        setDialogueHistory(prev => [...prev, {
          role: 'examiner',
          type: msg.response_type,
          text: '[语音内容]',
          timestamp: new Date().toISOString()
        }]);
        setCurrentHint(
          msg.response_type === 'repeat' ? '考官正在重复问题' : 
          msg.response_type === 'explanation' ? '考官正在解释' :
          msg.response_type === 'hint' ? '考官正在提示' : '考官正在提问'
        );
        break;
        
      case 'audio_start':
        setIsExaminerSpeaking(true);
        break;
        
      case 'audio_end':
        break;
        
      case 'exam_complete':
        setPhase('grading');
        setCurrentHint('口试结束，正在生成评估报告...');
        setTimeout(() => fetchFinalResult(evalId), 3000);
        break;
        
      case 'error':
        alert(`错误: ${msg.message}`);
        break;
    }
  };

  const startRecording = async () => {
    if (isExaminerSpeaking || isProcessing) return;
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };
      
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        sendAudio(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorder.start(100);
      setIsRecording(true);
      
      setTimeout(() => {
        if (mediaRecorderRef.current?.state === 'recording') {
          stopRecording();
        }
      }, 30000);
      
    } catch (err) {
      console.error('录音失败:', err);
      alert('无法访问麦克风，请检查权限设置');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsProcessing(true);
    }
  };

  const sendAudio = (blob: Blob) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64 = (reader.result as string).split(',')[1];
      wsRef.current?.send(JSON.stringify({
        type: 'audio_data',
        data: base64
      }));
    };
    reader.readAsDataURL(blob);
  };

  const sendTextCommand = (command: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(JSON.stringify({
      type: 'text_data',
      text: command
    }));
    setLastTranscript(command);
  };

  const endExam = () => {
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({ type: 'end_exam' }));
    }
  };

  const fetchFinalResult = async (id: string) => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/evaluation/${id}`);
      const data = await res.json();
      
      if (data.status === 'completed') {
        setFinalResult({
          evaluation_id: id,
          understanding_level: data.understanding_level,
          confidence: data.confidence,
          reasoning: data.reasoning || '',
          dialogue_text: data.dialogue_text || '',
          recommendations: data.recommendations || []
        });
        setPhase('result');
      } else {
        setTimeout(() => fetchFinalResult(id), 2000);
      }
    } catch (err) {
      console.error('获取结果失败:', err);
    }
  };

  const handleStartSubmit = async () => {
    try {
      const resp = await startOralExam({
        original_question: originalQuestion,
        original_answer: originalAnswer,
        student_id: studentId || undefined
      });
      
      setEvalId(resp.evaluation_id);
      setPhase('connecting');
      
      setTimeout(() => {
        connectWebSocket(resp.websocket_url, resp.evaluation_id);
      }, 500);
      
    } catch (err: any) {
      alert('启动失败: ' + err.message);
    }
  };

  if (phase === 'prepare') {
    return (
      <div className="max-w-2xl mx-auto p-6 bg-white rounded-lg shadow-md">
        <div className="mb-6 text-center">
          <div className="w-16 h-16 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
            </svg>
          </div>
          <h2 className="text-2xl font-bold text-gray-800">AI语音口试</h2>
          <p className="text-gray-600 mt-2">与AI考官进行语音对话，检验知识理解深度</p>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">考试主题/原题</label>
            <textarea
              value={originalQuestion}
              onChange={(e) => setOriginalQuestion(e.target.value)}
              placeholder="例如：解释快速排序算法的时间复杂度..."
              className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500 h-24"
              required
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">你的预习答案（供考官参考）</label>
            <textarea
              value={originalAnswer}
              onChange={(e) => setOriginalAnswer(e.target.value)}
              placeholder="简要描述你目前的理解，考官将基于此进行深度提问..."
              className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500 h-32"
              required
            />
          </div>

          <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4 text-sm text-yellow-800">
            <p className="font-semibold mb-1">考试说明：</p>
            <ul className="list-disc list-inside space-y-1 text-xs">
              <li>全程语音交互，考官通过语音提问，你通过语音回答</li>
              <li>你可以随时说"请重复"、"解释一下"、"给点提示"或"下一题"</li>
              <li>不评价口语表达，只关注知识理解</li>
              <li>建议使用耳机以获得最佳体验</li>
            </ul>
          </div>

          <button
            onClick={handleStartSubmit}
            disabled={!originalQuestion.trim() || !originalAnswer.trim()}
            className="w-full bg-indigo-600 text-white py-3 rounded-md hover:bg-indigo-700 disabled:bg-gray-400 font-medium transition-colors"
          >
            开始准备 → 连接考官
          </button>
        </div>
      </div>
    );
  }

  if (phase === 'connecting') {
    return (
      <div className="min-h-[60vh] flex flex-col items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mb-4"></div>
        <p className="text-gray-600">正在连接AI考官...</p>
      </div>
    );
  }

  if (phase === 'examining') {
    return (
      <div className="max-w-3xl mx-auto p-4">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 mb-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className={`w-2 h-2 rounded-full ${connectionStatus === 'connected' ? 'bg-green-500' : 'bg-red-500'}`}></span>
              <span className="text-sm text-gray-600">口试进行中</span>
            </div>
            <span className="text-xs text-gray-400">ID: {evalId.slice(0, 8)}...</span>
          </div>
        </div>

        <div className="bg-gradient-to-b from-gray-50 to-white rounded-2xl shadow-lg border border-gray-200 p-8 min-h-[400px] flex flex-col items-center justify-center relative">
          <div className={`mb-8 transition-all duration-500 ${isExaminerSpeaking ? 'scale-110' : 'scale-100'}`}>
            <div className={`w-32 h-32 rounded-full flex items-center justify-center relative ${
              isExaminerSpeaking ? 'bg-indigo-100 animate-pulse' : 'bg-gray-100'
            }`}>
              {isExaminerSpeaking && (
                <div className="absolute inset-0 rounded-full border-4 border-indigo-300 animate-ping opacity-30"></div>
              )}
              <svg className={`w-16 h-16 ${isExaminerSpeaking ? 'text-indigo-600' : 'text-gray-400'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <p className={`text-center mt-4 font-medium ${
              isExaminerSpeaking ? 'text-indigo-600' : 'text-gray-500'
            }`}>
              {isExaminerSpeaking ? '考官正在说话...' : isProcessing ? '思考中...' : '等待中'}
            </p>
          </div>

          <div className="text-center mb-8 h-8">
            <p className="text-gray-600">{currentHint}</p>
            {lastTranscript && !isRecording && (
              <p className="text-xs text-gray-400 mt-1">刚才你说: "{lastTranscript.slice(0, 30)}..."</p>
            )}
          </div>

          <div className="relative">
            <button
              onMouseDown={startRecording}
              onMouseUp={stopRecording}
              onTouchStart={startRecording}
              onTouchEnd={stopRecording}
              disabled={isExaminerSpeaking || isProcessing}
              className={`w-20 h-20 rounded-full flex items-center justify-center shadow-lg transition-all ${
                isRecording 
                  ? 'bg-red-500 scale-110 shadow-red-200' 
                  : isExaminerSpeaking || isProcessing
                    ? 'bg-gray-300 cursor-not-allowed'
                    : 'bg-indigo-600 hover:bg-indigo-700 hover:scale-105'
              }`}
            >
              {isRecording ? (
                <div className="w-6 h-6 bg-white rounded-sm animate-pulse"></div>
              ) : (
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
              )}
            </button>
            <p className="text-xs text-center mt-2 text-gray-500">
              {isRecording ? '松开结束' : '按住说话'}
            </p>
          </div>

          <div className="mt-8 grid grid-cols-4 gap-2 w-full max-w-md">
            {[
              { label: '请重复', cmd: '请重复' },
              { label: '解释一下', cmd: '什么意思' },
              { label: '给点提示', cmd: '给点提示' },
              { label: '下一题', cmd: '下一题' }
            ].map((btn) => (
              <button
                key={btn.label}
                onClick={() => sendTextCommand(btn.cmd)}
                disabled={isExaminerSpeaking || isRecording}
                className="px-3 py-2 bg-white border border-gray-300 rounded-md text-xs text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {btn.label}
              </button>
            ))}
          </div>
        </div>

        <div className="mt-6 text-center">
          <button
            onClick={endExam}
            className="text-gray-500 hover:text-gray-700 text-sm underline"
          >
            结束考试并查看结果
          </button>
        </div>

        {dialogueHistory.length > 0 && (
          <div className="mt-6 bg-white rounded-lg shadow-sm border border-gray-200 p-4">
            <details className="text-sm">
              <summary className="cursor-pointer text-gray-600 font-medium">查看对话记录</summary>
              <div className="mt-3 space-y-2 max-h-48 overflow-y-auto">
                {dialogueHistory.map((turn, idx) => (
                  <div key={idx} className={`p-2 rounded ${
                    turn.role === 'examiner' ? 'bg-indigo-50 text-indigo-900' : 'bg-gray-50 text-gray-700'
                  }`}>
                    <span className="text-xs font-bold block mb-1">
                      {turn.role === 'examiner' ? '考官' : '你'} 
                    </span>
                    <span className="text-xs">{turn.text}</span>
                  </div>
                ))}
              </div>
            </details>
          </div>
        )}
      </div>
    );
  }

  if (phase === 'grading') {
    return (
      <div className="min-h-[60vh] flex flex-col items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mb-4"></div>
        <p className="text-gray-600">正在评估你的回答...</p>
        <div className="mt-6 max-w-md text-xs text-gray-400 text-center">
          <p>阶段1: 多教师独立评分</p>
          <p>阶段2: 同行评议校准</p>
          <p>阶段3: 主席综合裁定</p>
        </div>
      </div>
    );
  }

  if (phase === 'result' && finalResult) {
    return (
      <div className="max-w-3xl mx-auto p-6 space-y-6">
        <div className="bg-gradient-to-r from-indigo-900 to-purple-900 text-white rounded-xl shadow-lg p-6">
          <h2 className="text-2xl font-bold mb-4">口试评估报告</h2>
          <div className="bg-white/10 rounded-lg p-4 backdrop-blur-sm">
            <div className="flex items-center mb-3">
              <span className="px-3 py-1 bg-green-500 rounded-full text-xs font-semibold mr-3">
                {finalResult.understanding_level}
              </span>
              <span className="text-sm opacity-90">
                置信度: {(finalResult.confidence * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-gray-200 leading-relaxed text-sm">
              {finalResult.reasoning}
            </p>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md border border-gray-200 p-6">
          <h3 className="font-bold text-gray-800 mb-4">对话记录</h3>
          <div className="bg-gray-50 rounded-lg p-4 h-64 overflow-y-auto text-xs font-mono leading-relaxed text-gray-700 whitespace-pre-wrap">
            {finalResult.dialogue_text}
          </div>
        </div>

        {finalResult.recommendations.length > 0 && (
          <div className="bg-white rounded-lg shadow-md border border-gray-200 p-6">
            <h3 className="font-bold text-gray-800 mb-3">学习建议</h3>
            <ul className="space-y-2">
              {finalResult.recommendations.map((rec, idx) => (
                <li key={idx} className="flex items-start text-sm text-gray-600">
                  <span className="mr-2 text-indigo-600">•</span>
                  {rec}
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="text-center pt-4">
          <button
            onClick={() => window.location.reload()}
            className="bg-indigo-600 text-white px-6 py-3 rounded-md hover:bg-indigo-700 transition-colors"
          >
            开始新的口试
          </button>
        </div>
      </div>
    );
  }

  return null;
};

// ==================== 主应用组件 ====================

const ExamEvaluationApp: React.FC = () => {
  const [examMode, setExamMode] = useState<'text' | 'oral'>('text');
  
  // 文字考试状态
  const [phase, setPhase] = useState<'start' | 'exam' | 'result'>('start');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
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
          <p className="text-gray-600">基于多Agent的自动考试与评估</p>
        </div>

        {/* 模式切换 */}
        <div className="flex justify-center space-x-4 mb-8">
          <button
            onClick={() => setExamMode('text')}
            className={`px-6 py-3 rounded-lg font-medium transition-all ${
              examMode === 'text' 
                ? 'bg-blue-600 text-white shadow-lg' 
                : 'bg-white text-gray-700 hover:bg-gray-50 border border-gray-300'
            }`}
          >
            <div className="flex items-center space-x-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
              </svg>
              <span>文字考试</span>
            </div>
            <p className="text-xs mt-1 opacity-80">适合详细论述</p>
          </button>
          
          <button
            onClick={() => setExamMode('oral')}
            className={`px-6 py-3 rounded-lg font-medium transition-all ${
              examMode === 'oral' 
                ? 'bg-indigo-600 text-white shadow-lg' 
                : 'bg-white text-gray-700 hover:bg-gray-50 border border-gray-300'
            }`}
          >
            <div className="flex items-center space-x-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
              <span>语音口试</span>
            </div>
            <p className="text-xs mt-1 opacity-80">实时对话交互</p>
          </button>
        </div>

        {/* 错误显示 */}
        {error && (
          <div className="max-w-3xl mx-auto mb-6 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
            <strong className="font-bold">错误:</strong> {error}
          </div>
        )}

        {/* 内容区域 */}
        {examMode === 'text' ? (
          <>
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
          </>
        ) : (
          <OralExamination />
        )}
      </div>
    </div>
  );
};

export default ExamEvaluationApp;