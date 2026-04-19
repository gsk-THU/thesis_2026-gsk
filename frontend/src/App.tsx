/**
 * App.tsx - LLM委员会考试系统（整合文字考试与语音口试）
 * 修复：静默计时逻辑 - 只在完全静默时计时，说话时不计时
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';

// ==================== 类型定义 ====================
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
  role: 'examiner' | 'student' | 'system';
  type?: 'repeat' | 'explanation' | 'hint' | 'follow_up' | 'new_topic' | 'question' | 'answer';
  text: string;
  codeSnippets?: string[];
  hasCode?: boolean;
  depth?: number;
  round?: number;
  isTyping?: boolean;
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

async function fetchExamResult(evaluation_id: string): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/api/evaluation/${evaluation_id}`);
  if (!res.ok) throw new Error('获取结果失败');
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

const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

// ==================== 代码展示组件 ====================
const CodeBlock: React.FC<{ code: string; index: number }> = ({ code, index }) => {
  const [copied, setCopied] = useState(false);
  
  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  
  const highlightCode = (code: string) => {
    return code
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/(['"])(.*?)\1/g, '<span class="text-green-400">$1$2$1</span>')
      .replace(/\b(def|return|if|else|elif|for|while|class|import|from|as|try|except|with|lambda|function|const|let|var)\b/g, '<span class="text-purple-400 font-semibold">$1</span>')
      .replace(/(#.*$|\/\/.*$)/gm, '<span class="text-gray-500 italic">$1</span>')
      .replace(/\b(\d+)\b/g, '<span class="text-orange-400">$1</span>');
  };
  
  const detectLang = (code: string): string => {
    if (code.includes('def ') || code.includes('import ')) return 'python';
    if (code.includes('function') || code.includes('const ')) return 'javascript';
    if (code.includes('public class')) return 'java';
    if (code.includes('#include')) return 'cpp';
    return 'code';
  };
  
  const lang = detectLang(code);
  
  return (
    <div className="bg-gray-900 rounded-lg my-3 overflow-hidden border border-gray-700">
      <div className="flex justify-between items-center px-3 py-2 bg-gray-800 border-b border-gray-700">
        <span className="text-xs font-semibold text-gray-300 uppercase tracking-wider">{lang}</span>
        <button
          onClick={handleCopy}
          className={`text-xs px-2 py-1 rounded transition-colors ${copied ? 'bg-green-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}
        >
          {copied ? '✓ 已复制' : '复制'}
        </button>
      </div>
      <div className="p-4 overflow-x-auto">
        <pre className="text-sm font-mono text-gray-300 leading-relaxed whitespace-pre-wrap break-all">
          <code dangerouslySetInnerHTML={{ __html: highlightCode(code) }} />
        </pre>
      </div>
    </div>
  );
};

const ExaminerMessage: React.FC<{ 
  text: string; 
  codeSnippets?: string[]; 
  hasCode?: boolean 
}> = ({ text, codeSnippets, hasCode }) => {
  if (!hasCode || !codeSnippets || codeSnippets.length === 0) {
    return <div className="whitespace-pre-wrap leading-relaxed">{text}</div>;
  }
  
  const parts = text.split('[代码片段]');
  
  return (
    <div className="leading-relaxed">
      {parts.map((part, index) => (
        <React.Fragment key={index}>
          {part && <div className="whitespace-pre-wrap my-1">{part}</div>}
          {index < codeSnippets.length && (
            <CodeBlock code={codeSnippets[index]} index={index} />
          )}
        </React.Fragment>
      ))}
    </div>
  );
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
        <p className="text-gray-600">使用Agent根据作业出题并自动评分</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">原始作业</label>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="例如：给定两个字符串形式输入的整数..."
            className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 h-24"
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">提交的结果</label>
          <textarea
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            placeholder="在此输入您对该问题的详细回答..."
            className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 h-48"
            required
          />
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
          className="w-full bg-blue-600 text-white py-3 rounded-md hover:bg-blue-700 disabled:bg-gray-400 transition-colors font-medium"
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
        <p className="text-sm text-blue-800 mt-1">基于您的回答，考官AI生成了{questions.length}个针对性问题</p>
      </div>

      <div className="space-y-6">
        {questions.map((q, idx) => (
          <div key={q.id} className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-start mb-4">
              <span className="flex-shrink-0 w-8 h-8 bg-gray-800 text-white rounded-full flex items-center justify-center font-bold text-sm mr-3">
                {idx + 1}
              </span>
              <h4 className="text-lg font-semibold text-gray-800">{q.text}</h4>
            </div>
            <textarea
              value={answers[q.id] || ''}
              onChange={(e) => setAnswers(prev => ({ ...prev, [q.id]: e.target.value }))}
              placeholder="请详细回答..."
              className="w-full p-4 border border-gray-300 rounded-md h-32"
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
          className="bg-green-600 text-white px-8 py-3 rounded-md hover:bg-green-700 disabled:bg-gray-400 font-medium"
        >
          {loading ? 'Agent评分中...' : '提交答案'}
        </button>
      </div>
    </div>
  );
};

const ResultPhase: React.FC<{ result: FinalEvaluationResult }> = ({ result }) => {
  const [expandedQuestion, setExpandedQuestion] = useState<string | null>(null);
  const avgScore = result.exam_scores.reduce((sum, s) => sum + s.final_score, 0) / result.exam_scores.length;

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-6">
      <div className="bg-gradient-to-r from-gray-900 to-gray-800 text-white rounded-xl shadow-lg p-6">
        <div className="flex justify-between items-start mb-4">
          <div>
            <h2 className="text-2xl font-bold">评估Agent最终裁定</h2>
            <p className="text-gray-300 text-sm mt-1">ID: {result.evaluation_id}</p>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold">{avgScore.toFixed(1)}/10</div>
          </div>
        </div>

        <div className="bg-white/10 rounded-lg p-4">
          <div className="flex items-center mb-3">
            <span className="px-3 py-1 bg-blue-500 rounded-full text-xs font-semibold mr-3">主席评估</span>
            <span className="text-lg font-semibold">{result.overall_assessment.understanding_level}</span>
          </div>
          <p className="text-gray-200 leading-relaxed">{result.overall_assessment.reasoning}</p>
        </div>
      </div>
    </div>
  );
};

// ==================== 语音口试组件（修复计时逻辑版）====================
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
  const [currentHint, setCurrentHint] = useState<string>('');
  
  const [silenceDuration, setSilenceDuration] = useState(0);
  const [timeoutLevel, setTimeoutLevel] = useState(0);
  const [isWaitingForAnswer, setIsWaitingForAnswer] = useState(false);
  
  const [recordingProgress, setRecordingProgress] = useState(0);
  const maxRecordingTime = 30;
  
  const [finalResult, setFinalResult] = useState<FinalOralResult | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const currentSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const currentAudioIdRef = useRef(0);
  const playLockRef = useRef(false);
  
  const silenceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const recordingTimerRef = useRef<NodeJS.Timeout | null>(null);
  const autoStopTimerRef = useRef<NodeJS.Timeout | null>(null);

  // 初始化音频上下文
  const initAudioContext = useCallback(async () => {
    if (!audioContextRef.current) {
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      if (!AudioContextClass) return false;
      try {
        audioContextRef.current = new AudioContextClass({ sampleRate: 24000 });
      } catch (e) {
        audioContextRef.current = new AudioContextClass();
      }
    }
    if (audioContextRef.current.state === 'suspended') {
      await audioContextRef.current.resume();
    }
    return true;
  }, []);

  // 停止当前播放
  const stopCurrentPlayback = useCallback(() => {
    currentAudioIdRef.current++;
    if (currentSourceRef.current) {
      try {
        currentSourceRef.current.onended = null;
        currentSourceRef.current.stop();
      } catch (e) {}
      currentSourceRef.current = null;
    }
    playLockRef.current = false;
    setIsExaminerSpeaking(false);
  }, []);

  // 播放音频Buffer
  const playAudioBuffer = useCallback(async (arrayBuffer: ArrayBuffer) => {
    if (!audioContextRef.current) await initAudioContext();
    const ctx = audioContextRef.current!;
    const thisAudioId = ++currentAudioIdRef.current;

    try {
      playLockRef.current = true;
      setIsExaminerSpeaking(true);
      const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);
      currentSourceRef.current = source;

      source.onended = () => {
        if (currentAudioIdRef.current !== thisAudioId) return;
        currentSourceRef.current = null;
        playLockRef.current = false;
        setIsExaminerSpeaking(false);
        setIsWaitingForAnswer(true);
        // 【修复】考官说完话后开始静默计时（等待学生回答）
        startSilenceTimer();
      };
      source.start(0);
    } catch (e) {
      console.error('[Audio] 播放失败:', e);
      playLockRef.current = false;
      setIsExaminerSpeaking(false);
    }
  }, [initAudioContext]);

  // 【核心修复】静默计时器 - 只在完全静默时计时
  const startSilenceTimer = useCallback(() => {
    // 清除旧计时器
    stopSilenceTimer();
    setSilenceDuration(0);
    setTimeoutLevel(0);
    
    // 开始新的计时
    silenceTimerRef.current = setInterval(() => {
      setSilenceDuration(prev => {
        const next = prev + 1;
        if (next >= 75) setTimeoutLevel(3);
        else if (next >= 50) setTimeoutLevel(2);
        else if (next >= 25) setTimeoutLevel(1);
        return next;
      });
    }, 1000);
  }, []);

  const stopSilenceTimer = useCallback(() => {
    if (silenceTimerRef.current) {
      clearInterval(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
  }, []);

  // 录音计时器
  const startRecordingTimer = useCallback(() => {
    stopRecordingTimer();
    setRecordingProgress(0);
    recordingTimerRef.current = setInterval(() => {
      setRecordingProgress(prev => prev + 0.1);
    }, 100);
  }, []);

  const stopRecordingTimer = useCallback(() => {
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
  }, []);

  const clearAutoStopTimer = useCallback(() => {
    if (autoStopTimerRef.current) {
      clearTimeout(autoStopTimerRef.current);
      autoStopTimerRef.current = null;
    }
  }, []);

  // 切换录音状态
  const toggleRecording = useCallback(async () => {
    if (isRecording) {
      stopRecording();
      return;
    }
    
    if (isExaminerSpeaking) {
      stopCurrentPlayback();
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'interrupt' }));
      }
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    if (isProcessing) return;
    
    await startRecording();
  }, [isRecording, isProcessing, isExaminerSpeaking, stopCurrentPlayback, stopRecording]);

  // 【核心修复】开始录音 - 停止静默计时（学生说话时不计时）
  const startRecording = async () => {
    const inited = await initAudioContext();
    if (!inited) return;
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true } 
      });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];
      
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };
      
      // 【核心修复】录音停止时（学生说完话）开始静默计时
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          const reader = new FileReader();
          reader.onloadend = () => {
            const base64 = (reader.result as string).split(',')[1];
            wsRef.current?.send(JSON.stringify({ type: 'audio_data', data: base64 }));
            setIsProcessing(true);
            setCurrentHint('识别中...');
            // 【修复】学生说完话，开始静默计时（等待考官回应）
            startSilenceTimer();
          };
          reader.readAsDataURL(audioBlob);
        }
        stream.getTracks().forEach(track => track.stop());
        stopRecordingTimer();
        clearAutoStopTimer();
      };
      
      mediaRecorder.start(100);
      setIsRecording(true);
      // 【核心修复】学生开始说话，停止静默计时
      stopSilenceTimer();
      startRecordingTimer();
      
      // 30秒自动停止
      autoStopTimerRef.current = setTimeout(() => {
        if (mediaRecorderRef.current?.state === 'recording') {
          stopRecording();
        }
      }, 30000);
      
    } catch (err) {
      alert('无法访问麦克风，请检查权限设置');
    }
  };

  // 停止录音
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsProcessing(true);
      clearAutoStopTimer();
      // 注意：静默计时在 onstop 回调中开始，这里不重复启动
    }
  }, [clearAutoStopTimer]);

  // 【核心修复】WebSocket消息处理 - 严格控制计时时机
  const handleWebSocketMessage = useCallback(async (data: any) => {
    if (data instanceof ArrayBuffer || data instanceof Blob) {
      const arrayBuffer = data instanceof Blob ? await data.arrayBuffer() : data;
      // 收到音频数据（考官开始说话），停止静默计时
      stopSilenceTimer();
      await playAudioBuffer(arrayBuffer);
      return;
    }

    const msg = JSON.parse(data);
    
    switch (msg.type) {
      case 'examiner_typing':
        // 考官正在生成问题（思考中），不计时（保持静默或等待状态）
        setDialogueHistory(prev => {
          if (prev.length > 0 && prev[prev.length - 1].role === 'examiner' && prev[prev.length - 1].isTyping) {
            return prev;
          }
          return [...prev, {
            role: 'examiner',
            text: '',
            codeSnippets: [],
            hasCode: false,
            isTyping: true,
            timestamp: new Date().toISOString()
          }];
        });
        setCurrentHint(msg.message || '考官正在准备问题...');
        break;
      
      case 'examiner_response':
        // 考官文字回复已生成，但还没开始语音
        setDialogueHistory(prev => {
          const newHistory = [...prev];
          if (newHistory.length > 0 && newHistory[newHistory.length - 1].isTyping) {
            newHistory.pop();
          }
          newHistory.push({
            role: 'examiner',
            text: msg.text || '',
            codeSnippets: msg.code_snippets || [],
            hasCode: msg.has_code || false,
            type: msg.response_type,
            depth: msg.depth || 0,
            round: msg.round || 0,
            isTyping: false,
            timestamp: new Date().toISOString()
          });
          return newHistory;
        });
        
        const hints: Record<string, string> = {
          'repeat': '考官重复问题',
          'explanation': '考官解释中',
          'hint': '考官提示中',
          'follow_up': '考官深入追问',
          'new_topic': '切换新话题',
          'question': '考官提出新问题'
        };
        setCurrentHint(msg.has_code ? `💻 ${hints[msg.response_type] || '考官展示代码'}` : (hints[msg.response_type] || '考官提问中'));
        
        setIsWaitingForAnswer(false);
        // 考官生成文字后，即将开始语音，不计时（等待audio_start）
        stopSilenceTimer();
        break;
        
      case 'audio_start':
        // 【核心修复】考官开始说话，确保停止静默计时
        setIsExaminerSpeaking(true);
        setIsWaitingForAnswer(false);
        stopSilenceTimer();
        break;
        
      case 'audio_generating':
        setCurrentHint(msg.message || '准备语音...');
        break;
        
      case 'audio_end':
        // 【核心修复】考官说完话，开始静默计时（已在playAudioBuffer的onended中处理，这里双重保险）
        setIsExaminerSpeaking(false);
        setIsWaitingForAnswer(true);
        startSilenceTimer();
        break;
      
      case 'listening':
        // 【核心修复】考官等待回答，开始静默计时
        setIsProcessing(false);
        setCurrentHint('等待你的回答...');
        setIsWaitingForAnswer(true);
        setSilenceDuration(0);
        setTimeoutLevel(0);
        startSilenceTimer();
        break;
        
      case 'input_ready':
        // 【核心修复】系统就绪（如识别失败后恢复），开始静默计时
        setIsProcessing(false);
        setCurrentHint(msg.message || '请回答');
        startSilenceTimer();
        break;
        
      case 'processing':
        // 系统处理中（语音识别中），不计时（等待transcription）
        setIsProcessing(true);
        setCurrentHint('识别中...');
        // 保持静默计时继续（学生在等待识别结果）
        break;
        
      case 'transcription':
        // 【核心修复】显示学生回答后，继续静默计时（等待考官回应）
        setDialogueHistory(prev => [...prev, {
          role: 'student',
          text: msg.text,
          type: 'answer',
          timestamp: new Date().toISOString()
        }]);
        // 识别完成，继续计时直到考官回应
        break;
        
      case 'silence_reminder':
      case 'timeout_reminder':
        // 静默提醒，更新UI但不改变计时状态
        setCurrentHint(msg.message);
        setTimeoutLevel(msg.level || 1);
        setDialogueHistory(prev => [...prev, {
          role: 'system',
          text: msg.message,
          timestamp: new Date().toISOString()
        }]);
        break;
        
      case 'timeout_action':
        setTimeoutLevel(3);
        setCurrentHint(msg.message);
        if (msg.action === 'end_exam') {
          stopSilenceTimer();
          setPhase('grading');
        }
        break;
        
      case 'exam_complete':
        setPhase('grading');
        setCurrentHint('评估中...');
        stopSilenceTimer();
        if (isRecording) {
          stopRecording();
        }
        setTimeout(() => fetchFinalResult(evalId), 3000);
        break;
        
      case 'interrupted':
        stopCurrentPlayback();
        setIsProcessing(false);
        setCurrentHint('已暂停');
        // 中断后处于不确定状态，不自动开始计时，等待考官响应
        break;
        
      case 'error':
        setCurrentHint('错误: ' + msg.message);
        setIsProcessing(false);
        // 出错后也不自动计时，等待明确的状态转换
        break;
    }
  }, [evalId, isRecording, playAudioBuffer, stopCurrentPlayback, stopRecording, startSilenceTimer, stopSilenceTimer]);

  // WebSocket连接
  const connectWebSocket = useCallback((url: string, id: string) => {
    setConnectionStatus('connecting');
    const ws = new WebSocket(url);
    wsRef.current = ws;
    ws.binaryType = 'arraybuffer';
    
    ws.onopen = () => {
      setConnectionStatus('connected');
      setPhase('examining');
      ws.send(JSON.stringify({ 
        type: 'start_exam',
        timeout_strategy: 'prompt',
        silence_thresholds: [25, 50, 75]
      }));
      setCurrentHint('连接成功，考官准备中...');
      // 初始状态不计时，等待考官第一个问题
    };
    
    ws.onmessage = async (event) => {
      await handleWebSocketMessage(event.data);
    };
    
    ws.onerror = () => {
      setConnectionStatus('error');
    };
    
    ws.onclose = () => {
      setConnectionStatus('idle');
      stopCurrentPlayback();
      stopSilenceTimer();
      stopRecordingTimer();
      clearAutoStopTimer();
      if (isRecording) {
        setIsRecording(false);
        setIsProcessing(false);
      }
    };
  }, [handleWebSocketMessage, stopCurrentPlayback, stopSilenceTimer, stopRecordingTimer, clearAutoStopTimer, isRecording]);

  const endExam = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'end_exam' }));
    }
    stopSilenceTimer();
    stopRecordingTimer();
    clearAutoStopTimer();
    if (isRecording) {
      stopRecording();
    }
    stopCurrentPlayback();
  }, [stopCurrentPlayback, stopRecording, isRecording, stopSilenceTimer, stopRecordingTimer, clearAutoStopTimer]);

  const fetchFinalResult = useCallback(async (id: string) => {
    try {
      const data = await fetchExamResult(id);
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
      setTimeout(() => fetchFinalResult(id), 5000);
    }
  }, []);

  const handleStartSubmit = async () => {
    if (!originalQuestion.trim() || !originalAnswer.trim()) {
      alert('请填写考试主题和预习答案');
      return;
    }
    await initAudioContext();
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

  // 清理函数
  useEffect(() => {
    return () => {
      stopSilenceTimer();
      stopRecordingTimer();
      clearAutoStopTimer();
      if (wsRef.current) wsRef.current.close();
    };
  }, [stopSilenceTimer, stopRecordingTimer, clearAutoStopTimer]);

  // 键盘快捷键
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && phase === 'examining' && !isProcessing && !isExaminerSpeaking) {
        e.preventDefault();
        toggleRecording();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [phase, isProcessing, isExaminerSpeaking, isRecording, toggleRecording]);

  // 渲染左侧语音面板
  const renderVoicePanel = () => (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 flex flex-col items-center h-fit sticky top-6">
      {/* 状态栏 */}
      <div className="w-full flex justify-between items-center mb-6 pb-4 border-b border-gray-100">
        <div className="flex items-center gap-2">
          <div className={`w-2.5 h-2.5 rounded-full ${connectionStatus === 'connected' ? 'bg-green-500 shadow-[0_0_0_3px_rgba(16,185,129,0.2)]' : 'bg-red-500'}`}></div>
          <span className="text-sm font-medium text-gray-700">
            {connectionStatus === 'connected' ? '考试中' : '未连接'}
          </span>
        </div>
        <button 
          onClick={endExam} 
          className="text-red-600 text-sm border border-red-200 bg-red-50 hover:bg-red-100 px-3 py-1 rounded-md font-medium transition-colors"
        >
          结束
        </button>
      </div>

      {/* 考官头像 */}
      <div className={`w-24 h-24 rounded-full flex items-center justify-center text-4xl mb-5 transition-all ${
        isExaminerSpeaking ? 'bg-indigo-50 border-indigo-500 animate-pulse' : isWaitingForAnswer ? 'bg-amber-50 border-amber-400' : 'bg-gray-50 border-gray-200'
      } border-2`}>
        {isExaminerSpeaking ? '🗣️' : isWaitingForAnswer ? '⏳' : '✋'}
      </div>

      {/* 提示文字 */}
      <p className="text-base font-medium text-gray-700 mb-6 text-center min-h-[3rem] leading-snug px-2">
        {currentHint || '准备就绪'}
      </p>

      {/* 【核心修复】静默进度条 - 只在静默时显示 */}
      {(isWaitingForAnswer && !isRecording && !isProcessing && !isExaminerSpeaking) && (
        <div className="w-full mb-6">
          <div className="flex justify-between text-xs text-gray-500 mb-1 font-medium">
            <span className={timeoutLevel > 0 ? 'text-red-600' : ''}>等待回答计时</span>
            <span className={timeoutLevel > 0 ? 'text-red-600' : ''}>{silenceDuration}秒 / 75秒</span>
          </div>
          <div className="w-full h-2 bg-gray-100 rounded-full overflow-hidden">
            <div className={`h-full transition-all duration-1000 rounded-full ${
              timeoutLevel === 0 ? 'bg-green-500' : timeoutLevel === 1 ? 'bg-yellow-500' : timeoutLevel === 2 ? 'bg-orange-500' : 'bg-red-500'
            }`} style={{ width: `${Math.min((silenceDuration / 75) * 100, 100)}%` }}></div>
          </div>
          <div className="flex justify-between text-[10px] text-gray-400 mt-1 font-medium">
            <span>25秒</span>
            <span>50秒</span>
            <span>75秒</span>
          </div>
        </div>
      )}

      {/* 录音按钮 */}
      <div className="relative mb-3">
        <button
          onClick={toggleRecording}
          disabled={isProcessing || isExaminerSpeaking}
          className={`w-24 h-24 rounded-full flex items-center justify-center text-white shadow-lg transition-all duration-200 ${
            isRecording 
              ? 'bg-red-500 scale-110 shadow-red-300 ring-4 ring-red-200' 
              : (isExaminerSpeaking || isProcessing) 
                ? 'bg-gray-300 cursor-not-allowed opacity-60' 
                : 'bg-indigo-600 hover:bg-indigo-700 hover:scale-105 active:scale-95'
          }`}
        >
          {isRecording ? (
            <svg className="w-10 h-10" fill="currentColor" viewBox="0 0 24 24">
              <rect x="6" y="6" width="12" height="12" rx="2" />
            </svg>
          ) : (
            <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
            </svg>
          )}
          
          {/* 录音进度圆环 */}
          {isRecording && (
            <div className="absolute -inset-2 rounded-full border-4 border-gray-200 border-t-indigo-600 animate-spin" style={{ animationDuration: '3s' }} />
          )}
        </button>
        
        {/* 录音时长显示 */}
        {isRecording && (
          <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 text-sm font-mono font-bold text-red-500">
            {formatTime(recordingProgress)} / 0:30
          </div>
        )}
      </div>

      {/* 按钮下方提示文字 */}
      <p className="text-sm text-gray-500 mt-4 text-center font-medium">
        {isRecording 
          ? '点击结束录音' 
          : isExaminerSpeaking 
            ? '考官说话中...' 
            : isProcessing 
              ? '识别中...' 
              : '点击开始录音'}
      </p>

      {/* 快捷指令 - 只在静默等待时显示 */}
      {(isWaitingForAnswer && !isRecording && !isProcessing && !isExaminerSpeaking) && (
        <div className="flex gap-2 justify-center mt-5 flex-wrap">
          {['请重复', '解释一下', '给点提示', '下一题'].map((cmd) => (
            <button
              key={cmd}
              onClick={() => {
                if (wsRef.current?.readyState === WebSocket.OPEN) {
                  wsRef.current.send(JSON.stringify({ type: 'text_data', text: cmd }));
                  setIsProcessing(true);
                  setCurrentHint('考官思考中...');
                  stopSilenceTimer(); // 发送快捷指令后停止计时（等待考官回应）
                }
              }}
              className="px-3 py-1.5 rounded-full border border-gray-200 bg-white text-xs text-gray-700 hover:bg-gray-50 font-medium transition-colors shadow-sm"
            >
              {cmd}
            </button>
          ))}
        </div>
      )}
    </div>
  );

  // 渲染右侧对话面板
  const renderDialoguePanel = () => (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 flex flex-col h-[calc(100vh-140px)] min-h-[500px]">
      <h3 className="text-base font-semibold text-gray-900 mb-4 pb-3 border-b-2 border-gray-100 flex items-center justify-between">
        <span>对话记录</span>
        {dialogueHistory.length > 0 && (
          <span className="text-xs text-gray-500 font-normal bg-gray-100 px-2 py-1 rounded-full">
            {dialogueHistory.filter(d => !d.isTyping).length} 条消息
          </span>
        )}
      </h3>
      
      <div className="flex-1 overflow-y-auto space-y-3 pr-2">
        {dialogueHistory.length === 0 ? (
          <div className="text-center text-gray-400 mt-10 italic">
            <div className="text-5xl mb-4">💬</div>
            等待考官第一个问题...
          </div>
        ) : (
          dialogueHistory.map((turn, idx) => (
            <div 
              key={idx} 
              className={`p-4 rounded-lg border-l-4 ${
                turn.role === 'examiner' ? 'bg-gray-50 border-indigo-500' : 
                turn.role === 'student' ? 'bg-green-50 border-green-500' : 
                'bg-amber-50 border-amber-400'
              } ${turn.isTyping ? 'animate-pulse' : ''}`}
            >
              <div className="flex justify-between items-center mb-2 text-xs text-gray-500">
                <span className="font-semibold flex items-center gap-1.5">
                  {turn.role === 'examiner' ? '👨‍🏫 考官' : turn.role === 'student' ? '🎓 你' : '🔔 系统'}
                  {turn.role === 'examiner' && turn.depth && turn.depth > 0 && (
                    <span className="text-amber-600 text-[10px] bg-amber-100 px-1.5 py-0.5 rounded">追问{turn.depth}</span>
                  )}
                </span>
                <span className="text-[11px]">
                  {new Date(turn.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'})}
                </span>
              </div>
              
              <div className="text-gray-800 text-sm leading-relaxed">
                {turn.isTyping ? (
                  <div className="flex items-center gap-2 text-gray-400 italic">
                    <span className="w-4 h-4 border-2 border-gray-300 border-t-indigo-500 rounded-full animate-spin"></span>
                    考官正在输入...
                  </div>
                ) : turn.role === 'examiner' ? (
                  <ExaminerMessage text={turn.text} codeSnippets={turn.codeSnippets} hasCode={turn.hasCode} />
                ) : (
                  <div className="whitespace-pre-wrap">{turn.text}</div>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );

  // 渲染准备阶段
  if (phase === 'prepare') {
    return (
      <div className="max-w-2xl mx-auto">
        <div className="bg-white rounded-lg shadow-md p-8">
          <div className="text-center mb-8">
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
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">学生ID（可选）</label>
              <input
                type="text"
                value={studentId}
                onChange={(e) => setStudentId(e.target.value)}
                placeholder="例如：student_001"
                className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500"
              />
            </div>
            <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4 text-sm text-yellow-800 my-4">
              <p className="font-semibold mb-1">考试说明：</p>
              <ul className="list-disc list-inside space-y-1 text-xs">
                <li>全程语音交互，考官通过语音提问，你通过语音回答</li>
                <li>你可以随时说"请重复"、"解释一下"、"给点提示"或"下一题"</li>
                <li>75秒未回答将触发超时提醒</li>
                <li>计时规则：只在双方都不说话时计时，说话时自动暂停</li>
                <li>建议使用耳机以获得最佳体验</li>
              </ul>
            </div>
            <button
              onClick={handleStartSubmit}
              disabled={!originalQuestion.trim() || !originalAnswer.trim()}
              className="w-full bg-indigo-600 text-white py-3 rounded-md hover:bg-indigo-700 disabled:bg-gray-400 font-medium transition-colors"
            >
              开始考试
            </button>
          </div>
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
      <div className="max-w-7xl mx-auto p-4">
        <div className="grid grid-cols-1 lg:grid-cols-[360px_1fr] gap-6 items-start">
          {renderVoicePanel()}
          {renderDialoguePanel()}
        </div>
      </div>
    );
  }

  if (phase === 'grading') {
    return (
      <div className="min-h-[60vh] flex flex-col items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mb-4"></div>
        <p className="text-gray-600">正在评估你的回答...</p>
      </div>
    );
  }

  if (phase === 'result' && finalResult) {
    return (
      <div className="max-w-4xl mx-auto p-6 space-y-6">
        <div className="bg-gradient-to-r from-indigo-900 to-purple-900 text-white rounded-xl shadow-lg p-6">
          <h2 className="text-2xl font-bold mb-4">口试评估报告</h2>
          <div className="bg-white/10 rounded-lg p-4">
            <div className="flex items-center mb-3">
              <span className="px-3 py-1 bg-green-500 rounded-full text-xs font-semibold mr-3">{finalResult.understanding_level}</span>
              <span className="text-sm opacity-90">置信度: {(finalResult.confidence * 100).toFixed(0)}%</span>
            </div>
            <p className="text-gray-200 leading-relaxed text-sm">{finalResult.reasoning}</p>
          </div>
        </div>
      </div>
    );
  }

  return null;
};

// ==================== 主应用组件 ====================
const ExamEvaluationApp: React.FC = () => {
  const [examMode, setExamMode] = useState<'text' | 'oral'>('text');
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
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">LLM委员会考试系统</h1>
          <p className="text-gray-600">基于多Agent的自动考试与评估</p>
        </div>

        <div className="flex justify-center space-x-4 mb-8">
          <button
            onClick={() => setExamMode('text')}
            className={`px-6 py-3 rounded-lg font-medium transition-all ${
              examMode === 'text' ? 'bg-blue-600 text-white shadow-lg' : 'bg-white text-gray-700 border border-gray-300'
            }`}
          >
            文字考试
          </button>
          <button
            onClick={() => setExamMode('oral')}
            className={`px-6 py-3 rounded-lg font-medium transition-all ${
              examMode === 'oral' ? 'bg-indigo-600 text-white shadow-lg' : 'bg-white text-gray-700 border border-gray-300'
            }`}
          >
            语音口试
          </button>
        </div>

        {error && (
          <div className="max-w-3xl mx-auto mb-6 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
            <strong>错误:</strong> {error}
          </div>
        )}

        {examMode === 'text' ? (
          <>
            {phase === 'start' && <StartPhase onStart={handleStart} loading={loading} />}
            {phase === 'exam' && examQuestions && (
              <ExamPhase
                evaluationId={evaluationId}
                questions={examQuestions.exam_questions}
                originalQuestion={examQuestions.original_question}
                onSubmit={handleSubmitExam}
                loading={loading}
              />
            )}
            {phase === 'result' && finalResult && <ResultPhase result={finalResult} />}
          </>
        ) : (
          <OralExamination />
        )}
      </div>
    </div>
  );
};

export default ExamEvaluationApp;