/**
 * App.tsx - LLM委员会考试系统（文字考试 + 语音口试 + OS实验模式）
 * 兼容后端 API 版本 3.0.0
 * 
 * 改进：点击开始考试后立即跳转到语音考试界面，在考试界面等待出题
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';

// ==================== 类型定义 (保持不变) ====================
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

interface OralExamStartResponse {
  evaluation_id: string;
  status: string;
  websocket_url: string;
  config: {
    sample_rate: number;
    language: string;
    tts_voice: string;
  };
  total_questions?: number;
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

// ==================== API 客户端 ====================
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

async function startOSEvaluation(formData: FormData): Promise<ExamQuestionsResponse> {
  const res = await fetch(`${API_BASE_URL}/api/os-experiment/start`, {
    method: 'POST',
    body: formData,
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

async function startOralExam(data: any): Promise<OralExamStartResponse> {
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

const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

// ==================== 代码展示组件 ====================
const highlightCode = (code: string) => {
  // 先转义 HTML，防止原始代码中的 < > & 破坏结构
  let result = code
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // 1. 先处理多行注释 /* ... */（必须在字符串之前，避免字符串里的 /* 被误匹配）
  // 但更好的做法是先处理字符串，因为字符串里可能有 /*

  // 1. 先处理字符串（单引号和双引号）
  result = result.replace(
    /(['"])(.*?)\1/g,
    '<span class="text-green-400">$1$2$1</span>'
  );

  // 2. 处理多行注释 /* ... */
  result = result.replace(
    /\/\*[\s\S]*?\*\//g,
    '<span class="text-gray-500 italic">$&</span>'
  );

  // 3. 处理单行注释 // ...
  result = result.replace(
    /\/\/.*$/gm,
    '<span class="text-gray-500 italic">$&</span>'
  );

  // 4. 处理关键字（必须在注释之后，避免注释里的关键字被高亮）
  const keywords = [
    'def', 'return', 'if', 'else', 'elif', 'for', 'while', 'class',
    'import', 'from', 'as', 'try', 'except', 'with', 'lambda',
    'function', 'const', 'let', 'var', 'struct', 'switch', 'case',
    'break', 'continue', 'goto', 'typedef', 'unsigned', 'int', 'char',
    'uint8_t', 'uint64_t', 'uint64', 'unsigned long', 'void', 'static',
    'inline', 'volatile', 'sizeof'
  ];
  const kwPattern = new RegExp(`\\b(${keywords.join('|')})\\b`, 'g');
  result = result.replace(
    kwPattern,
    '<span class="text-purple-400 font-semibold">$1</span>'
  );

  // 5. 处理数字
  result = result.replace(
    /\b(\d+)\b/g,
    '<span class="text-orange-400">$1</span>'
  );

  return result;
};

const ExaminerMessage: React.FC<{ text: string; codeSnippets?: string[]; hasCode?: boolean }> = ({
  text,
  codeSnippets,
  hasCode,
}) => {
  if (!hasCode || !codeSnippets || codeSnippets.length === 0) {
    return <div className="whitespace-pre-wrap leading-relaxed">{text}</div>;
  }

  const parts = text.split('[代码片段]');

  return (
    <div className="leading-relaxed">
      {parts.map((part, index) => (
        <React.Fragment key={index}>
          {part && <div className="whitespace-pre-wrap my-1">{part}</div>}
          {index < codeSnippets.length && <CodeBlock code={codeSnippets[index]} index={index} />}
        </React.Fragment>
      ))}
    </div>
  );
};

// ==================== 文字考试子组件 (保持不变) ====================
const OriginalStartForm: React.FC<{
  question: string;
  setQuestion: (v: string) => void;
  answer: string;
  setAnswer: (v: string) => void;
  studentId: string;
  setStudentId: (v: string) => void;
  disabled: boolean;
}> = ({ question, setQuestion, answer, setAnswer, studentId, setStudentId, disabled }) => (
  <>
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1">原始作业</label>
      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="例如：给定两个字符串形式输入的整数..."
        className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 h-24"
        required
        disabled={disabled}
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
        disabled={disabled}
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
        disabled={disabled}
      />
    </div>
  </>
);

const OSStartForm: React.FC<{
  requirement: string;
  setRequirement: (v: string) => void;
  beforeFile: File | null;
  setBeforeFile: (f: File | null) => void;
  afterFile: File | null;
  setAfterFile: (f: File | null) => void;
  studentId: string;
  setStudentId: (v: string) => void;
  disabled: boolean;
}> = ({
  requirement,
  setRequirement,
  beforeFile,
  setBeforeFile,
  afterFile,
  setAfterFile,
  studentId,
  setStudentId,
  disabled,
}) => {
  const fileInputRef1 = useRef<HTMLInputElement>(null);
  const fileInputRef2 = useRef<HTMLInputElement>(null);

  return (
    <>
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">实验要求描述</label>
        <textarea
          value={requirement}
          onChange={(e) => setRequirement(e.target.value)}
          placeholder="例如：实验三：内存管理。实现一个简化的内存分配器，支持首次适应和最佳适应算法..."
          className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 h-32"
          required
          disabled={disabled}
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">修改前代码 (ZIP)</label>
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => fileInputRef1.current?.click()}
            className="px-4 py-2 bg-gray-100 border border-gray-300 rounded-md text-sm text-gray-700 hover:bg-gray-200 transition-colors"
            disabled={disabled}
          >
            选择文件
          </button>
          <span className="text-sm text-gray-600 truncate max-w-[200px]">
            {beforeFile ? beforeFile.name : '未选择文件'}
          </span>
        </div>
        <input
          ref={fileInputRef1}
          type="file"
          accept=".zip"
          className="hidden"
          onChange={(e) => setBeforeFile(e.target.files?.[0] || null)}
          disabled={disabled}
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">修改后代码 (ZIP)</label>
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => fileInputRef2.current?.click()}
            className="px-4 py-2 bg-gray-100 border border-gray-300 rounded-md text-sm text-gray-700 hover:bg-gray-200 transition-colors"
            disabled={disabled}
          >
            选择文件
          </button>
          <span className="text-sm text-gray-600 truncate max-w-[200px]">
            {afterFile ? afterFile.name : '未选择文件'}
          </span>
        </div>
        <input
          ref={fileInputRef2}
          type="file"
          accept=".zip"
          className="hidden"
          onChange={(e) => setAfterFile(e.target.files?.[0] || null)}
          disabled={disabled}
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
          disabled={disabled}
        />
      </div>
    </>
  );
};

const StartPhase: React.FC<{
  onStartOriginal: (data: StartEvaluationRequest) => void;
  onStartOS: (formData: FormData) => void;
  loading: boolean;
}> = ({ onStartOriginal, onStartOS, loading }) => {
  const [mode, setMode] = useState<'original' | 'os'>('original');
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [studentId, setStudentId] = useState('');
  const [requirement, setRequirement] = useState('');
  const [beforeFile, setBeforeFile] = useState<File | null>(null);
  const [afterFile, setAfterFile] = useState<File | null>(null);
  const [osStudentId, setOsStudentId] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (mode === 'original') {
      onStartOriginal({
        original_question: question,
        original_answer: answer,
        student_id: studentId || undefined,
      });
    } else {
      if (!beforeFile || !afterFile) {
        alert('请上传修改前和修改后的代码ZIP文件');
        return;
      }
      const formData = new FormData();
      formData.append('experiment_requirement', requirement);
      formData.append('before_zip', beforeFile);
      formData.append('after_zip', afterFile);
      formData.append('num_questions', '5');
      if (osStudentId) formData.append('student_id', osStudentId);
      onStartOS(formData);
    }
  };

  const isSubmitDisabled =
    loading ||
    (mode === 'original'
      ? !question.trim() || !answer.trim()
      : !requirement.trim() || !beforeFile || !afterFile);

  return (
    <div className="max-w-3xl mx-auto p-6 bg-white rounded-lg shadow-md">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">AI文字考试</h2>
        <p className="text-gray-600">使用Agent根据作业出题并自动评分</p>
      </div>

      <div className="flex bg-gray-100 rounded-lg p-1 mb-6">
        <button
          type="button"
          onClick={() => setMode('original')}
          className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all ${
            mode === 'original' ? 'bg-white text-blue-700 shadow-sm' : 'text-gray-600 hover:text-gray-800'
          }`}
        >
          简单问题模式
        </button>
        <button
          type="button"
          onClick={() => setMode('os')}
          className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all ${
            mode === 'os' ? 'bg-white text-blue-700 shadow-sm' : 'text-gray-600 hover:text-gray-800'
          }`}
        >
          操作系统实验模式
        </button>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {mode === 'original' ? (
          <OriginalStartForm
            question={question}
            setQuestion={setQuestion}
            answer={answer}
            setAnswer={setAnswer}
            studentId={studentId}
            setStudentId={setStudentId}
            disabled={loading}
          />
        ) : (
          <OSStartForm
            requirement={requirement}
            setRequirement={setRequirement}
            beforeFile={beforeFile}
            setBeforeFile={setBeforeFile}
            afterFile={afterFile}
            setAfterFile={setAfterFile}
            studentId={osStudentId}
            setStudentId={setOsStudentId}
            disabled={loading}
          />
        )}

        <button
          type="submit"
          disabled={isSubmitDisabled}
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
    const unanswered = questions.filter((q) => !answers[q.id]?.trim());
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
          基于您的回答，考官AI生成了{questions.length}个针对性问题
        </p>
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
              onChange={(e) => setAnswers((prev) => ({ ...prev, [q.id]: e.target.value }))}
              placeholder="请详细回答..."
              className="w-full p-4 border border-gray-300 rounded-md h-32"
            />
          </div>
        ))}
      </div>

      <div className="mt-8 flex justify-between items-center">
        <div className="text-sm text-gray-600">
          已完成: {Object.values(answers).filter((v) => v.trim()).length} / {questions.length}
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

// ==================== 语音口试组件（改进版：点击立即跳转考试界面） ====================
const OralExamination: React.FC<{ osOnly?: boolean }> = ({ osOnly = false }) => {
  // 模式选择
  const oralMode = osOnly ? 'os' : 'debug';

  // 通用模式字段
  const [originalQuestion, setOriginalQuestion] = useState('');
  const [originalAnswer, setOriginalAnswer] = useState('');

  // OS 模式字段
  const [experimentRequirement, setExperimentRequirement] = useState('');
  const [beforeZip, setBeforeZip] = useState<File | null>(null);
  const [afterZip, setAfterZip] = useState<File | null>(null);
  const [osNumQuestions, setOsNumQuestions] = useState(5);
  const [osType, setOsType] = useState<'ucore' | 'rcore'>('rcore');
  const [studentId, setStudentId] = useState('');

  // 会话状态
  const [phase, setPhase] = useState<'prepare' | 'examining' | 'grading' | 'result'>('prepare');
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
  const [finalResult, setFinalResult] = useState<any>(null);

  // 新增：标记是否正在生成第一个问题 / 连接中
  const [isGeneratingFirstQuestion, setIsGeneratingFirstQuestion] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);
  const [initError, setInitError] = useState<string | null>(null);

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const currentSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const currentAudioIdRef = useRef(0);
  const silenceTimerRef = useRef<NodeJS.Timeout | null>(null);
  const recordingTimerRef = useRef<NodeJS.Timeout | null>(null);
  const autoStopTimerRef = useRef<NodeJS.Timeout | null>(null);
  const lastProgressTimeRef = useRef(Date.now());

  const silenceThresholds = [120, 180, 240];

  // 辅助函数
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

  const stopCurrentPlayback = useCallback(() => {
    currentAudioIdRef.current++;
    if (currentSourceRef.current) {
      try {
        currentSourceRef.current.onended = null;
        currentSourceRef.current.stop();
      } catch (e) {}
      currentSourceRef.current = null;
    }
    setIsExaminerSpeaking(false);
  }, []);

  const playAudioBuffer = useCallback(async (arrayBuffer: ArrayBuffer) => {
    if (!audioContextRef.current) await initAudioContext();
    const ctx = audioContextRef.current!;
    const thisAudioId = ++currentAudioIdRef.current;

    try {
      setIsExaminerSpeaking(true);
      const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);
      currentSourceRef.current = source;

      source.onended = () => {
        if (currentAudioIdRef.current !== thisAudioId) return;
        currentSourceRef.current = null;
        setIsExaminerSpeaking(false);
        setIsWaitingForAnswer(true);
        startSilenceTimer();
      };
      source.start(0);
    } catch (e) {
      console.error('[Audio] 播放失败:', e);
      setIsExaminerSpeaking(false);
    }
  }, [initAudioContext]);

  const startSilenceTimer = useCallback(() => {
    stopSilenceTimer();
    setSilenceDuration(0);
    setTimeoutLevel(0);
    silenceTimerRef.current = setInterval(() => {
      setSilenceDuration((prev) => {
        const next = prev + 1;
        if (next >= silenceThresholds[2]) setTimeoutLevel(3);
        else if (next >= silenceThresholds[1]) setTimeoutLevel(2);
        else if (next >= silenceThresholds[0]) setTimeoutLevel(1);
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

  const startRecordingTimer = useCallback(() => {
    stopRecordingTimer();
    setRecordingProgress(0);
    recordingTimerRef.current = setInterval(() => {
      setRecordingProgress((prev) => prev + 0.1);
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

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsProcessing(true);
      clearAutoStopTimer();
    }
  }, [clearAutoStopTimer]);

  const startRecording = useCallback(async () => {
    const inited = await initAudioContext();
    if (!inited) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          const reader = new FileReader();
          reader.onloadend = () => {
            const base64 = (reader.result as string).split(',')[1];
            wsRef.current?.send(JSON.stringify({ type: 'audio_data', data: base64 }));
            setIsProcessing(true);
            setCurrentHint('识别中...');
            startSilenceTimer();
          };
          reader.readAsDataURL(audioBlob);
        }
        stream.getTracks().forEach((track) => track.stop());
        stopRecordingTimer();
        clearAutoStopTimer();
      };

      mediaRecorder.start(100);
      setIsRecording(true);
      stopSilenceTimer();
      startRecordingTimer();

      autoStopTimerRef.current = setTimeout(() => {
        if (mediaRecorderRef.current?.state === 'recording') {
          stopRecording();
        }
      }, 30000);
    } catch (err) {
      alert('无法访问麦克风，请检查权限设置');
    }
  }, [initAudioContext, stopSilenceTimer, startRecordingTimer, clearAutoStopTimer, stopRecording]);

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
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
    if (isProcessing) return;
    await startRecording();
  }, [isRecording, isProcessing, isExaminerSpeaking, stopCurrentPlayback, stopRecording, startRecording]);

  const handleWebSocketMessage = useCallback(
    async (data: any) => {
      lastProgressTimeRef.current = Date.now();

      if (data instanceof ArrayBuffer || data instanceof Blob) {
        const arrayBuffer = data instanceof Blob ? await data.arrayBuffer() : data;
        stopSilenceTimer();
        await playAudioBuffer(arrayBuffer);
        return;
      }

      const msg = JSON.parse(data);

      switch (msg.type) {
        case 'examiner_typing':
          // 收到考官正在输入的消息，说明第一个问题已开始生成，清除"生成问题中"状态
          if (isGeneratingFirstQuestion) setIsGeneratingFirstQuestion(false);
          setDialogueHistory((prev) => {
            if (prev.length > 0 && prev[prev.length - 1].role === 'examiner' && prev[prev.length - 1].isTyping) {
              return prev;
            }
            return [
              ...prev,
              {
                role: 'examiner',
                text: '',
                codeSnippets: [],
                hasCode: false,
                isTyping: true,
                timestamp: new Date().toISOString(),
              },
            ];
          });
          setCurrentHint(msg.message || '考官正在准备问题...');
          break;

        case 'examiner_response':
          // 考官已经给出正式响应，清除生成状态
          if (isGeneratingFirstQuestion) setIsGeneratingFirstQuestion(false);
          setDialogueHistory((prev) => {
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
              timestamp: new Date().toISOString(),
            });
            return newHistory;
          });
          const hints: Record<string, string> = {
            repeat: '考官重复问题',
            explanation: '考官解释中',
            hint: '考官提示中',
            follow_up: '考官深入追问',
            new_topic: '切换新话题',
            question: '考官提出新问题',
          };
          setCurrentHint(
            msg.has_code
              ? `💻 ${hints[msg.response_type] || '考官展示代码'}`
              : hints[msg.response_type] || '考官提问中'
          );
          setIsWaitingForAnswer(false);
          stopSilenceTimer();
          break;

        case 'audio_start':
          setIsExaminerSpeaking(true);
          setIsWaitingForAnswer(false);
          stopSilenceTimer();
          break;

        case 'audio_generating':
          setCurrentHint(msg.message || '准备语音...');
          break;

        case 'audio_end':
          setIsExaminerSpeaking(false);
          setIsWaitingForAnswer(true);
          startSilenceTimer();
          break;

        case 'listening':
          setIsProcessing(false);
          setCurrentHint('等待你的回答...');
          setIsWaitingForAnswer(true);
          setSilenceDuration(0);
          setTimeoutLevel(0);
          startSilenceTimer();
          break;

        case 'input_ready':
          setIsProcessing(false);
          setCurrentHint(msg.message || '请回答');
          startSilenceTimer();
          break;

        case 'processing':
          setIsProcessing(true);
          setCurrentHint('识别中...');
          break;

        case 'transcription':
          setDialogueHistory((prev) => [
            ...prev,
            {
              role: 'student',
              text: msg.text,
              type: 'answer',
              timestamp: new Date().toISOString(),
            },
          ]);
          break;

        case 'silence_reminder':
        case 'timeout_reminder':
          setCurrentHint(msg.message);
          setTimeoutLevel(msg.level || 1);
          setDialogueHistory((prev) => [
            ...prev,
            {
              role: 'system',
              text: msg.message,
              timestamp: new Date().toISOString(),
            },
          ]);
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
          setTimeout(() => fetchResultLoop(evalId), 3000);
          break;

        case 'interrupted':
          stopCurrentPlayback();
          setIsProcessing(false);
          setCurrentHint('已暂停');
          break;

        case 'error':
          setCurrentHint('错误: ' + msg.message);
          setIsProcessing(false);
          break;
      }
    },
    [evalId, isRecording, playAudioBuffer, stopCurrentPlayback, stopRecording, startSilenceTimer, stopSilenceTimer, isGeneratingFirstQuestion]
  );

  const connectWebSocket = useCallback(
    (url: string, id: string) => {
      setConnectionStatus('connecting');
      const ws = new WebSocket(url);
      wsRef.current = ws;
      ws.binaryType = 'arraybuffer';

      ws.onopen = () => {
        setConnectionStatus('connected');
        ws.send(
          JSON.stringify({
            type: 'start_exam',
            timeout_strategy: 'prompt',
            silence_thresholds: silenceThresholds,
          })
        );
        // 告诉前端正在生成第一个问题
        setIsGeneratingFirstQuestion(true);
        setCurrentHint('考官正在出题，请稍候...');
      };

      ws.onmessage = async (event) => {
        await handleWebSocketMessage(event.data);
      };

      ws.onerror = () => {
        setConnectionStatus('error');
        setCurrentHint('连接出错，请刷新页面重试');
        setIsGeneratingFirstQuestion(false);
      };

      ws.onclose = () => {
        setConnectionStatus('idle');
        stopCurrentPlayback();
        stopSilenceTimer();
        stopRecordingTimer();
        clearAutoStopTimer();
        setIsGeneratingFirstQuestion(false);
        if (isRecording) {
          setIsRecording(false);
          setIsProcessing(false);
        }
      };
    },
    [handleWebSocketMessage, stopCurrentPlayback, stopSilenceTimer, stopRecordingTimer, clearAutoStopTimer, isRecording]
  );

  const fetchResultLoop = useCallback(async (id: string) => {
    try {
      const data = await fetchExamResult(id);
      if (data.status === 'completed') {
        setFinalResult(data);
        setPhase('result');
      } else {
        setTimeout(() => fetchResultLoop(id), 2000);
      }
    } catch (err) {
      setTimeout(() => fetchResultLoop(id), 5000);
    }
  }, []);

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
    setIsGeneratingFirstQuestion(false);
  }, [stopCurrentPlayback, stopRecording, isRecording, stopSilenceTimer, stopRecordingTimer, clearAutoStopTimer]);

  // ★★★ 核心修改：点击按钮立即跳转到考试界面，后台异步初始化 ★★★
  const handleStartSubmit = async () => {
    // 1. 立即切换到考试界面
    setPhase('examining');
    setIsInitializing(true);
    setInitError(null);
    setConnectionStatus('connecting');
    setCurrentHint('正在连接考官，请稍候...');
    setDialogueHistory([]);

    // 2. 初始化音频上下文（必须在用户交互中）
    await initAudioContext();

    // 3. 后台异步调用 API
    try {
      if (oralMode === 'debug') {
        if (!originalQuestion.trim() || !originalAnswer.trim()) {
          throw new Error('请填写考试主题和预习答案');
        }
        const resp = await startOralExam({
          original_question: originalQuestion,
          original_answer: originalAnswer,
          student_id: studentId || undefined,
        });
        setEvalId(resp.evaluation_id);
        setIsInitializing(false);
        connectWebSocket(resp.websocket_url, resp.evaluation_id);
      } else {
        // OS实验模式
        if (!experimentRequirement.trim() || !beforeZip || !afterZip) {
          throw new Error('请填写实验要求，并上传修改前/后的代码 ZIP 文件');
        }
        const formData = new FormData();
        formData.append('experiment_requirement', experimentRequirement);
        formData.append('before_zip', beforeZip);
        formData.append('after_zip', afterZip);
        formData.append('num_questions', String(osNumQuestions));
        formData.append('course', osType);
        if (studentId.trim()) formData.append('student_id', studentId);

        const resp = await fetch(`${API_BASE_URL}/api/oral-exam/os-start`, {
          method: 'POST',
          body: formData,
        });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();
        setEvalId(data.evaluation_id);
        setIsInitializing(false);
        connectWebSocket(data.websocket_url, data.evaluation_id);
      }
    } catch (err: any) {
      setIsInitializing(false);
      setInitError(err.message || '启动失败');
      setConnectionStatus('error');
      setCurrentHint('启动失败: ' + err.message);
    }
  };

  useEffect(() => {
    return () => {
      stopSilenceTimer();
      stopRecordingTimer();
      clearAutoStopTimer();
      if (wsRef.current) wsRef.current.close();
    };
  }, [stopSilenceTimer, stopRecordingTimer, clearAutoStopTimer]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && phase === 'examining' && !isProcessing && !isExaminerSpeaking && connectionStatus === 'connected') {
        e.preventDefault();
        toggleRecording();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [phase, isProcessing, isExaminerSpeaking, toggleRecording, connectionStatus]);

  const canStart = oralMode === 'debug'
    ? originalQuestion.trim() && originalAnswer.trim()
    : experimentRequirement.trim() && beforeZip && afterZip;

  // 渲染准备界面（考试设置）
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

          {/* 模式标识 */}
          <div className="mb-4">
            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-indigo-100 text-indigo-800">
              {osOnly ? 'OS实验考试' : '语音调试'}
            </span>
          </div>

          <div className="space-y-4">
            {oralMode === 'debug' ? (
              <>
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
              </>
            ) : (
              <>
                {/* OS类型选择 */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">操作系统类型</label>
                  <div className="flex gap-3">
                    <button
                      type="button"
                      onClick={() => setOsType('ucore')}
                      className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all ${
                        osType === 'ucore'
                          ? 'bg-indigo-600 text-white shadow-sm'
                          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                      }`}
                    >
                      uCore
                    </button>
                    <button
                      type="button"
                      onClick={() => setOsType('rcore')}
                      className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-all ${
                        osType === 'rcore'
                          ? 'bg-indigo-600 text-white shadow-sm'
                          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                      }`}
                    >
                      rCore
                    </button>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    选择后将使用对应的课程知识库（ChromaDB）增强提问质量
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">实验要求描述</label>
                  <textarea
                    value={experimentRequirement}
                    onChange={(e) => setExperimentRequirement(e.target.value)}
                    placeholder="例如：实验三：内存管理。实现一个简化的内存分配器..."
                    className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500 h-24"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">修改前代码 (ZIP)</label>
                  <input
                    type="file"
                    accept=".zip"
                    onChange={(e) => setBeforeZip(e.target.files?.[0] || null)}
                    className="w-full p-2 border border-gray-300 rounded-md"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">修改后代码 (ZIP)</label>
                  <input
                    type="file"
                    accept=".zip"
                    onChange={(e) => setAfterZip(e.target.files?.[0] || null)}
                    className="w-full p-2 border border-gray-300 rounded-md"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">生成问题数量</label>
                  <input
                    type="number"
                    min={1}
                    max={10}
                    value={osNumQuestions}
                    onChange={(e) => setOsNumQuestions(Number(e.target.value))}
                    className="w-full p-2 border border-gray-300 rounded-md"
                  />
                </div>
              </>
            )}
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
            <button
              onClick={handleStartSubmit}
              disabled={!canStart}
              className="w-full bg-indigo-600 text-white py-3 rounded-md hover:bg-indigo-700 disabled:bg-gray-400 font-medium transition-colors"
            >
              开始考试
            </button>
          </div>
        </div>
      </div>
    );
  }

  // 渲染考试界面（左侧控制面板 + 右侧对话记录）
  if (phase === 'examining') {
    const isNotConnected = connectionStatus !== 'connected';
    // 是否显示"正在生成第一个问题"的占位符
    const showGeneratingPlaceholder = isGeneratingFirstQuestion && dialogueHistory.length === 0;

    return (
      <div className="max-w-7xl mx-auto p-4">
        <div className="grid grid-cols-1 lg:grid-cols-[360px_1fr] gap-6 items-start">
          {/* 左侧语音控制面板 */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 flex flex-col items-center h-fit sticky top-6">
            <div className="w-full flex justify-between items-center mb-6 pb-4 border-b border-gray-100">
              <div className="flex items-center gap-2">
                <div
                  className={`w-2.5 h-2.5 rounded-full ${
                    connectionStatus === 'connected' 
                      ? 'bg-green-500' 
                      : connectionStatus === 'error'
                      ? 'bg-red-500'
                      : 'bg-yellow-500 animate-pulse'
                  }`}
                ></div>
                <span className="text-sm font-medium text-gray-700">
                  {connectionStatus === 'connected' 
                    ? '考试中' 
                    : connectionStatus === 'error'
                    ? '连接错误'
                    : '连接中...'}
                </span>
              </div>
              <button
                onClick={endExam}
                className="text-red-600 text-sm border border-red-200 bg-red-50 hover:bg-red-100 px-3 py-1 rounded-md font-medium transition-colors"
              >
                结束
              </button>
            </div>

            {/* 考官/等待状态图标 */}
            <div
              className={`w-24 h-24 rounded-full flex items-center justify-center text-4xl mb-5 transition-all ${
                isNotConnected
                  ? 'bg-gray-100 border-gray-300'
                  : isExaminerSpeaking
                  ? 'bg-indigo-50 border-indigo-500 animate-pulse'
                  : isWaitingForAnswer
                  ? 'bg-amber-50 border-amber-400'
                  : 'bg-gray-50 border-gray-200'
              } border-2`}
            >
              {isNotConnected 
                ? '⏳' 
                : isExaminerSpeaking 
                ? '🗣️' 
                : isWaitingForAnswer 
                ? '⏳' 
                : '✋'}
            </div>

            <p className="text-base font-medium text-gray-700 mb-6 text-center min-h-[3rem] leading-snug px-2">
              {isNotConnected
                ? (connectionStatus === 'error' ? '连接失败，请刷新页面重试' : '正在连接考官，请稍候...')
                : currentHint || '准备就绪'}
            </p>

            {/* 静音计时条仅当已连接且等待回答时显示 */}
            {!isNotConnected && isWaitingForAnswer && !isRecording && !isProcessing && !isExaminerSpeaking && (
              <div className="w-full mb-6">
                <div className="flex justify-between text-xs text-gray-500 mb-1 font-medium">
                  <span className={timeoutLevel > 0 ? 'text-red-600' : ''}>等待回答计时</span>
                  <span className={timeoutLevel > 0 ? 'text-red-600' : ''}>{silenceDuration}秒 / 240秒</span>
                </div>
                <div className="w-full h-2 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-1000 rounded-full ${
                      timeoutLevel === 0
                        ? 'bg-green-500'
                        : timeoutLevel === 1
                        ? 'bg-yellow-500'
                        : timeoutLevel === 2
                        ? 'bg-orange-500'
                        : 'bg-red-500'
                    }`}
                    style={{ width: `${Math.min((silenceDuration / 240) * 100, 100)}%` }}
                  ></div>
                </div>
              </div>
            )}

            {/* 录音按钮 */}
            <div className="relative mb-3">
              <button
                onClick={toggleRecording}
                disabled={isNotConnected || isProcessing || isExaminerSpeaking}
                className={`w-24 h-24 rounded-full flex items-center justify-center text-white shadow-lg transition-all duration-200 ${
                  isRecording
                    ? 'bg-red-500 scale-110 shadow-red-300 ring-4 ring-red-200'
                    : isNotConnected || isExaminerSpeaking || isProcessing
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
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
                    />
                  </svg>
                )}
                {isRecording && (
                  <div
                    className="absolute -inset-2 rounded-full border-4 border-gray-200 border-t-indigo-600 animate-spin"
                    style={{ animationDuration: '3s' }}
                  />
                )}
              </button>
              {isRecording && (
                <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 text-sm font-mono font-bold text-red-500">
                  {formatTime(recordingProgress)} / 0:30
                </div>
              )}
            </div>

            <p className="text-sm text-gray-500 mt-4 text-center font-medium">
              {isNotConnected
                ? '连接中...'
                : isRecording
                ? '点击结束录音'
                : isExaminerSpeaking
                ? '考官说话中...'
                : isProcessing
                ? '识别中...'
                : '点击开始录音'}
            </p>

            {/* 快捷指令（仅已连接时显示） */}
            {!isNotConnected && isWaitingForAnswer && !isRecording && !isProcessing && !isExaminerSpeaking && (
              <div className="flex gap-2 justify-center mt-5 flex-wrap">
                {['请重复', '解释一下', '给点提示', '下一题'].map((cmd) => (
                  <button
                    key={cmd}
                    onClick={() => {
                      if (wsRef.current?.readyState === WebSocket.OPEN) {
                        wsRef.current.send(JSON.stringify({ type: 'text_data', text: cmd }));
                        setIsProcessing(true);
                        setCurrentHint('考官思考中...');
                        stopSilenceTimer();
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

          {/* 右侧对话面板 */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 flex flex-col h-[calc(100vh-140px)] min-h-[500px]">
            <h3 className="text-base font-semibold text-gray-900 mb-4 pb-3 border-b-2 border-gray-100 flex items-center justify-between">
              <span>对话记录</span>
              {dialogueHistory.length > 0 && (
                <span className="text-xs text-gray-500 font-normal bg-gray-100 px-2 py-1 rounded-full">
                  {dialogueHistory.filter((d) => !d.isTyping).length} 条消息
                </span>
              )}
            </h3>
            <div className="flex-1 overflow-y-auto space-y-3 pr-2">
              {isNotConnected ? (
                <div className="text-center text-gray-400 mt-10 italic">
                  <div className="text-5xl mb-4">🔌</div>
                  正在连接考官，请稍候...
                  <div className="mt-3 animate-pulse text-sm">连接成功后自动开始考试</div>
                </div>
              ) : showGeneratingPlaceholder ? (
                <div className="text-center text-gray-400 mt-10 italic">
                  <div className="text-5xl mb-4">🤖</div>
                  <div className="animate-pulse text-gray-600">考官正在出题，请稍候...</div>
                  <div className="mt-2 text-sm">AI 正在分析您的信息并生成第一个问题</div>
                </div>
              ) : dialogueHistory.length === 0 ? (
                <div className="text-center text-gray-400 mt-10 italic">
                  <div className="text-5xl mb-4">💬</div>
                  等待考官第一个问题...
                </div>
              ) : (
                dialogueHistory.map((turn, idx) => (
                  <div
                    key={idx}
                    className={`p-4 rounded-lg border-l-4 ${
                      turn.role === 'examiner'
                        ? 'bg-gray-50 border-indigo-500'
                        : turn.role === 'student'
                        ? 'bg-green-50 border-green-500'
                        : 'bg-amber-50 border-amber-400'
                    } ${turn.isTyping ? 'animate-pulse' : ''}`}
                  >
                    <div className="flex justify-between items-center mb-2 text-xs text-gray-500">
                      <span className="font-semibold flex items-center gap-1.5">
                        {turn.role === 'examiner' ? '👨‍🏫 考官' : turn.role === 'student' ? '🎓 你' : '🔔 系统'}
                        {turn.role === 'examiner' && turn.depth && turn.depth > 0 && (
                          <span className="text-amber-600 text-[10px] bg-amber-100 px-1.5 py-0.5 rounded">
                            追问{turn.depth}
                          </span>
                        )}
                      </span>
                      <span className="text-[11px]">
                        {new Date(turn.timestamp).toLocaleTimeString([], {
                          hour: '2-digit',
                          minute: '2-digit',
                          second: '2-digit',
                        })}
                      </span>
                    </div>
                    <div className="text-gray-800 text-sm leading-relaxed">
                      {turn.isTyping ? (
                        <div className="flex items-center gap-2 text-gray-400 italic">
                          <div className="w-4 h-4 border-2 border-gray-300 border-t-indigo-500 rounded-full animate-spin"></div>
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
              <span className="px-3 py-1 bg-green-500 rounded-full text-xs font-semibold mr-3">
                {finalResult.overall_assessment?.understanding_level || '评估完成'}
              </span>
              <span className="text-sm opacity-90">
                置信度: {((finalResult.overall_assessment?.confidence || 0) * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-gray-200 leading-relaxed text-sm">
              {finalResult.overall_assessment?.reasoning || '评估完成，请查看详细评分。'}
            </p>
          </div>
        </div>
      </div>
    );
  }

  return null;
};

// ==================== 主应用组件 ====================
const ExamEvaluationApp: React.FC = () => {
  const [appMode, setAppMode] = useState<'os' | 'normal'>('os');
  const [examMode, setExamMode] = useState<'text' | 'oral'>('text');
  const [phase, setPhase] = useState<'start' | 'exam' | 'result'>('start');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [evaluationId, setEvaluationId] = useState<string>('');
  const [examQuestions, setExamQuestions] = useState<ExamQuestionsResponse | null>(null);
  const [finalResult, setFinalResult] = useState<FinalEvaluationResult | null>(null);

  const handleStartOriginal = async (data: StartEvaluationRequest) => {
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

  const handleStartOS = async (formData: FormData) => {
    setLoading(true);
    setError(null);
    try {
      const response = await startOSEvaluation(formData);
      setEvaluationId(response.evaluation_id);
      setExamQuestions(response);
      setPhase('exam');
    } catch (err: any) {
      setError(`启动OS实验评估失败: ${err.message}`);
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
            onClick={() => setAppMode('os')}
            className={`px-6 py-3 rounded-lg font-medium transition-all ${
              appMode === 'os' ? 'bg-indigo-600 text-white shadow-lg' : 'bg-white text-gray-700 border border-gray-300'
            }`}
          >
            OS实验考试
          </button>
          <button
            onClick={() => setAppMode('normal')}
            className={`px-6 py-3 rounded-lg font-medium transition-all ${
              appMode === 'normal' ? 'bg-blue-600 text-white shadow-lg' : 'bg-white text-gray-700 border border-gray-300'
            }`}
          >
            普通模式
          </button>
        </div>

        {error && (
          <div className="max-w-3xl mx-auto mb-6 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
            <strong>错误:</strong> {error}
          </div>
        )}

        {appMode === 'os' ? (
          <OralExamination osOnly />
        ) : (
          <>
            <div className="flex justify-center space-x-4 mb-6">
              <button
                onClick={() => setExamMode('text')}
                className={`px-4 py-2 rounded-md font-medium transition-all ${
                  examMode === 'text' ? 'bg-blue-600 text-white shadow' : 'bg-white text-gray-700 border border-gray-300'
                }`}
              >
                文字考试
              </button>
              <button
                onClick={() => setExamMode('oral')}
                className={`px-4 py-2 rounded-md font-medium transition-all ${
                  examMode === 'oral' ? 'bg-indigo-600 text-white shadow' : 'bg-white text-gray-700 border border-gray-300'
                }`}
              >
                语音口试
              </button>
            </div>
            {examMode === 'text' ? (
              <>
                {phase === 'start' && (
                  <StartPhase onStartOriginal={handleStartOriginal} onStartOS={handleStartOS} loading={loading} />
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
                {phase === 'result' && finalResult && <ResultPhase result={finalResult} />}
              </>
            ) : (
              <OralExamination />
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default ExamEvaluationApp;