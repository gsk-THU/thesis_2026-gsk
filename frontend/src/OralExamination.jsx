/**
 * OralExamination.jsx - 语音口试组件（点击切换录音模式）
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';

const API_BASE_URL = 'http://localhost:8000';

async function startOralExam(data) {
  const res = await fetch(`${API_BASE_URL}/api/oral-exam/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function fetchExamResult(evaluation_id) {
  const res = await fetch(`${API_BASE_URL}/api/evaluation/${evaluation_id}`);
  if (!res.ok) throw new Error('获取结果失败');
  return res.json();
}

// 代码展示组件
const CodeBlock = ({ code, index }) => {
  const [copied, setCopied] = useState(false);
  
  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  
  const highlightCode = (code) => {
    let highlighted = code
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
    
    highlighted = highlighted.replace(/(['"])(.*?)\1/g, '<span style="color: #a6e3a1;">$1$2$1</span>');
    
    const keywords = ['def', 'return', 'if', 'else', 'elif', 'for', 'while', 'class', 'import', 'from', 'as', 'try', 'except', 'with', 'lambda'];
    keywords.forEach(kw => {
      const regex = new RegExp(`\\b${kw}\\b`, 'g');
      highlighted = highlighted.replace(regex, `<span style="color: #cba6f7; font-weight: 600;">${kw}</span>`);
    });
    
    highlighted = highlighted.replace(/(#.*$|\/\/.*$)/gm, '<span style="color: #6c7086; font-style: italic;">$1</span>');
    highlighted = highlighted.replace(/\b(\d+)\b/g, '<span style="color: #fab387;">$1</span>');
    
    return { __html: highlighted };
  };
  
  const detectLang = (code) => {
    if (code.includes('def ') || code.includes('import ')) return 'python';
    if (code.includes('function') || code.includes('const ')) return 'javascript';
    if (code.includes('public class')) return 'java';
    if (code.includes('#include')) return 'cpp';
    return 'code';
  };
  
  const lang = detectLang(code);
  
  return (
    <div style={{
      backgroundColor: '#1e1e2e',
      borderRadius: '6px',
      margin: '8px 0',
      overflow: 'hidden',
      border: '1px solid #313244'
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '6px 10px',
        backgroundColor: '#181825',
        borderBottom: '1px solid #313244'
      }}>
        <span style={{
          fontSize: '11px',
          color: '#cdd6f4',
          fontWeight: '600',
          textTransform: 'uppercase'
        }}>
          {lang}
        </span>
        <button
          onClick={handleCopy}
          style={{
            backgroundColor: copied ? '#a6e3a1' : '#313244',
            color: copied ? '#1e1e2e' : '#cdd6f4',
            border: 'none',
            padding: '2px 8px',
            borderRadius: '4px',
            fontSize: '11px',
            cursor: 'pointer'
          }}
        >
          {copied ? '✓' : '复制'}
        </button>
      </div>
      <div style={{
        padding: '10px',
        overflowX: 'auto',
        fontFamily: 'monospace',
        fontSize: '12px',
        lineHeight: '1.5',
        color: '#cdd6f4'
      }}>
        <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
          <code dangerouslySetInnerHTML={highlightCode(code)} />
        </pre>
      </div>
    </div>
  );
};

// 考官消息渲染组件
const ExaminerMessage = ({ text, codeSnippets, hasCode }) => {
  if (!hasCode || !codeSnippets || codeSnippets.length === 0) {
    return <div style={{ whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>{text}</div>;
  }
  
  const parts = text.split('[代码片段]');
  
  return (
    <div style={{ lineHeight: '1.6' }}>
      {parts.map((part, index) => (
        <React.Fragment key={index}>
          {part && <div style={{ whiteSpace: 'pre-wrap', margin: '4px 0' }}>{part}</div>}
          {index < codeSnippets.length && <CodeBlock code={codeSnippets[index]} index={index} />}
        </React.Fragment>
      ))}
    </div>
  );
};

export default function OralExamination() {
  const [phase, setPhase] = useState('prepare');
  const [originalQuestion, setOriginalQuestion] = useState('');
  const [originalAnswer, setOriginalAnswer] = useState('');
  
  const [evalId, setEvalId] = useState('');
  const [connectionStatus, setConnectionStatus] = useState('idle');
  
  const [isExaminerSpeaking, setIsExaminerSpeaking] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [dialogueHistory, setDialogueHistory] = useState([]);
  const [currentHint, setCurrentHint] = useState('');
  
  const [silenceDuration, setSilenceDuration] = useState(0);
  const [timeoutLevel, setTimeoutLevel] = useState(0);
  const [isWaitingForAnswer, setIsWaitingForAnswer] = useState(false);
  
  const [recordingProgress, setRecordingProgress] = useState(0);
  const maxRecordingTime = 30;

  const wsRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioContextRef = useRef(null);
  const currentSourceRef = useRef(null);
  const currentAudioIdRef = useRef(0);
  const playLockRef = useRef(false);
  
  const silenceTimerRef = useRef(null);
  const recordingTimerRef = useRef(null);
  const autoStopTimerRef = useRef(null); // 【新增】自动停止定时器
  
  const silenceThresholds = [50, 100, 150];

  // 初始化音频上下文
  const initAudioContext = useCallback(async () => {
    if (!audioContextRef.current) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) {
        alert('浏览器不支持 Web Audio API');
        return false;
      }
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
  const playAudioBuffer = useCallback(async (arrayBuffer) => {
    if (!audioContextRef.current) await initAudioContext();
    const ctx = audioContextRef.current;
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
        startSilenceTimer();
      };
      source.start(0);
    } catch (e) {
      console.error('[Audio] 播放失败:', e);
      playLockRef.current = false;
      setIsExaminerSpeaking(false);
    }
  }, []);

  // 静默计时器
  const startSilenceTimer = useCallback(() => {
    stopSilenceTimer();
    setSilenceDuration(0);
    setTimeoutLevel(0);
    silenceTimerRef.current = setInterval(() => {
      setSilenceDuration(prev => {
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

  // 【新增】清理自动停止定时器
  const clearAutoStopTimer = useCallback(() => {
    if (autoStopTimerRef.current) {
      clearTimeout(autoStopTimerRef.current);
      autoStopTimerRef.current = null;
    }
  }, []);

  // 【关键修改】切换录音状态 - 点击开始，再点击结束
  const toggleRecording = useCallback(async () => {
    // 如果正在录音，则停止
    if (isRecording) {
      stopRecording();
      return;
    }
    
    // 如果考官正在说话，先打断
    if (isExaminerSpeaking) {
      stopCurrentPlayback();
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'interrupt' }));
      }
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    // 如果正在处理中，不能开始
    if (isProcessing) return;
    
    // 开始录音
    await startRecording();
  }, [isRecording, isProcessing, isExaminerSpeaking, stopCurrentPlayback]);

  // 开始录音（内部实现）
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
      
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          const reader = new FileReader();
          reader.onloadend = () => {
            const base64 = reader.result.split(',')[1];
            wsRef.current.send(JSON.stringify({ type: 'audio_data', data: base64 }));
            setIsProcessing(true);
            setCurrentHint('识别中...');
          };
          reader.readAsDataURL(audioBlob);
        }
        stream.getTracks().forEach(track => track.stop());
        stopRecordingTimer();
        clearAutoStopTimer(); // 【新增】清理自动停止定时器
      };
      
      mediaRecorder.start(100);
      setIsRecording(true);
      stopSilenceTimer();
      startRecordingTimer();
      
      // 【新增】30秒自动停止的安全机制
      autoStopTimerRef.current = setTimeout(() => {
        if (mediaRecorderRef.current?.state === 'recording') {
          stopRecording();
        }
      }, 30000);
      
    } catch (err) {
      alert('无法访问麦克风，请检查权限设置');
    }
  };

  // 停止录音（内部实现）
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsProcessing(true);
      clearAutoStopTimer(); // 【新增】清理自动停止定时器
    }
  }, [clearAutoStopTimer]);

  // WebSocket消息处理
  const handleWebSocketMessage = useCallback(async (data) => {
    lastProgressTimeRef.current = Date.now();
    
    if (data instanceof ArrayBuffer) {
      await playAudioBuffer(data);
      return;
    }

    const msg = JSON.parse(data);
    
    switch (msg.type) {
      case 'examiner_typing':
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
        
        const hints = {
          'repeat': '考官重复问题',
          'explanation': '考官解释中',
          'hint': '考官提示中',
          'follow_up': '考官深入追问',
          'new_topic': '切换新话题',
          'question': '考官提出新问题'
        };
        setCurrentHint(msg.has_code ? `💻 ${hints[msg.response_type] || '考官展示代码'}` : (hints[msg.response_type] || '考官提问中'));
        
        setIsWaitingForAnswer(false);
        setSilenceDuration(0);
        setTimeoutLevel(0);
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
        break;
        
      case 'processing':
        setIsProcessing(true);
        setCurrentHint('识别中...');
        stopSilenceTimer();
        break;
        
      case 'transcription':
        setDialogueHistory(prev => [...prev, {
          role: 'student',
          text: msg.text,
          type: 'answer',
          timestamp: new Date().toISOString()
        }]);
        break;
        
      case 'silence_reminder':
      case 'timeout_reminder':
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
        // 【关键】录音中时先停止
        if (isRecording) {
          stopRecording();
        }
        setTimeout(() => fetchResultLoop(evalId), 3000);
        break;
        
      case 'interrupted':
        stopCurrentPlayback();
        setCurrentHint('已暂停');
        break;
        
      case 'error':
        setCurrentHint('错误: ' + msg.message);
        break;
    }
  }, [evalId, isRecording, playAudioBuffer, stopCurrentPlayback, stopRecording, startSilenceTimer, stopSilenceTimer]);

  const lastProgressTimeRef = useRef(Date.now());

  // WebSocket连接
  const connectWebSocket = useCallback((url, id) => {
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
        silence_thresholds: silenceThresholds
      }));
      setCurrentHint('连接成功，考官准备中...');
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
      // 如果还在录音，强制停止
      if (isRecording) {
        setIsRecording(false);
        setIsProcessing(false);
      }
    };
  }, [handleWebSocketMessage, stopCurrentPlayback, stopSilenceTimer, stopRecordingTimer, clearAutoStopTimer, isRecording]);

  const fetchResultLoop = useCallback(async (id) => {
    try {
      const data = await fetchExamResult(id);
      if (data.status === 'completed') {
        setExamResult(data);
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
  }, [stopCurrentPlayback, stopRecording, isRecording, stopSilenceTimer, stopRecordingTimer, clearAutoStopTimer]);

  const handleStartSubmit = async () => {
    if (!originalQuestion.trim() || !originalAnswer.trim()) {
      alert('请填写考试主题和预习答案');
      return;
    }
    await initAudioContext();
    try {
      const resp = await startOralExam({
        original_question: originalQuestion,
        original_answer: originalAnswer
      });
      setEvalId(resp.evaluation_id);
      connectWebSocket(resp.websocket_url, resp.evaluation_id);
      setPhase('connecting');
    } catch (err) {
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

  // 【可选】键盘快捷键 - 空格键切换录音
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.code === 'Space' && phase === 'examining' && !isProcessing && !isExaminerSpeaking) {
        e.preventDefault();
        toggleRecording();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [phase, isProcessing, isExaminerSpeaking, isRecording, toggleRecording]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // 渲染左侧语音面板 - 【关键修改】按钮改为 onClick 模式
  const renderVoicePanel = () => (
    <div style={{
      backgroundColor: 'white',
      borderRadius: '12px',
      boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
      padding: '20px',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      height: 'fit-content',
      position: 'sticky',
      top: '20px'
    }}>
      {/* 状态栏 */}
      <div style={{
        width: '100%',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '20px',
        paddingBottom: '16px',
        borderBottom: '1px solid #f3f4f6'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div style={{
            width: '10px',
            height: '10px',
            borderRadius: '50%',
            backgroundColor: connectionStatus === 'connected' ? '#10b981' : '#ef4444',
            boxShadow: connectionStatus === 'connected' ? '0 0 0 3px rgba(16, 185, 129, 0.2)' : 'none'
          }}></div>
          <span style={{ fontSize: '14px', fontWeight: '500', color: '#374151' }}>
            {connectionStatus === 'connected' ? '考试中' : '未连接'}
          </span>
        </div>
        <button 
          onClick={endExam} 
          style={{
            color: '#dc2626',
            fontSize: '13px',
            border: '1px solid #fecaca',
            background: '#fef2f2',
            cursor: 'pointer',
            padding: '4px 10px',
            borderRadius: '6px',
            fontWeight: '500'
          }}
        >
          结束
        </button>
      </div>

      {/* 考官头像 */}
      <div style={{
        width: '100px',
        height: '100px',
        borderRadius: '50%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '40px',
        backgroundColor: isExaminerSpeaking ? '#eef2ff' : isWaitingForAnswer ? '#fef3c7' : '#f9fafb',
        border: `3px solid ${isExaminerSpeaking ? '#4f46e5' : isWaitingForAnswer ? '#f59e0b' : '#e5e7eb'}`,
        marginBottom: '16px',
        animation: isExaminerSpeaking ? 'pulse 2s infinite' : 'none'
      }}>
        {isExaminerSpeaking ? '🗣️' : isWaitingForAnswer ? '⏳' : '✋'}
      </div>

      {/* 提示文字 */}
      <p style={{
        fontSize: '15px',
        fontWeight: '500',
        color: '#374151',
        marginBottom: '20px',
        textAlign: 'center',
        minHeight: '44px'
      }}>
        {currentHint || '准备就绪'}
      </p>

      {/* 静默进度条 */}
      {isWaitingForAnswer && !isRecording && (
        <div style={{ width: '100%', marginBottom: '20px' }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: '12px',
            color: timeoutLevel > 0 ? '#dc2626' : '#6b7280',
            marginBottom: '4px'
          }}>
            <span>等待回答计时</span>
            <span>{silenceDuration}秒 / 150秒</span>
          </div>
          <div style={{
            width: '100%',
            height: '6px',
            backgroundColor: '#f3f4f6',
            borderRadius: '3px',
            overflow: 'hidden'
          }}>
            <div style={{
              height: '100%',
              width: `${Math.min((silenceDuration / 150) * 100, 100)}%`,
              backgroundColor: timeoutLevel === 0 ? '#10b981' : timeoutLevel === 1 ? '#f59e0b' : timeoutLevel === 2 ? '#f97316' : '#dc2626',
              transition: 'all 0.3s ease'
            }}></div>
          </div>
        </div>
      )}

      {/* 【关键修改】录音按钮 - 点击切换模式 */}
      <div style={{ position: 'relative', marginBottom: '12px' }}>
        <button
          onClick={toggleRecording}
          disabled={isProcessing || isExaminerSpeaking}
          style={{
            width: '90px',
            height: '90px',
            borderRadius: '50%',
            border: 'none',
            fontSize: '28px',
            cursor: (isProcessing || isExaminerSpeaking) ? 'not-allowed' : 'pointer',
            backgroundColor: isRecording ? '#ef4444' : (isExaminerSpeaking || isProcessing) ? '#fbbf24' : '#4f46e5',
            color: 'white',
            boxShadow: isRecording ? '0 0 0 4px rgba(239, 68, 68, 0.3)' : '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
            transform: isRecording ? 'scale(1.1)' : 'scale(1)',
            opacity: (isProcessing || isExaminerSpeaking) ? 0.6 : 1,
            transition: 'all 0.2s ease',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          {isRecording ? (
            // 录音中显示停止方块
            <div style={{ width: '30px', height: '30px', backgroundColor: 'white', borderRadius: '4px' }}></div>
          ) : (
            // 未录音显示麦克风
            '🎤'
          )}
          
          {/* 录音进度圆环 */}
          {isRecording && (
            <div style={{
              position: 'absolute',
              top: '-4px',
              left: '-4px',
              right: '-4px',
              bottom: '-4px',
              borderRadius: '50%',
              border: '3px solid #e5e7eb',
              borderTopColor: '#4f46e5',
              transform: `rotate(${recordingProgress * 12}deg)`
            }} />
          )}
        </button>
        
        {/* 录音时长显示 */}
        {isRecording && (
          <div style={{
            position: 'absolute',
            bottom: '-24px',
            left: '50%',
            transform: 'translateX(-50%)',
            fontSize: '12px',
            fontWeight: '600',
            color: isRecording ? '#ef4444' : '#4f46e5',
            fontFamily: 'monospace'
          }}>
            {formatTime(recordingProgress)} / 0:30
          </div>
        )}
      </div>

      {/* 【修改】按钮下方提示文字 */}
      <p style={{
        fontSize: '12px',
        color: '#9ca3af',
        marginTop: '20px',
        textAlign: 'center'
      }}>
        {isRecording ? '点击结束录音' : isExaminerSpeaking ? '考官说话中...' : isProcessing ? '处理中...' : '点击开始录音'}
      </p>

      {/* 快捷指令 */}
      {isWaitingForAnswer && !isRecording && (
        <div style={{
          display: 'flex',
          gap: '6px',
          justifyContent: 'center',
          marginTop: '16px',
          flexWrap: 'wrap'
        }}>
          {['请重复', '解释一下', '给点提示', '下一题'].map((cmd) => (
            <button
              key={cmd}
              onClick={() => {
                if (wsRef.current?.readyState === WebSocket.OPEN) {
                  wsRef.current.send(JSON.stringify({ type: 'text_data', text: cmd }));
                }
              }}
              style={{
                padding: '5px 10px',
                borderRadius: '12px',
                border: '1px solid #e5e7eb',
                backgroundColor: 'white',
                fontSize: '12px',
                color: '#374151',
                cursor: 'pointer'
              }}
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
    <div style={{
      backgroundColor: 'white',
      borderRadius: '12px',
      boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
      padding: '20px',
      display: 'flex',
      flexDirection: 'column',
      height: 'calc(100vh - 140px)',
      minHeight: '500px'
    }}>
      <h3 style={{
        fontSize: '16px',
        fontWeight: '600',
        color: '#111827',
        marginBottom: '16px',
        paddingBottom: '12px',
        borderBottom: '2px solid #f3f4f6'
      }}>
        对话记录 {dialogueHistory.length > 0 && `(${dialogueHistory.filter(d => !d.isTyping).length})`}
      </h3>
      
      <div style={{
        flex: 1,
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px'
      }}>
        {dialogueHistory.length === 0 ? (
          <div style={{ 
            textAlign: 'center', 
            color: '#9ca3af', 
            marginTop: '40px',
            fontStyle: 'italic' 
          }}>
            <div style={{ fontSize: '48px', marginBottom: '16px' }}>💬</div>
            等待考官第一个问题...
          </div>
        ) : (
          dialogueHistory.map((turn, idx) => (
            <div 
              key={idx} 
              style={{
                padding: '12px 16px',
                borderRadius: '8px',
                backgroundColor: turn.role === 'examiner' ? '#fafafa' : turn.role === 'student' ? '#f0fdf4' : '#fef3c7',
                border: `1px solid ${turn.role === 'examiner' ? '#e5e7eb' : turn.role === 'student' ? '#86efac' : '#fcd34d'}`,
                borderLeft: `4px solid ${turn.role === 'examiner' ? '#4f46e5' : turn.role === 'student' ? '#10b981' : '#f59e0b'}`
              }}
            >
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '8px',
                fontSize: '12px',
                color: '#6b7280'
              }}>
                <span style={{ fontWeight: '600', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  {turn.role === 'examiner' ? '👨‍🏫 考官' : turn.role === 'student' ? '🎓 你' : '🔔 系统'}
                  {turn.role === 'examiner' && turn.depth > 0 && (
                    <span style={{ color: '#f59e0b', fontSize: '11px' }}>追问{turn.depth}</span>
                  )}
                </span>
                <span style={{ fontSize: '11px' }}>
                  {new Date(turn.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'})}
                </span>
              </div>
              
              <div style={{ color: '#1f2937' }}>
                {turn.isTyping ? (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#9ca3af', fontStyle: 'italic' }}>
                    <span className="animate-spin" style={{
                      width: '16px',
                      height: '16px',
                      border: '2px solid #e5e7eb',
                      borderTopColor: '#4f46e5',
                      borderRadius: '50%'
                    }} />
                    考官正在输入...
                  </div>
                ) : turn.role === 'examiner' ? (
                  <ExaminerMessage 
                    text={turn.text} 
                    codeSnippets={turn.codeSnippets} 
                    hasCode={turn.hasCode} 
                  />
                ) : (
                  <div style={{ whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>{turn.text}</div>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );

  // 其余渲染逻辑...
  if (phase === 'prepare') {
    return (
      <div style={{ minHeight: '100vh', backgroundColor: '#f3f4f6', padding: '20px' }}>
        <div style={{ maxWidth: '600px', margin: '0 auto' }}>
          <div style={{ backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', padding: '24px' }}>
            <h1 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '16px', textAlign: 'center' }}>AI语音口试</h1>
            <textarea
              style={{ width: '100%', padding: '12px', border: '1px solid #d1d5db', borderRadius: '6px', marginBottom: '12px', minHeight: '80px' }}
              placeholder="考试主题（如：快速排序算法的时间复杂度分析）"
              value={originalQuestion}
              onChange={(e) => setOriginalQuestion(e.target.value)}
            />
            <textarea
              style={{ width: '100%', padding: '12px', border: '1px solid #d1d5db', borderRadius: '6px', marginBottom: '12px', minHeight: '120px' }}
              placeholder="预习答案（考官将基于此深入提问，支持包含代码片段）"
              value={originalAnswer}
              onChange={(e) => setOriginalAnswer(e.target.value)}
            />
            <button
              style={{ width: '100%', padding: '12px', backgroundColor: (!originalQuestion.trim() || !originalAnswer.trim()) ? '#9ca3af' : '#4f46e5', color: 'white', border: 'none', borderRadius: '6px', cursor: (!originalQuestion.trim() || !originalAnswer.trim()) ? 'not-allowed' : 'pointer' }}
              onClick={handleStartSubmit}
              disabled={!originalQuestion.trim() || !originalAnswer.trim()}
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
      <div style={{ minHeight: '100vh', backgroundColor: '#f3f4f6', padding: '20px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '48px', marginBottom: '16px' }}>⏳</div>
          <p>连接中...</p>
        </div>
      </div>
    );
  }

  if (phase === 'examining') {
    return (
      <div style={{ minHeight: '100vh', backgroundColor: '#f3f4f6', padding: '20px' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'grid', gridTemplateColumns: '360px 1fr', gap: '20px' }}>
          {renderVoicePanel()}
          {renderDialoguePanel()}
        </div>
      </div>
    );
  }

  if (phase === 'grading') {
    return (
      <div style={{ minHeight: '100vh', backgroundColor: '#f3f4f6', padding: '20px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '48px', marginBottom: '16px' }}>⏳</div>
          <p>正在评估...</p>
        </div>
      </div>
    );
  }

  if (phase === 'result') {
    return (
      <div style={{ minHeight: '100vh', backgroundColor: '#f3f4f6', padding: '20px' }}>
        <div style={{ maxWidth: '800px', margin: '0 auto', backgroundColor: 'white', borderRadius: '8px', padding: '24px' }}>
          <h2 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '16px' }}>评估结果</h2>
          <button onClick={() => window.location.reload()} style={{ width: '100%', padding: '12px', backgroundColor: '#4f46e5', color: 'white', border: 'none', borderRadius: '6px' }}>
            再次考试
          </button>
        </div>
      </div>
    );
  }

  return null;
}