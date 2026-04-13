/**
 * OralExamination.jsx - 完整版（进度条 + 延长静默时间 25/50/75秒）
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

export default function OralExamination() {
  const [phase, setPhase] = useState('prepare');
  const [originalQuestion, setOriginalQuestion] = useState('');
  const [originalAnswer, setOriginalAnswer] = useState('');
  const [timeoutStrategy, setTimeoutStrategy] = useState('prompt');
  
  const [evalId, setEvalId] = useState('');
  const [connectionStatus, setConnectionStatus] = useState('idle');
  
  const [isExaminerSpeaking, setIsExaminerSpeaking] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [dialogueHistory, setDialogueHistory] = useState([]);
  const [currentHint, setCurrentHint] = useState('');
  
  // 静默计时（延长到25/50/75秒）
  const [silenceDuration, setSilenceDuration] = useState(0);
  const silenceThresholds = [25, 50, 75]; // 与后端同步
  const maxSilenceTime = 75;
  
  const [timeoutLevel, setTimeoutLevel] = useState(0);
  const [isWaitingForAnswer, setIsWaitingForAnswer] = useState(false);
  const [examResult, setExamResult] = useState(null);
  
  // 录音进度（30秒倒计时）
  const [recordingProgress, setRecordingProgress] = useState(0);
  const maxRecordingTime = 30;

  // Refs
  const wsRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioContextRef = useRef(null);
  const currentSourceRef = useRef(null);
  const currentAudioIdRef = useRef(0);
  const playLockRef = useRef(false);
  
  const silenceTimerRef = useRef(null);
  const recordingTimerRef = useRef(null);
  const lastProgressTimeRef = useRef(Date.now());
  const keepaliveIntervalRef = useRef(null);

  // 初始化 AudioContext（24000Hz匹配edge-tts）
  const initAudioContext = useCallback(async () => {
    if (!audioContextRef.current) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) {
        alert('浏览器不支持 Web Audio API');
        return false;
      }
      
      try {
        audioContextRef.current = new AudioContextClass({ sampleRate: 24000 });
        console.log('[Audio] 采样率:', audioContextRef.current.sampleRate);
      } catch (e) {
        audioContextRef.current = new AudioContextClass();
      }
    }
    
    if (audioContextRef.current.state === 'suspended') {
      await audioContextRef.current.resume();
    }
    return true;
  }, []);

  // 打断播放
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

  // 播放音频
  const playAudioBuffer = useCallback(async (arrayBuffer) => {
    if (!audioContextRef.current) await initAudioContext();
    
    const ctx = audioContextRef.current;
    const thisAudioId = ++currentAudioIdRef.current;

    try {
      playLockRef.current = true;
      setIsExaminerSpeaking(true);

      const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
      console.log('[Audio] 播放时长:', audioBuffer.duration.toFixed(1), '秒');

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
        startSilenceTimer(); // 播放完成后开始静默计时
      };

      source.start(0);
    } catch (e) {
      console.error('[Audio] 播放失败:', e);
      playLockRef.current = false;
      setIsExaminerSpeaking(false);
    }
  }, []);

  // WebSocket 处理
  const handleWebSocketMessage = useCallback(async (data) => {
    lastProgressTimeRef.current = Date.now();
    
    if (data instanceof ArrayBuffer) {
      await playAudioBuffer(data);
      return;
    }

    const msg = JSON.parse(data);
    
    switch (msg.type) {
      case 'audio_start':
        setIsExaminerSpeaking(true);
        setIsWaitingForAnswer(false);
        stopSilenceTimer(); // 考官说话时停止静默计时
        break;
        
      case 'audio_generating':
        setCurrentHint(msg.message || '准备语音...');
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
          timestamp: new Date().toISOString(),
          type: 'answer'
        }]);
        break;
        
      case 'examiner_response':
        setDialogueHistory(prev => [...prev, {
          role: 'examiner',
          type: msg.response_type,
          text: msg.text || '[语音]',
          timestamp: new Date().toISOString()
        }]);
        const hints = {
          'repeat': '考官重复问题',
          'explanation': '考官解释中',
          'hint': '考官提示中',
          'follow_up': '考官追问中',
          'new_topic': '切换话题中'
        };
        setCurrentHint(hints[msg.response_type] || '考官提问中');
        break;
        
      case 'silence_reminder':
      case 'timeout_reminder':
        setCurrentHint(msg.message);
        setTimeoutLevel(msg.level || 1);
        setDialogueHistory(prev => [...prev, {
          role: 'system',
          type: 'reminder',
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
  }, [evalId, playAudioBuffer, stopCurrentPlayback]);

  const connectWebSocket = useCallback((url, id) => {
    setConnectionStatus('connecting');
    
    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;
      ws.binaryType = 'arraybuffer';
      
      ws.onopen = () => {
        setConnectionStatus('connected');
        setPhase('examining');
        setCurrentHint('连接成功');
        
        ws.send(JSON.stringify({ 
          type: 'start_exam',
          timeout_strategy: timeoutStrategy,
          silence_thresholds: silenceThresholds // 25/50/75秒
        }));
        
        keepaliveIntervalRef.current = setInterval(() => {
          if (Date.now() - lastProgressTimeRef.current > 20000) {
            ws.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
          }
        }, 5000);
      };
      
      ws.onmessage = async (event) => {
        await handleWebSocketMessage(event.data);
      };
      
      ws.onerror = () => {
        setConnectionStatus('error');
        setCurrentHint('连接错误');
      };
      
      ws.onclose = () => {
        setConnectionStatus('idle');
        stopCurrentPlayback();
        stopSilenceTimer();
        stopRecordingTimer();
        if (keepaliveIntervalRef.current) clearInterval(keepaliveIntervalRef.current);
      };
    } catch (err) {
      setConnectionStatus('error');
    }
  }, [timeoutStrategy, handleWebSocketMessage, stopCurrentPlayback]);

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

  // 静默计时器（带进度）
  const startSilenceTimer = () => {
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
  };

  const stopSilenceTimer = () => {
    if (silenceTimerRef.current) {
      clearInterval(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
  };
  
  // 录音计时器
  const startRecordingTimer = () => {
    stopRecordingTimer();
    setRecordingProgress(0);
    
    recordingTimerRef.current = setInterval(() => {
      setRecordingProgress(prev => {
        if (prev >= maxRecordingTime) {
          stopRecording();
          return maxRecordingTime;
        }
        return prev + 0.1;
      });
    }, 100);
  };
  
  const stopRecordingTimer = () => {
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
    setRecordingProgress(0);
  };

  const startRecording = async () => {
    const inited = await initAudioContext();
    if (!inited) return;
    
    if (isExaminerSpeaking) {
      stopCurrentPlayback();
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'interrupt' }));
      }
    }
    
    if (isProcessing) return;
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        } 
      });
      
      const mediaRecorder = new MediaRecorder(stream, { 
        mimeType: 'audio/webm;codecs=opus' 
      });
      
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
      };
      
      mediaRecorder.start(100);
      setIsRecording(true);
      stopSilenceTimer();
      startRecordingTimer();
      
    } catch (err) {
      alert('无法访问麦克风');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

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
        timeout_strategy: timeoutStrategy
      });
      setEvalId(resp.evaluation_id);
      connectWebSocket(resp.websocket_url, resp.evaluation_id);
      setPhase('connecting');
    } catch (err) {
      alert('启动失败: ' + err.message);
    }
  };

  const endExam = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'end_exam' }));
    }
    stopSilenceTimer();
    stopRecordingTimer();
    stopCurrentPlayback();
  };

  useEffect(() => {
    return () => {
      stopSilenceTimer();
      stopRecordingTimer();
      if (wsRef.current) wsRef.current.close();
      if (keepaliveIntervalRef.current) clearInterval(keepaliveIntervalRef.current);
    };
  }, []);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // 计算进度条百分比
  const silenceProgressPercent = Math.min((silenceDuration / maxSilenceTime) * 100, 100);
  const recordingProgressPercent = (recordingProgress / maxRecordingTime) * 100;

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#f9fafb', padding: '16px', fontFamily: 'system-ui, sans-serif' }}>
      <div style={{ maxWidth: '768px', margin: '0 auto' }}>
        
        {phase === 'prepare' && (
          <div style={{ backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', padding: '24px', marginBottom: '16px' }}>
            <h1 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '16px', color: '#111827' }}>AI语音口试</h1>
            <textarea
              style={{ width: '100%', padding: '12px', border: '1px solid #d1d5db', borderRadius: '6px', marginBottom: '12px', fontSize: '14px', minHeight: '80px' }}
              placeholder="考试主题"
              value={originalQuestion}
              onChange={(e) => setOriginalQuestion(e.target.value)}
            />
            <textarea
              style={{ width: '100%', padding: '12px', border: '1px solid #d1d5db', borderRadius: '6px', marginBottom: '12px', fontSize: '14px', minHeight: '120px' }}
              placeholder="预习答案"
              value={originalAnswer}
              onChange={(e) => setOriginalAnswer(e.target.value)}
            />
            <select 
              style={{ width: '100%', padding: '10px', border: '1px solid #d1d5db', borderRadius: '6px', marginBottom: '16px' }}
              value={timeoutStrategy}
              onChange={(e) => setTimeoutStrategy(e.target.value)}
            >
              <option value="prompt">超时策略：提醒并等待（25/50/75秒）</option>
              <option value="skip">超时策略：自动下一题</option>
              <option value="end">超时策略：结束考试</option>
            </select>
            <button
              style={{ width: '100%', padding: '12px', backgroundColor: (!originalQuestion.trim() || !originalAnswer.trim()) ? '#9ca3af' : '#4f46e5', color: 'white', border: 'none', borderRadius: '6px', fontSize: '16px', cursor: (!originalQuestion.trim() || !originalAnswer.trim()) ? 'not-allowed' : 'pointer' }}
              onClick={handleStartSubmit}
              disabled={!originalQuestion.trim() || !originalAnswer.trim()}
            >
              开始考试
            </button>
          </div>
        )}

        {phase === 'connecting' && (
          <div style={{ backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', padding: '24px', marginBottom: '16px', textAlign: 'center' }}>
            <div style={{ marginBottom: '16px' }}>⏳ 连接中...</div>
            <p>{currentHint}</p>
          </div>
        )}

        {phase === 'examining' && (
          <div style={{ backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', padding: '24px', marginBottom: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: connectionStatus === 'connected' ? '#10b981' : '#ef4444' }}></div>
                <span style={{ fontSize: '14px', color: '#6b7280' }}>
                  {connectionStatus === 'connected' ? '已连接' : '未连接'}
                </span>
              </div>
              <button onClick={endExam} style={{ color: '#dc2626', fontSize: '14px', border: 'none', background: 'none', cursor: 'pointer' }}>
                结束考试
              </button>
            </div>

            <div style={{ textAlign: 'center', padding: '32px 0' }}>
              <div style={{ width: '100px', height: '100px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px', fontSize: '40px', backgroundColor: isExaminerSpeaking ? '#eef2ff' : '#f3f4f6', animation: isExaminerSpeaking ? 'pulse 2s infinite' : 'none' }}>
                {isExaminerSpeaking ? '🗣️' : '🎙️'}
              </div>
              
              <p style={{ fontSize: '16px', color: '#374151', marginBottom: '8px' }}>{currentHint || '准备就绪'}</p>
              
              {/* 静默等待进度条 */}
              {isWaitingForAnswer && !isRecording && (
                <div style={{ marginTop: '12px', marginBottom: '12px' }}>
                  <div style={{ fontSize: '14px', color: timeoutLevel > 0 ? '#dc2626' : '#6b7280', marginBottom: '4px' }}>
                    等待回答中... {silenceDuration}秒 / 75秒
                    {timeoutLevel > 0 && ` (提醒 ${timeoutLevel}/3)`}
                  </div>
                  <div style={{ width: '100%', height: '6px', backgroundColor: '#e5e7eb', borderRadius: '3px', overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${silenceProgressPercent}%`, backgroundColor: timeoutLevel === 0 ? '#10b981' : timeoutLevel === 1 ? '#f59e0b' : timeoutLevel === 2 ? '#f97316' : '#dc2626', transition: 'all 0.3s ease' }}></div>
                  </div>
                </div>
              )}
              
              {/* 录音按钮 */}
              <div style={{ position: 'relative', display: 'inline-block' }}>
                <button
                  onMouseDown={startRecording}
                  onMouseUp={stopRecording}
                  onMouseLeave={isRecording ? stopRecording : undefined}
                  disabled={isProcessing || isExaminerSpeaking}
                  style={{ width: '80px', height: '80px', borderRadius: '50%', border: 'none', fontSize: '24px', cursor: (isProcessing || isExaminerSpeaking) ? 'not-allowed' : 'pointer', backgroundColor: isRecording ? '#ef4444' : (isExaminerSpeaking || isProcessing ? '#fbbf24' : '#4f46e5'), color: 'white', transform: isRecording ? 'scale(1.1)' : 'scale(1)', opacity: (isProcessing || isExaminerSpeaking) ? 0.6 : 1 }}
                >
                  {isRecording ? '⏹' : isExaminerSpeaking ? '⏸' : '🎤'}
                  
                  {/* 录音进度环 */}
                  {isRecording && (
                    <div style={{ position: 'absolute', top: '-4px', left: '-4px', right: '-4px', bottom: '-4px', borderRadius: '50%', border: '3px solid #e5e7eb', borderTopColor: '#4f46e5', transform: `rotate(${recordingProgressPercent * 3.6}deg)`, transition: 'transform 0.1s linear' }} />
                  )}
                </button>
              </div>

              {/* 录音进度条 */}
              {isRecording && (
                <div style={{ marginTop: '12px' }}>
                  <div style={{ fontSize: '12px', color: '#6b7280', marginBottom: '4px' }}>
                    录音中... {formatTime(recordingProgress)} / 0:30
                  </div>
                  <div style={{ width: '100%', height: '6px', backgroundColor: '#e5e7eb', borderRadius: '3px', overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${recordingProgressPercent}%`, backgroundColor: '#4f46e5', transition: 'width 0.1s linear' }}></div>
                  </div>
                </div>
              )}

              <p style={{ fontSize: '12px', color: '#9ca3af', marginTop: '8px' }}>
                {isRecording ? '按住录音中...' : '按住说话，松开发送'}
              </p>
            </div>

            {/* 对话记录 */}
            <div style={{ maxHeight: '200px', overflowY: 'auto', borderTop: '1px solid #e5e7eb', paddingTop: '16px' }}>
              <h3 style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '8px', color: '#374151' }}>对话记录</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {dialogueHistory.map((turn, idx) => (
                  <div key={idx} style={{ padding: '12px', borderRadius: '6px', fontSize: '14px', backgroundColor: turn.role === 'examiner' ? '#eef2ff' : turn.role === 'student' ? '#f0fdf4' : '#fef3c7', color: turn.role === 'examiner' ? '#312e81' : turn.role === 'student' ? '#166534' : '#92400e', marginLeft: turn.role === 'student' ? '32px' : '0' }}>
                    <div style={{ fontSize: '12px', opacity: 0.7, marginBottom: '4px' }}>
                      {turn.role === 'examiner' ? '考官' : turn.role === 'student' ? '你' : '系统'} • {new Date(turn.timestamp).toLocaleTimeString()}
                    </div>
                    {turn.text}
                  </div>
                ))}
                {dialogueHistory.length === 0 && (
                  <p style={{ color: '#9ca3af', fontSize: '14px', fontStyle: 'italic' }}>暂无记录</p>
                )}
              </div>
            </div>
          </div>
        )}

        {phase === 'grading' && (
          <div style={{ backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', padding: '24px', marginBottom: '16px', textAlign: 'center' }}>
            <div style={{ fontSize: '48px', marginBottom: '16px' }}>⏳</div>
            <p>正在评估...</p>
            <p style={{ fontSize: '14px', color: '#6b7280' }}>{currentHint}</p>
          </div>
        )}

        {phase === 'result' && examResult && (
          <div style={{ backgroundColor: 'white', borderRadius: '8px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', padding: '24px', marginBottom: '16px' }}>
            <h2 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '16px', color: '#111827' }}>评估结果</h2>
            
            {examResult.overall_assessment && (
              <div>
                <div style={{ padding: '16px', borderRadius: '6px', marginBottom: '16px', backgroundColor: examResult.overall_assessment.confidence > 0.7 ? '#f0fdf4' : examResult.overall_assessment.confidence > 0.4 ? '#fef3c7' : '#fef2f2', border: '1px solid ' + (examResult.overall_assessment.confidence > 0.7 ? '#86efac' : examResult.overall_assessment.confidence > 0.4 ? '#fcd34d' : '#fecaca') }}>
                  <h3 style={{ fontWeight: 'bold', marginBottom: '8px' }}>理解程度: {examResult.overall_assessment.understanding_level}</h3>
                  <p style={{ fontSize: '14px', color: '#6b7280', marginBottom: '8px' }}>置信度: {(examResult.overall_assessment.confidence * 100).toFixed(1)}%</p>
                  <p>{examResult.overall_assessment.reasoning}</p>
                </div>

                {examResult.overall_assessment.knowledge_gaps?.length > 0 && (
                  <div style={{ padding: '12px', borderRadius: '6px', marginBottom: '16px', backgroundColor: '#fff7ed' }}>
                    <h4 style={{ fontWeight: 'bold', marginBottom: '8px', color: '#c2410c' }}>知识漏洞</h4>
                    <ul style={{ paddingLeft: '20px', margin: 0, fontSize: '14px' }}>
                      {examResult.overall_assessment.knowledge_gaps.map((gap, i) => (
                        <li key={i} style={{ marginBottom: '4px' }}>{gap}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {examResult.overall_assessment.recommendations?.length > 0 && (
                  <div style={{ padding: '12px', borderRadius: '6px', marginBottom: '16px', backgroundColor: '#eff6ff' }}>
                    <h4 style={{ fontWeight: 'bold', marginBottom: '8px', color: '#1d4ed8' }}>学习建议</h4>
                    <ul style={{ paddingLeft: '20px', margin: 0, fontSize: '14px' }}>
                      {examResult.overall_assessment.recommendations.map((rec, i) => (
                        <li key={i} style={{ marginBottom: '4px' }}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            <button onClick={() => { setPhase('prepare'); setDialogueHistory([]); setExamResult(null); setEvalId(''); setCurrentHint(''); }} style={{ width: '100%', padding: '12px', backgroundColor: '#4f46e5', color: 'white', border: 'none', borderRadius: '6px', fontSize: '16px', cursor: 'pointer' }}>
              再次考试
            </button>
          </div>
        )}
      </div>
      
      <style>{`@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }`}</style>
    </div>
  );
}