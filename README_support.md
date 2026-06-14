# 支持文档
2026.06
## 环境配置
```
pip install requirements.txt
```
## 大模型配置
### 使用kimi
```
export MOONSHOT_API_KEY="your_api_key"
```
### 使用其他大模型厂商api
除了配置apikey，注意修改：
/llm-council/backend/config.py
/llm-council/backend/kimi.py
/questions/os_proposer.py
/questions/question_proposer.py
/test/exp2_scoring_discrimination.py
## ASR配置
```
export TENCENT_SECRET_ID="your_secret_id"
export TENCENT_SECRET_KEY="your_secret_key"
export TENCENT_ASR_REGION="your_region"
export COS_BUCKET="your_cos_bucket"
export COS_REGION="your_region"
```
## TTS配置
```
export WHISPER_MODEL_SIZE=small
```

