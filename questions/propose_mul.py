import json
import os
import re
from typing import List, Dict, Optional
from tqdm import tqdm
from question_proposer import get_questions, AgentResponse

def decode_escapes(text: Optional[str]) -> str:
    """
    安全解码字符串中的转义序列（如 \\n -> 换行符）
    
    处理方式：
    1. 先处理 \\n, \\t 等基本转义
    2. 避免 unicode_escape 过度解码（如 \\u0041 变成 A）
    """
    if not text or not isinstance(text, str):
        return text or ""
    
    # 先处理反斜杠转义，避免与 unicode 编码冲突
    # 使用临时占位符防止二次转义
    placeholder = "\x00ESC\x00"
    
    # 第一步：保护真正的 Unicode 转义（如 \u0041）
    # 将 \\u 替换为临时占位符
    text = text.replace('\\\\u', placeholder + 'u')
    
    # 第二步：解码基本转义字符
    escapes = {
        '\\n': '\n',
        '\\t': '\t',
        '\\r': '\r',
        '\\\\': '\\',
        '\\"': '"',
        "\\'": "'",
        '\\b': '\b',
        '\\f': '\f',
        '\\v': '\v'
    }
    
    for escaped, unescaped in escapes.items():
        text = text.replace(escaped, unescaped)

    text = text.replace(placeholder + 'u', '\\u')
    
    return text


def parse_generated_questions(text: str) -> List[str]:
    """
    解析AI返回的文本，提取编号的问题列表
    支持格式：1. xxx, 1、xxx, 1) xxx, - xxx 等
    """
    questions = []
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 使用正则去除行首的编号（数字+标点）
        cleaned = re.sub(r'^[\d\s\.\、\)\-\*\•]+\s*', '', line).strip()
        
        # 过滤条件：长度>10且不是空行/标题
        if cleaned and len(cleaned) > 10 and not cleaned.startswith('【'):
            questions.append(cleaned)
    
    return questions


def process_jsonl(
    input_path: str, 
    output_path: str, 
    api_key: str = None, 
    use_kimi: bool = True,
    resume: bool = False,
    decode_escapes_flag: bool = True,
    normalize_whitespace: bool = False
):
    """
    批量处理JSONL文件，为每对QA生成AI测试问题
    
    Args:
        input_path: 输入JSONL路径，格式: {"question": "...", "answer": "..."}
        output_path: 输出JSONL路径
        api_key: Moonshot API Key（默认从环境变量 MOONSHOT_API_KEY 读取）
        use_kimi: 是否调用真实Kimi API（False则使用模拟模式）
        resume: 是否断点续跑（跳过输出文件中已存在的记录）
        decode_escapes_flag: 是否解码转义字符（如 \\n -> 换行符）
        normalize_whitespace: 是否标准化空白（多个连续空白转为单个空格）
    """
    api_key = api_key or os.getenv("MOONSHOT_API_KEY")
    
    # 读取已处理记录（断点续跑）
    processed_ids = set()
    if resume and os.path.exists(output_path):
        print(f"🔄 检测到已存在的输出文件，启用断点续跑模式...")
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        # 使用原始 question 内容作为唯一标识（解码前）
                        original_q = item.get("original_question", "")
                        processed_ids.add(hash(original_q))
                    except:
                        pass
        print(f"📋 已跳过 {len(processed_ids)} 条已处理记录")
    
    # 读取输入
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line for line in f if line.strip()]
    
    total = len(lines)
    print(f"📖 输入文件: {input_path} ({total} 条)")
    print(f"💾 输出文件: {output_path}")
    print(f"🔧 转义解码: {'开启' if decode_escapes_flag else '关闭'}")
    print(f"🔧 空白标准化: {'开启' if normalize_whitespace else '关闭'}")

    # 追加模式打开输出文件
    mode = 'a' if resume else 'w'
    success_count = 0
    error_count = 0
    skip_count = 0
    
    with open(output_path, mode, encoding='utf-8') as out_f:
        
        for idx, line in enumerate(tqdm(lines, desc="处理进度"), 1):
            try:
                data = json.loads(line.strip())
                question = data.get("question", "") or ""
                answer = data.get("answer", "") or ""
                
                # 保存原始值用于断点续跑检查
                original_question = question
                
                # 解码转义字符（处理 \\n 等）
                if decode_escapes_flag:
                    question = decode_escapes(question)
                    answer = decode_escapes(answer)
                
                # 标准化空白（可选）
                if normalize_whitespace:
                    question = re.sub(r'\s+', ' ', question).strip()
                    answer = re.sub(r'\s+', ' ', answer).strip()
                
                # 断点续跑检查（使用原始文本的 hash）
                if resume and hash(original_question) in processed_ids:
                    skip_count += 1
                    continue
                
                # 调用考官模式生成问题
                response = get_questions(
                    question=question,
                    answer=answer,
                    api_key=api_key,
                    use_kimi=use_kimi
                )
                
                # 解析生成的问题列表
                ai_questions = parse_generated_questions(response.content)
                
                # 构建输出结构
                result = {
                    "original_question": original_question,  # 保存原始文本
                    "processed_question": question if decode_escapes_flag else None,  # 保存处理后的文本（如果处理了）
                    "original_answer": data.get("answer", ""),  # 原始 answer
                    "processed_answer": answer if decode_escapes_flag else None,
                    "ai_questions": ai_questions,           # 结构化问题列表
                    "ai_response_raw": response.content,    # 原始响应文本
                    "used_tools": response.used_tools,
                    "process_metadata": {
                        "history_length": response.metadata.get("history_length"),
                        "has_error": "错误" in response.content or "[Kimi API 错误]" in response.content,
                        "decode_escapes": decode_escapes_flag,
                        "normalize_whitespace": normalize_whitespace
                    }
                }

                result = {k: v for k, v in result.items() if v is not None}

                out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                out_f.flush()
                success_count += 1
                
            except Exception as e:
                error_count += 1
                # 记录错误但不中断流程
                error_result = {
                    "original_data": line.strip(),
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "line_number": idx,
                    "ai_questions": []
                }
                out_f.write(json.dumps(error_result, ensure_ascii=False) + '\n')
                out_f.flush()
                if error_count <= 5:  # 只显示前5个错误详情
                    print(f"\n❌ 第{idx}条处理失败: {str(e)[:100]}")
    
    print(f"\n{'='*50}")
    print(f"✅ 处理完成！")
    print(f"   成功: {success_count} 条")
    print(f"   失败: {error_count} 条")
    if resume:
        print(f"   跳过: {skip_count} 条（已存在）")
    print(f"💾 结果已保存至: {output_path}")


# ==================== 使用示例 ====================

if __name__ == "__main__":
    
    # 路径配置
    INPUT_FILE = "/home/gsk/thesis_2026-gsk/questions/q&a/multiplier.jsonl"
    OUTPUT_FILE = "/home/gsk/thesis_2026-gsk/questions/results/multiplier_ai_questions.jsonl"

    API_KEY = os.getenv("MOONSHOT_API_KEY")
    
    USE_KIMI = True
    
    # 执行处理
    process_jsonl(
        input_path=INPUT_FILE,
        output_path=OUTPUT_FILE,
        api_key=API_KEY,
        use_kimi=USE_KIMI,
        resume=True,             
        decode_escapes_flag=True,
        normalize_whitespace=False 
    )