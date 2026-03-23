from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import json
import subprocess
from pathlib import Path
import yaml

app = Flask(__name__)
CORS(app)

# 从配置文件加载路径（使其通用）
CONFIG_PATH = Path("config.yaml")

def load_config():
    """加载或创建默认配置"""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    # 默认配置
    return {
        "workspace": "/home/gsk/thesis_2026-gsk/questions",
        "input_dir": "q&a",
        "output_dir": "results",
        "script_name": "propose_mul.py",
        "output_filename": "multiplier_ai_questions.jsonl"
    }

CONFIG = load_config()
BASE_DIR = Path(CONFIG['workspace'])
QNA_DIR = BASE_DIR / CONFIG['input_dir']
RESULTS_DIR = BASE_DIR / CONFIG['output_dir']
SCRIPT_PATH = BASE_DIR / CONFIG['script_name']
OUTPUT_FILE = RESULTS_DIR / CONFIG['output_filename']

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/config')
def get_config():
    """获取当前系统配置（供前端动态显示路径）"""
    return jsonify({
        "input_path": str(QNA_DIR),
        "script_path": str(SCRIPT_PATH),
        "output_path": str(OUTPUT_FILE),
        "script_name": SCRIPT_PATH.name
    })

@app.route('/api/list-files')
def list_files():
    """获取输入目录下的所有 jsonl 文件"""
    try:
        files = []
        if QNA_DIR.exists():
            for f in QNA_DIR.glob("*.jsonl"):
                stat = f.stat()
                files.append({
                    "name": f.name,
                    "path": str(f),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "records": count_jsonl_lines(f)  # 统计行数
                })
        return jsonify({"success": True, "files": files})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def count_jsonl_lines(filepath):
    """统计 JSONL 文件行数"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
    except:
        return 0

@app.route('/api/execute', methods=['POST'])
def execute_script():
    """执行处理脚本"""
    try:
        data = request.get_json()
        selected_files = data.get('files', [])
        
        if not selected_files:
            return jsonify({"success": False, "error": "未选择文件"}), 400
        
        # 构建环境变量
        env = os.environ.copy()
        env['INPUT_FILES'] = ','.join(selected_files)
        env['OUTPUT_DIR'] = str(RESULTS_DIR)
        
        # 执行脚本
        result = subprocess.run(
            ['python', str(SCRIPT_PATH)],
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
            env=env,
            timeout=300  # 5分钟超时
        )
        
        return jsonify({
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "executed_at": result.stderr if result.returncode != 0 else "执行成功"
        })
    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "error": "执行超时（超过5分钟）"}), 504
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/read-results')
def read_results():
    """读取生成的 JSONL 结果文件"""
    try:
        if not OUTPUT_FILE.exists():
            return jsonify({
                "success": False,
                "error": "结果文件尚未生成",
                "data": []
            }), 404
        
        data = []
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    obj['_meta'] = {'line': line_num}  # 添加元数据
                    data.append(obj)
                except json.JSONDecodeError as e:
                    data.append({
                        '_meta': {'line': line_num, 'error': True},
                        'parse_error': str(e),
                        'raw': line[:200]
                    })
        
        return jsonify({
            "success": True,
            "file_path": str(OUTPUT_FILE),
            "count": len(data),
            "data": data,
            "generated_at": os.path.getmtime(OUTPUT_FILE)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/check-status')
def check_status():
    """检查系统状态"""
    return jsonify({
        "workspace_exists": BASE_DIR.exists(),
        "input_dir_exists": QNA_DIR.exists(),
        "output_dir_exists": RESULTS_DIR.exists(),
        "script_exists": SCRIPT_PATH.exists(),
        "output_ready": OUTPUT_FILE.exists(),
        "config": CONFIG
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')