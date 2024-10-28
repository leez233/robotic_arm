# asr_client.py

import subprocess
import re

def run_command(command):
    """执行命令并返回输出"""
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("Error running command:")
        print(e.stderr)
        raise KeyboardInterrupt  # 若需要退出程序

def extract_audio_text(output):
    """提取识别结果中的文字部分"""
    match = re.search(r"demo: (.*?) timestamp:", output)
    if match:
        return match.group(1).strip()
    return "识别结果未找到"

def get_asr_text():
    # 定义命令参数
    command = [
        "python", "funasr_wss_client.py",
        "--host", "10.24.4.39",
        "--port", "10085",
        "--mode", "offline",
        "--audio_in", "output.wav",
        "--ssl", "1"
    ]

    # 执行命令并提取识别文字
    output = run_command(command)
    audio_text = extract_audio_text(output)
    return audio_text
