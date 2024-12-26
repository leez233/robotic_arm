import pyaudio
import wave
import requests
from pydub import AudioSegment


def play_audio(file_path):
    # 打开 .wav 文件
    wf = wave.open(file_path, 'rb')

    # 创建 PyAudio 流
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # 读取并播放音频数据
    chunk = 1024
    data = wf.readframes(chunk)
    while data:
        stream.write(data)
        data = wf.readframes(chunk)

    # 关闭流和 PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()


def convert_mp3_to_wav(input_mp3_path, output_wav_path):
    # 加载 MP3 文件并转换为 WAV 格式
    audio = AudioSegment.from_mp3(input_mp3_path)

    # 保存 WAV 文件
    audio.export(output_wav_path, format="wav")
    print(f"MP3文件已转换为WAV格式，保存为 {output_wav_path}")


def convert_to_16k(input_wav_path, output_wav_path):
    # 加载音频文件
    audio = AudioSegment.from_wav(input_wav_path)

    # 设置采样率为16kHz
    audio = audio.set_frame_rate(16000)

    # 保存新的音频文件
    audio.export(output_wav_path, format="wav")


def stt(input_path):
    # 音频文件的本地路径
    url = 'http://10.24.4.39:5000/asr'
    convert_to_16k(input_path, "output.wav")

    # 打开音频文件并读取内容
    with open(input_path, 'rb') as audio_file:
        # 构造文件上传的表单数据
        files = {'file': (audio_file.name, audio_file, 'audio/wav')}

        # 发送 POST 请求到 Flask 服务端
        response = requests.post(url, files=files)

        # 检查请求是否成功
        if response.status_code == 200:
            # 解析返回的 JSON 数据
            data = response.json()
            # 打印 ASR 结果
            print("ASR Result:", data['result'])
            return data['result']
        else:
            # 打印错误信息
            print("Error:", response.json()['error'])


# 输入的是一段字符串
def tts(text, output_name, voice="zh-CN-XiaoyiNeural", speed=1.0):
    # 定义 TTS API URL
    url = 'http://10.24.4.39:7899/v1/audio/speech'

    # 构建请求的参数
    params = {
        'voice': voice,
        'input': text,
        'speed': speed
    }

    # 发送 POST 请求到 TTS API
    response = requests.post(url, json=params)

    # 检查请求是否成功
    if response.status_code == 200:
        # 获取响应的音频内容
        audio_content = response.content

        # 将音频文件保存到本地（MP3格式）
        mp3_path = f"{output_name}.mp3"
        with open(mp3_path, 'wb') as audio_file:
            audio_file.write(audio_content)
        print(f'语音文件已保存为 {mp3_path}')

        # 转换 MP3 为 WAV 并保存
        wav_path = f"{output_name}.wav"
        convert_mp3_to_wav(mp3_path, wav_path)

        return wav_path
    else:
        print('请求失败，状态码：', response.status_code)
        return None


# 测试调用
def test_tts_and_stt():
    # 1. 进行 TTS 测试
    text = "吃葡萄不吐葡萄皮，黄色方块向前移动5厘米"
    output_name = "output"  # 输出文件名（不带扩展名）

    # 调用 TTS 生成语音并保存为 WAV 文件
    wav_file = tts(text, output_name)

    if wav_file:
        print(f"TTS成功，音频已保存为 {wav_file}")

        # 播放生成的语音文件
        print("正在播放 TTS 语音...")
        play_audio(wav_file)

        # 2. 进行 STT 测试，使用生成的 WAV 文件
        result = stt(wav_file)

        if result:
            print(f"STT成功，识别的文本是：{result}")
        else:
            print("STT请求失败")
    else:
        print("TTS请求失败")


# 调用综合测试
#test_tts_and_stt()
