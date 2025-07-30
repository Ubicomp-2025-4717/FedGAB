import librosa
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
# 128*94*1
def wav_pkl(file_path):
    # file_path = "D:\\Audio-Visual-Vehicle-Dataset\\Highway\\Bus\\Audio\\audio_0001_8_4.wav"
    y, sr = librosa.load(file_path, sr=16000)
    # 计算梅尔频谱图
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    # 将梅尔频谱图转换为对数尺度
    log_S = librosa.power_to_db(S, ref=np.max)
    # 标准化梅尔频谱图
    log_S -= np.mean(log_S)
    log_S /= np.std(log_S)
    log_S = np.expand_dims(log_S, axis=-1)  # 添加通道维度
    print(log_S.shape)
    save_path=file_path.replace(".wav",".pkl")
    torch.save(log_S,save_path)

# wav_pkl("D:\\Audio-Visual-Vehicle-Dataset\\Highway\\Bus\\Audio\\audio_0001_8_4.wav")

file_path="D:\\Audio-Visual-Vehicle-Dataset\\Highway_a"
class_name=os.listdir(file_path)
for i in class_name:
    file=file_path + "\\" + i + "\\" + "Audio"
    audio_name=os.listdir(file)
    for j in audio_name:
        wav_pkl(file+"\\"+j)


