import mne
import scipy.io as sio
import os
import numpy as np
from sklearn.preprocessing import normalize
# 超参数
# 通道名顺序

ch_names = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1',
            'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1',
            'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
            'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1',
            'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1',
            'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
            'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
select_names = ['FT7', 'FT8', 'T7', 'C5', 'C6', 'T8', 'TP7', 'CP5', 'CP6', 'TP8', 'P7', 'P8']
select_channels = [14, 22, 23, 24, 30, 31, 32, 33, 39, 40, 43, 49]
# 采样频率
sfreq = 200
per_unit_num = 150
# 每个.mat文件中的数据label
basic_label = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]


def read_one_file(file_path):
    """
    input:单个.mat文件路径
    output:raw格式数据
    """
    data = sio.loadmat(file_path)
    # 获取keys并转化为list，获取数据所在key
    keys = list(data.keys())[3:]
    # print(keys)
    # 获取数据
    raw_list = []
    for i in range(len(keys)):
        # 获取数据
        tmp = data[keys[i]]
        stamp = []
        for _ in select_channels:
            stamp.append(tmp[_])
        # print(stamp.shape)
        # 创建info
        info = mne.create_info(ch_names=select_names, sfreq=sfreq, ch_types='eeg')
        # 创建raw，取第5秒开始的数据
        raw = mne.io.RawArray(stamp, info).crop(tmin=10, tmax=170)
        # 添加到raw_list
        raw_list.append(raw)
    return raw_list

def get_fea(file_path):
    fea_data = sio.loadmat(file_path)
    n = 15
    pick_fea = "de_LDS"

    fealist = []

    for i in range(1, n+1):
        stamp = []
        for j in select_channels:
            tmp = fea_data[pick_fea+str(i)][j, 10:170, :]
            stamp.append(tmp)
        stamp = np.concatenate(stamp, axis=1)
        fealist.extend(stamp)
    fealist = np.array(fealist)

    return fealist

def read_all_files(pre_path, fea_path, max_files_num=1, select = 4):
    # 读取文件夹下所有.mat文件

    print("read_all_files start...")
    label = sio.loadmat(os.path.join(pre_path, "label.mat"))
    label = label['label'].tolist()[0]
    labelArray = []
    for i in label:
        labelArray.extend([i + 1 for _ in range(per_unit_num)])
    labelArray = np.array(labelArray)
    # 遍历Preprocessed_EEG文件夹下所有.mat文件
    data_list = []
    # 读取文件数量（每个文件中有15段数据）
    files_num = 0
    files = os.listdir(pre_path)
    root1 = pre_path
    root2 = fea_path
    for file in files:
        if os.path.splitext(file)[1] == '.mat':
            file_path1 = os.path.join(root1, file)
            raw_list1 = read_one_file(file_path1)
            file_path2 = os.path.join(root2, file)
            fea = get_fea(file_path2)

            # 将raw_list中的每一个元素添加到data_list
            # data_list.extend(raw_list)
            files_num += 1
            out = frequency_spectrum(raw_list1)
            out.append(normalize(fea, axis=0))
            for _ in range(len(out)-1):
                sp = out[_].shape
                tmp = out[_].reshape(sp[0], -1)
                tmp = normalize(tmp, axis=0)
                out[_] = tmp.reshape(sp[0], sp[1], sp[2])
            #np.save('./features/'+file, out)
            if files_num == max_files_num:
                break
    np.savez('EEG_4.npz',
             labels=labelArray,
             n_views=np.array([6]),
             view_0=out[0],
             view_1=out[1],
             view_2=out[2],
             view_3=out[3],
             view_4=out[4],
             view_5=out[5])
    print("共读取了{}个文件".format(files_num))
    print("共有{}段数据".format(len(data_list)))
    print("read ended...")
    # return data_list, label_list
    return 0

def pre_denoise(raws):

    return raws

def frequency_spectrum(raws, select = 4):
    # 提取EEG数据在五个频段的能量特征
    # delta(0.5-4Hz) theta(4-8Hz) alpha(8-13Hz) beta(13-30Hz) gamma(30-100Hz)
    # 特定频带
    data_dic = {"delta": [],
                  "theta": [],
                  "alpha": [],
                  "sigma": [],
                  "beta": []}
    FREQ_BANDS = {"delta": [0.5, 4.5],
                  "theta": [4.5, 8.5],
                  "alpha": [8.5, 13.5],
                  "sigma": [13.5, 32.5],
                  "beta": [32.5, 50]}
    # 特征矩阵
    feature_matrix = []
    # 遍历每个raw
    for raw in raws:
        # 生成频谱特征向量
        feature_vector = []
        # 遍历每个频段
        for band in FREQ_BANDS:
            # 提取每个频段的数据，不打印信息
            raw_band = raw.copy().filter(l_freq=FREQ_BANDS[band][0], h_freq=FREQ_BANDS[band][1], verbose=False)

            np_band, _ = raw_band[:, :]
            np_band = np.delete(np_band, -1, axis=1)
            np_band = np_band.reshape(len(select_channels),
                         sfreq * select, -1)# 按照每秒分片

            np_band = np_band.transpose(2, 0, 1)
            data_dic[band].extend(np_band.tolist())
    outlist = []
    for _ in data_dic.keys():
        tmp = np.array(data_dic[_])
        outlist.append(tmp)



    #         # 计算能量
    #         power = np.sum(raw_band.get_data() ** 2, axis=1) / raw_band.n_times
    #         # 添加到特征向量
    #         feature_vector.extend(power)
    #     # 添加到特征矩阵
    #     feature_matrix.append(feature_vector)
    # # 返回特征矩阵
    # print("频谱特征矩阵的shape为：{}".format(np.array(feature_matrix).shape))
    # # print("频谱特征矩阵内容为：{}".format(np.array(feature_matrix)))
    return outlist


#get_fea("ExtractedFeatures_1s/1_20131027")
read_all_files("Preprocessed_EEG/", "ExtractedFeatures_4s/", 2)
