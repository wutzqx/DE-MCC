import os
import cv2
import numpy as np
from EEG.EEG_DAE import DenoiseAutoEncoder
from torch import nn

from sklearn.model_selection import train_test_split

import torch

import torch.utils.data as Data
import torch.optim as optim


global_np_data_ph = 'EEG.npz'
batch_size = 32
LR = 0.0003
epoch_num = 100

#图像处理器
def dataDeal(data_path):
    img_list = []
    files = os.listdir(data_path)
    for imn in files:
        img = cv2.imread(os.path.join(data_path, imn))
        img = cv2.resize(img, (96, 96))
        img_list.append(img[:, :, 0])
    image_array = np.array(img_list)
    data1 = np.reshape(image_array, [-1, 1, 96, 96])
    data1 = data1/255
    # data1 = np.transpose(data1, [0, 3, 2, 1])
    np.savez(global_np_data_ph,
             img=data1)

# 数据加载器
def dataTensor():

    data1 = np.load(global_np_data_ph)['view_4']

    X_train, X_val = train_test_split(data1, test_size=0.2, random_state=123)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    train_loader = Data.DataLoader(
        dataset=X_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = Data.DataLoader(
        dataset=X_val,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    return train_loader, val_loader




if __name__== '__main__':
    # ph = "E:\\project\\cluster\\DeepDPM-main\\DeepDPM-main\\data\\train\\0"
    train_loader, val_loader = dataTensor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    DAEmodel = DenoiseAutoEncoder(int(200), int(50))

    DAEmodel.to(device)

    optimizer = optim.Adam(DAEmodel.parameters(), lr=LR)
    loss_func = nn.MSELoss()

    #history = hl.History()
    #canvas = hl.Canvas()

    train_num, val_num = 0, 0

    for epoch in range(epoch_num):
        train_loss_epoch, val_loss_epoch = 0, 0

        # 训练
        for step, b_x in enumerate(train_loader):

            b_x = b_x.to(device)

            DAEmodel.train()

            _, output = DAEmodel(b_x)
            loss = loss_func(output, b_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item() * b_x.size(0)
            train_num += b_x.size(0)

        # 验证
        for step, b_x in enumerate(val_loader):
            DAEmodel.eval()
            b_x = b_x.to(device)
            _, output = DAEmodel(b_x)
            loss = loss_func(output, b_x)
            val_loss_epoch += loss.item() * b_x.size(0)
            val_num += b_x.size(0)

        train_loss = train_loss_epoch / train_num
        val_loss = val_loss_epoch / val_num
        print(f'epoch {epoch}: {val_loss}')


        torch.save(DAEmodel.state_dict(), "./DAEmodel_view4.pkl")





