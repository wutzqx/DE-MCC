import os

import cv2
import torch.utils.data as Data
import numpy as np
import torch
from EEG_DAE import DenoiseAutoEncoder

ph = "E:\\project\\cluster\\DeepDPM-main\\DeepDPM-main\\data\\train\\0"
data_path = "../data/EEG.npz"

DAEmodel = DenoiseAutoEncoder(200, 50)


data_all = np.load(data_path)
view_list = []
for i in range(5):
    data = data_all['view_'+str(i)]
    DAEmodel.load_state_dict(torch.load("DAEmodel_view"+str(i)+".pkl"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    DAEmodel.to(device)
    data = torch.tensor(data, dtype=torch.float32)


    dataLoader = Data.DataLoader(
        dataset=data,
        batch_size=16,
        shuffle=True,
        num_workers=0)
    DAEmodel.eval()

    featurelist = []
    for step, b_x in enumerate(dataLoader):
        b_x = b_x.to(device)
        with torch.no_grad():
            features, output = DAEmodel(b_x)
            features = features.cpu()
            features = features.detach().numpy()
            for _ in features:
                featurelist.append(_.flatten())
        # if step % 20 == 0:
        #     fig1 = output[0, :, :].cpu().detach().numpy()
        #     fig2 = b_x[0, :, :].cpu().detach().numpy()
        #     imgs = np.hstack([fig1, fig2])
        #     cv2.imshow('1', imgs)

            cv2.waitKey()
    view_list.append(np.array(featurelist))

np.savez('../data/EEG.npz',
         labels=data_all['labels'],
         n_views=np.array([6]),
         view_0=view_list[0],
         view_1=view_list[1],
         view_2=view_list[2],
         view_3=view_list[3],
         view_4=view_list[4],
         view_5=data_all['view_5']
         )



