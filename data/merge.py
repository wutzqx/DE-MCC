import numpy as np

view0 = np.load('view_0.npy')
view1 = np.load('view_1.npy')
view2 = np.load('view_2.npy')
view3 = np.load('view_3.npy')
view4 = np.load('view_4.npy')
view5 = np.load('view_5.npy')
labels = np.load('labels.npy')
np.savez('EEG.npz',
         labels=labels,
         n_views=np.array([6]),
         view_0=view0,
         view_1=view1,
         view_2=view2,
         view_3=view3,
         view_4=view4,
         view_5=view5
         )