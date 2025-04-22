import numpy as np

data = np.load('EEG.npz')
view0 = data['view_0']
np.save('view_0.npy', view0)
view1 = data['view_1']
np.save('view_1.npy', view0)
view2 = data['view_2']
np.save('view_2.npy', view1)
view3 = data['view_3']
np.save('view_3.npy', view0)
view4 = data['view_4']
np.save('view_4.npy', view1)
view5 = data['view_5']
np.save('view_5.npy', view2)
label = data['labels']
np.save('labels.npy', label)


