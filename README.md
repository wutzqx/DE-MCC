## DE-MCC

```consol
# requeirments
python == 3.7
scikit-learn
numpy
scipy
torch == 1.13.1+cu116
```

For publicly available EEG datasets, we provide processed EEG data and store them in data/EEG.npz.

```
# To start training a new model on provided datasets, e.g., Digit, run:
python data/merge.py
python train_DE_MCC.py --dataset EEG
```

If you want to train a new AE with your own source dataset, please dealing data with **EEG\\EEGdel.py** then, using **EEG\\EegAeTrain.py** and **EEG\\run.py** for dimensionality reduction.

