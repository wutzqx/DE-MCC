
from torch import nn

class DenoiseAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DenoiseAutoEncoder, self).__init__()
        # Encoder
        self.Encoder = nn.Sequential(
            # param [input_c, output_c, kernel_size, stride, padding]
            nn.Conv1d(12, 16, 3, padding=1),  # [, 64, 96, 96]
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 8, 3, padding=1),  # [, 64, 96, 96]
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Conv1d(8, 8, 3, 1, 1),  # [, 64, 48, 48]
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Conv1d(8, 8, 3, 1, 1),  # [, 64, 48, 48]
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Conv1d(8, 4, 3, 1, 1),  # [, 128, 48, 48]
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(4),
            nn.Conv1d(4, 1, 3, 1, 1),  # [, 128, 48, 48]
            nn.ReLU(),
            nn.BatchNorm1d(1),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.BatchNorm1d(1)

        )

        # decoder
        self.Decoder = nn.Sequential(
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.BatchNorm1d(1),
            nn.ConvTranspose1d(1, 4, 3, 1, 1),  # [, 128, 24, 24]
            nn.ConvTranspose1d(4, 4, 3, 2, 1, 1),
            nn.ReLU(),
            nn.BatchNorm1d(4),
            nn.ConvTranspose1d(4, 8, 3, 1, 1),  # [, 128, 48, 48]
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.ConvTranspose1d(8, 8, 3, 1, 1),  # [, 64, 48, 48]
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.ConvTranspose1d(8, 8, 3, 1, 1),  # [, 64, 48, 48]
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.ConvTranspose1d(8, 16, 3, 1, 1),  # [, 32, 48, 48]
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.ConvTranspose1d(16, 12, 3, 1, 1),  # [, 32, 48, 48]
            nn.Sigmoid()
        )

    def forward(self, x):
        encoder = self.Encoder(x)
        decoder = self.Decoder(encoder)
        return encoder, decoder