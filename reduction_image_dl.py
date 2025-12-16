import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # Conv2d: convoluzione 2D per estrarre feature map
        # in_ch = numero di canali in ingresso (es. 3 per RGB o output layer precedente)
        # out_ch = numero di feature map in uscita (quante "feature" vogliamo estrarre)
        # kernel_size=3 → finestra 3x3 standard
        # stride → passo della convoluzione (stride>1 riduce la dimensione spaziale)
        # padding=1 → mantiene dimensioni spaziali se stride=1
        # bias=False → inutile quando si usa BatchNorm
        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        # Batch Normalization → stabilizza e velocizza il training
        self.bn = nn.BatchNorm2d(out_ch)
        # LeakyReLU → attivazione non lineare per introdurre capacità espressiva
        # slope negativo 0.1 per evitare neuroni morti
        self.act = nn.LeakyReLU(0.1, inplace=True)

    # Forward pass del blocco
    def forward(self, x):
        # Ordine: convoluzione → batch norm → attivazione
        return self.act(self.bn(self.conv(x)))

# AUTOENCODER PER IMMAGINI 300x300
class ImageAutoencoder300(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        # ================= ENCODER =================
        # Encoder CNN → estrazione feature dalle immagini
        self.encoder_cnn = nn.Sequential(
            ConvBlock(3, 32, stride=2),    # Downsampling 300x300 → 150x150
            ConvBlock(32, 64, stride=2),   # 150x150 → 75x75
            ConvBlock(64, 128, stride=2),  # 75x75 → 38x38
            ConvBlock(128, 256, stride=2), # 38x38 → 19x19
            nn.Conv2d(256, 512, 3, padding=1), # aumento feature map a 512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Global Average Pooling → riduce dimensione spaziale (19x19 → 1x1) mantenendo 512 feature
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # Linear layer → riduce feature a latent_dim
        # Questo è il "bottleneck" o spazio latente
        self.fc_latent = nn.Linear(512, latent_dim)

        # ================= DECODER =================
        # Linear layer → espande latent_dim a feature map compatibile con decoder
        self.fc_decode = nn.Linear(latent_dim, 19 * 19 * 256)

        # Decoder CNN → ricostruzione immagine
        self.decoder_cnn = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),  # 19x19 → 38x38
            ConvBlock(256, 256),

            nn.Upsample(scale_factor=2, mode='nearest'),  # 38x38 → 76x76
            ConvBlock(256, 128),

            nn.Upsample(scale_factor=2, mode='nearest'),  # 76x76 → 152x152
            ConvBlock(128, 64),

            nn.Upsample(scale_factor=2, mode='nearest'),  # 152x152 → 304x304
            ConvBlock(64, 32),

            nn.Conv2d(32, 3, kernel_size=3, padding=1),  # output 3 canali (RGB)
            nn.Sigmoid()  # porta valori tra 0 e 1
        )

    # ================= ENCODE =================
    def encode(self, x):
        # Passa l'immagine attraverso CNN encoder
        x = self.encoder_cnn(x)
        # Applica Global Average Pooling → riduce dimensioni spaziali
        x = self.gap(x)
        # Appiattisce feature map → vettore (batch, 512)
        x = torch.flatten(x, 1)
        # Linear layer → vettore latente (batch, latent_dim)
        return self.fc_latent(x)

    # ================= DECODE =================
    def decode(self, z):
        # Espande vettore latente a feature map compatibile con decoder
        x = self.fc_decode(z)
        x = x.view(-1, 256, 19, 19)  # reshape in batch_size x canali x altezza x larghezza
        # Passa attraverso il decoder CNN → ricostruzione immagine
        return self.decoder_cnn(x)

    # ================= FORWARD =================
    def forward(self, x):
        # Encode → vettore latente
        z = self.encode(x)
        # Decode → immagine ricostruita
        return self.decode(z)
