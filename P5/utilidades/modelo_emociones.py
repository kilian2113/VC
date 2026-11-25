from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# Red densa (MLP): capas ocultas con BatchNorm, ReLU y Dropout, y una salida por emoción
class ClasificadorEmociones(nn.Module):
    def __init__(self, n_entrada, n_clases=7, ocultas=(512, 256, 128), dropout=0.4):
        super().__init__()
        capas, n = [], n_entrada
        for h in ocultas:
            capas += [nn.Linear(n, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            n = h
        capas.append(nn.Linear(n, n_clases))
        self.red = nn.Sequential(*capas)

    def forward(self, x):
        return self.red(x)


# Carga el modelo entrenado y predice emociones a partir de landmarks normalizados.
# Para "delta" y "ambos" hay que calibrar antes la cara neutra del jugador.
class PredictorEmociones:
    def __init__(self, ruta_red, dispositivo="cpu"):
        ruta_red = Path(ruta_red)
        if not ruta_red.exists():
            raise FileNotFoundError(f"No encuentro el modelo {ruta_red.resolve()}")
        punto = torch.load(ruta_red, map_location=dispositivo)

        arq = punto["arquitectura"]
        self.red = ClasificadorEmociones(arq["n_entrada"], arq["n_clases"],
                                         tuple(arq["ocultas"]), arq["dropout"])
        self.red.load_state_dict(punto["estado_modelo"])
        self.red.to(dispositivo).eval()

        self.dispositivo = dispositivo
        self.emociones = list(punto["emociones"])
        self.tipo = punto["tipo_caracteristicas"]
        self.media = punto["escalador_media"].cpu().numpy()
        self.escala = punto["escalador_escala"].cpu().numpy()
        self.neutra = None   # cara neutra del jugador

    @property
    def necesita_calibracion(self):
        return self.tipo in ("delta", "ambos")

    # Cara neutra = media de varios fotogramas
    def calibrar(self, lista_landmarks):
        if len(lista_landmarks) == 0:
            raise ValueError("No hay fotogramas para calibrar")
        self.neutra = np.mean(np.stack(lista_landmarks), axis=0).astype(np.float32)

    def olvidar_calibracion(self):
        self.neutra = None

    # Vector de entrada, construido igual que construir_X() en el entrenamiento
    def caracteristicas(self, lm):
        lm = np.asarray(lm, dtype=np.float32)
        absoluto = lm.reshape(-1)
        if self.tipo == "absoluto":
            return absoluto
        if self.neutra is None:
            raise RuntimeError("Hay que calibrar la cara neutra antes de predecir")
        delta = (lm - self.neutra).reshape(-1)
        if self.tipo == "delta":
            return delta
        return np.concatenate([absoluto, delta])

    # Probabilidad de cada emoción (suman 1)
    def predecir(self, lm):
        x = (self.caracteristicas(lm) - self.media) / self.escala
        with torch.no_grad():
            t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(self.dispositivo)
            return torch.softmax(self.red(t), dim=1)[0].cpu().numpy()