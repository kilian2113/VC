from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class ClasificadorEmociones(nn.Module):
    def __init__(self, n_entrada, n_clases=7, ocultas=(512, 256, 128), dropout=0.4):
        super().__init__()
        capas, n = [], n_entrada
        for h in ocultas:
            capas += [nn.Linear(n, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            n = h
        capas.append(nn.Linear(n, n_clases))   # salida: una puntuación por emoción
        self.red = nn.Sequential(*capas)

    def forward(self, x):
        return self.red(x)


class PredictorEmociones:
    """Carga un modelo guardado en el entrenamiento y predice emociones.

    Uso:
        pred = PredictorEmociones("modelo_emociones_ambos.pt")
        pred.calibrar(lista_de_landmarks_neutros)   # si el modelo usa "delta" o "ambos"
        probs = pred.predecir(landmarks_normalizados)  # array (7,) que suma 1
    """

    def __init__(self, ruta_red, dispositivo="cpu", tipo=None):
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
        # Tipo de características: el guardado en el modelo, o el indicado a mano
        if tipo is not None:
            self.tipo = tipo
        elif "tipo_caracteristicas" in punto:
            self.tipo = punto["tipo_caracteristicas"]
        else:
            raise KeyError(
                "El modelo no guarda 'tipo_caracteristicas'. Añádelo en el torch.save del "
                "entrenamiento y reentrena, o indícalo a mano: PredictorEmociones(ruta, tipo='ambos')"
            )
        self.media = punto["escalador_media"].cpu().numpy().astype(np.float32)
        self.escala = punto["escalador_escala"].cpu().numpy().astype(np.float32)
        self.neutra = None   # landmarks neutros de la persona (calibración)

    # ------------------------------------------------------------ calibración
    @property
    def necesita_calibracion(self):
        return self.tipo in ("delta", "ambos")

    @property
    def calibrado(self):
        return (not self.necesita_calibracion) or self.neutra is not None

    def calibrar(self, lista_landmarks):
        """Guarda la cara neutra como la media de varios fotogramas (N, 3) normalizados."""
        if len(lista_landmarks) == 0:
            raise ValueError("No hay fotogramas para calibrar")
        self.neutra = np.mean(np.stack(lista_landmarks), axis=0).astype(np.float32)

    def olvidar_calibracion(self):
        self.neutra = None

    # ------------------------------------------------------------ predicción
    def caracteristicas(self, lm):
        """Construye el vector de entrada igual que construir_X() en el entrenamiento."""
        lm = np.asarray(lm, dtype=np.float32)
        absoluto = lm.reshape(-1)
        if self.tipo == "absoluto":
            return absoluto
        if self.neutra is None:
            raise RuntimeError("Este modelo necesita calibrar la cara neutra antes de predecir")
        delta = (lm - self.neutra).reshape(-1)
        if self.tipo == "delta":
            return delta
        if self.tipo == "ambos":
            return np.concatenate([absoluto, delta])
        raise ValueError(f"Tipo de características desconocido: {self.tipo}")

    def predecir(self, lm):
        """Probabilidades (array de len(emociones)) para unos landmarks normalizados (N, 3)."""
        x = (self.caracteristicas(lm) - self.media) / self.escala
        with torch.no_grad():
            t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(self.dispositivo)
            probs = torch.softmax(self.red(t), dim=1)[0].cpu().numpy()
        return probs