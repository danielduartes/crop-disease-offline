import sys, types

class Falso(types.ModuleType):
    def __getattr__(self, nome):
        if nome.startswith("__"):
            raise AttributeError(nome)
        return Falso(f"{self.__name__}.{nome}")

for nome in ("tensorflow_decision_forests", "jax", "jax.experimental", "jax.experimental.jax2tf"):
    sys.modules[nome] = Falso(nome)

import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs

base = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights=None)
modelo = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(15, activation="softmax"),
])

dados = np.load("modelos/pesos_fase2.npz")
pesos = [dados[f"arr_{i}"] for i in range(len(dados.files))]
esperados = modelo.get_weights()
print(len(pesos), "arrays lidos |", len(esperados), "esperados")

formas_ok = all(p.shape == e.shape for p, e in zip(pesos, esperados))
print("formas iguais:", formas_ok)
if len(pesos) != len(esperados) or not formas_ok:
    raise SystemExit("os pesos não casam com a arquitetura; não converter")

modelo.set_weights(pesos)

x = np.random.default_rng(0).random((1, 160, 160, 3)).astype("float32")
print("saida pra conferir:", modelo.predict(x, verbose=0)[0][:5])

tfjs.converters.save_keras_model(modelo, "web/modelo")
print("convertido")