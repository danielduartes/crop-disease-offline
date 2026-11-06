# Arquivo responsável por tirar os pesos do modelo treinado e gravar num formato que não depende da versão do Keras
import numpy as np
import tensorflow as tf

modelo = tf.keras.models.load_model("modelos/fase2.keras")
pesos = modelo.get_weights()
np.savez("modelos/pesos_fase2.npz", *pesos)
print(len(pesos), "arrays")
print(modelo.count_params(), "parâmetros")
print([tuple(p.shape) for p in pesos[:3]], "...", [tuple(p.shape) for p in pesos[-3:]])