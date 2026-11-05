import sys, types

class Falso(types.ModuleType):
    def __getattr__(self, nome):
        if nome.startswith("__"):
            raise AttributeError(nome)
        return Falso(f"{self.__name__}.{nome}")

for nome in ("tensorflow_decision_forests", "jax", "jax.experimental", "jax.experimental.jax2tf"):
    sys.modules[nome] = Falso(nome)

import tensorflowjs as tfjs
print("ok", tfjs.__version__)