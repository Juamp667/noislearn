'''
    Module with tools to create noise data.
'''

from numpy.random import randint, seed
import numpy as np

def urlf(y, noise_level=0.1, random_state=42):
    """
    UniformRandomizedLabelFlip (corregida)

    - No modifica y original (trabaja sobre copia).
    - Cambia un % de etiquetas a otra etiqueta (uniforme).
    - Funciona con etiquetas arbitrarias (strings, ints, etc.)
    """
    y = np.asarray(y)
    y_out = y.copy()

    seed(random_state)

    classes, y_idx = np.unique(y_out, return_inverse=True)
    n = len(y_out)
    k = int(noise_level * n)
    if k <= 0:
        return y_out

    idx_to_change = randint(low=0, high=n, size=k)
    # nuevos índices de clase (0..K-1)
    new_idx = randint(low=0, high=len(classes), size=len(idx_to_change))

    # opcional: asegurar "flip" real (que no caiga la misma clase)
    # (si no te importa que a veces se quede igual, comenta estas 4 líneas)
    same = new_idx == y_idx[idx_to_change]
    if np.any(same) and len(classes) > 1:
        new_idx[same] = (new_idx[same] + 1) % len(classes)

    y_out[idx_to_change] = classes[new_idx]
    return y_out