r"""
This module defines the gate sets of the various hardware models.

.. dropdown:: IBM gate set
    :color: primary

    Entangling gate: CNOT

    .. math::

        \begin{bmatrix}
        1 & 0 & 0 & 0\\
        0 & 1 & 0 & 0\\
        0 & 0 & 0 & 1\\
        0 & 0 & 1 & 0
        \end{bmatrix}

    Single qubit gate: :math:`U_3(\theta,\phi,\lambda)`

    .. math::

        \begin{bmatrix}
        \cos(\theta/2) & -e^{i\lambda}\sin(\theta/2)\\
        e^{i\phi}\sin(\theta/2) & e^{i(\phi + \lambda)}\cos(\theta/2)
        \end{bmatrix}

"""
import numpy as np
from qat.lang.AQASM import AbstractGate


def u3_matrix(theta, phi, lamb):
    """
    A function that generates the matrix implementation of the U3 gate.
    """
    return np.array(
        [
            [np.cos(theta / 2), -np.exp(1j * lamb) * np.sin(theta / 2)],
            [
                np.exp(1j * phi) * np.sin(theta / 2),
                np.exp(1j * (phi + lamb)) * np.cos(theta / 2),
            ],
        ]
    )


U3 = AbstractGate("U3", [float] * 3, arity=1, matrix_generator=u3_matrix)
