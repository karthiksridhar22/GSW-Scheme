import dataclasses

import numpy as np

import numpy as np
import dataclasses

@dataclasses.dataclass
class Polynomial:
    """A polynomial in the ring Z_q[x] / (x^N + 1)"""

    N: int
    coeff: np.ndarray

def build_monomial(c: int, i: int, N: int) -> Polynomial:
    """Build a monomial c*x^i in the ring Z[x]/(x^N + 1)"""
    coeff = np.zeros(N, dtype=np.int32)

    # Find k such that: 0 <= i + k*N < N
    i_mod_N = i % N
    k = (i_mod_N - i) // N

    # If k is odd then the monomial picks up a negative sign since:
    # x^i = (-1)^k * x^(i + k*N) = (-1)^k * x^(i % N)
    sign = 1 if k % 2 == 0 else -1

    coeff[i_mod_N] = sign * c
    return Polynomial(N=N, coeff=coeff)

@dataclasses.dataclass
class RlweConfig:
    degree: int  # Messages will be in the space Z[X]/(x^degree + 1)
    noise_std: float  # The std of the noise added during encryption.


@dataclasses.dataclass
class RlweEncryptionKey:
    config: RlweConfig
    key: Polynomial


@dataclasses.dataclass
class RlwePlaintext:
    config: RlweConfig
    message: Polynomial


@dataclasses.dataclass
class RlweCiphertext:
    config: RlweConfig
    a: Polynomial
    b: Polynomial

