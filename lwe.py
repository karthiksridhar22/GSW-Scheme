import dataclasses
import numpy as np
from typing import Optional

INT32_MIN = np.iinfo(np.int32).min
INT32_MAX = np.iinfo(np.int32).max

def uniform_sample_int32(size: int) -> np.ndarray:
    return np.random.randint(
        low=INT32_MIN,
        high=INT32_MAX + 1,
        size=size,
        dtype=np.int32,
    )

def gaussian_sample_int32(std: float, size: Optional[float]) -> np.ndarray:
    return np.int32(INT32_MAX * np.random.normal(loc=0.0, scale=std, size=size))


def encode(i: int) -> np.int32:
    """Encode an integer in [-4, 4) as an int32"""
    return np.multiply(i, 1 << 29, dtype=np.int32)


def decode(i: np.int32) -> int:
    """Decode an int32 to an integer in the range [-4, 4) mod 8"""
    d = int(np.rint(i / (1 << 29)))
    return ((d + 4) % 8) - 4

@dataclasses.dataclass
class LweConfig:
    # Size of the LWE encryption key.
    dimension: int

    # Standard deviation of the encryption noise.
    noise_std: float


@dataclasses.dataclass
class LwePlaintext:
    message: np.int32


@dataclasses.dataclass
class LweCiphertext:
    config: LweConfig
    a: np.ndarray  # An int32 array of size config.dimension
    b: np.int32


@dataclasses.dataclass
class LweEncryptionKey:
    config: LweConfig
    key: np.ndarray  # An int32 array of size config.dimension


def lwe_encode(i: int) -> LwePlaintext:
    """Encode an integer in [-4,4) as an LWE plaintext."""
    return LwePlaintext(encode(i))


def lwe_decode(plaintext: LwePlaintext) -> int:
    """Decode an LWE plaintext to an integer in [-4,4) mod 8."""
    return decode(plaintext.message)

def generate_lwe_key(config: LweConfig) -> LweEncryptionKey:
    return LweEncryptionKey(
        config=config,
        key=np.random.randint(
            low=0, high=2, size=(config.dimension,), dtype=np.int32
        ),
    )


def lwe_encrypt(
    plaintext: LwePlaintext, key: LweEncryptionKey
) -> LweCiphertext:
    a = uniform_sample_int32(size=key.config.dimension)
    noise = gaussian_sample_int32(std=key.config.noise_std, size=None)

    # b = (a, key) + message + noise
    b = np.add(np.dot(a, key.key), plaintext.message, dtype=np.int32)
    b = np.add(b, noise, dtype=np.int32)

    return LweCiphertext(config=key.config, a=a, b=b)


def lwe_decrypt(
    ciphertext: LweCiphertext, key: LweEncryptionKey
) -> LwePlaintext:
    return LwePlaintext(
        np.subtract(ciphertext.b, np.dot(ciphertext.a, key.key), dtype=np.int32)
    )

def lwe_add(
    ciphertext_left: LweCiphertext, ciphertext_right: LweCiphertext
) -> LweCiphertext:
    """Homomorphically add two LWE ciphertexts."""
    return LweCiphertext(
        ciphertext_left.config,
        np.add(ciphertext_left.a, ciphertext_right.a, dtype=np.int32),
        np.add(ciphertext_left.b, ciphertext_right.b, dtype=np.int32),
    )

