import lwe
import numpy as np
import dataclasses
from collections.abc import Sequence

#for multiplication we need to redefine our scheme, we also need to define the message space as polynomials 

@dataclasses.dataclass
class Polynomial:
    #A polynomial in the ring Z_q[x] / (x^N + 1)
    N: int
    coeff: np.ndarray

def polynomial_constant_multiply(c: int, p: Polynomial) -> Polynomial:
    return Polynomial(N=p.N, coeff=np.multiply(c, p.coeff, dtype=np.int32))


def polynomial_multiply(p1: Polynomial, p2: Polynomial) -> Polynomial:
    #Multiply two negacyclic polynomials.
    N = p1.N

    # Multiply and pad the result to have length 2N-1
    prod = np.polymul(p1.coeff[::-1], p2.coeff[::-1])[::-1]
    prod_padded = np.zeros(2 * N - 1, dtype=np.int32)
    prod_padded[: len(prod)] = prod

    # Use the relation x^N = -1 to obtain a polynomial of degree N-1
    result = prod_padded[:N]
    result[:-1] -= prod_padded[N:]
    return Polynomial(N=N, coeff=result)


def polynomial_add(p1: Polynomial, p2: Polynomial) -> Polynomial:
    return Polynomial(N=p1.N, coeff=np.add(p1.coeff, p2.coeff, dtype=np.int32))

def polynomial_subtract(p1: Polynomial, p2: Polynomial) -> Polynomial:
    return Polynomial(
        N=p1.N, coeff=np.subtract(p1.coeff, p2.coeff, dtype=np.int32)
    )

def zero_polynomial(N: int) -> Polynomial:
    return Polynomial(N=N, coeff=np.zeros(N, dtype=np.int32))


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


def polynomial_to_string(p):
    terms = []
    for i, coeff in enumerate(p.coeff):
        if coeff != 0:  # We only add non-zero terms
            if i == 0:
                terms.append(f"{coeff}")
            elif i == 1:
                terms.append(f"{coeff}x")
            else:
                terms.append(f"{coeff}x^{i}")
    # Join all terms, handling the signs correctly
    return " + ".join(terms).replace("+-", "- ")


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


def rlwe_encode(p: Polynomial, config: RlweConfig) -> RlwePlaintext:
    #Encode a polynomial with coefficients in [-4, 4) as an RLWE plaintext.
    encode_coeff = np.array([lwe.encode(i) for i in p.coeff])
    return RlwePlaintext(
        config=config, message=Polynomial(N=p.N, coeff=encode_coeff)
    )


def rlwe_decode(plaintext: RlwePlaintext) -> Polynomial:
    #Decode an RLWE plaintext to a polynomial with coefficients in [-4, 4) mod 8.
    decode_coeff = np.array([lwe.decode(i) for i in plaintext.message.coeff])
    return Polynomial(N=plaintext.message.N, coeff=decode_coeff)


def build_zero_rlwe_plaintext(config: RlweConfig) -> RlwePlaintext:
    #Build a an RLWE plaintext containing the zero 
    return RlwePlaintext(
        config=config, message=zero_polynomial(config.degree)
    )


def build_monomial_rlwe_plaintext(
    c: int, i: int, config: RlweConfig
) -> RlwePlaintext:
    #Build an RLWE plaintext containing the monomial c*x^i
    return RlwePlaintext(
        config=config, message=build_monomial(c, i, config.degree)
    )


def convert_lwe_key_to_rlwe(lwe_key: lwe.LweEncryptionKey) -> RlweEncryptionKey:
    rlwe_config = RlweConfig(
        degree=lwe_key.config.dimension, noise_std=lwe_key.config.noise_std
    )
    return RlweEncryptionKey(
        config=rlwe_config,
        key=Polynomial(N=rlwe_config.degree, coeff=lwe_key.key),
    )


def rlwe_encrypt(
    plaintext: RlwePlaintext, key: RlweEncryptionKey
) -> RlweCiphertext:
    a = Polynomial(
        N=key.config.degree,
        coeff=lwe.uniform_sample_int32(size=key.config.degree),
    )
    noise = Polynomial(
        N=key.config.degree,
        coeff=lwe.gaussian_sample_int32(
            std=key.config.noise_std, size=key.config.degree
        ),
    )

    b = polynomial_add(
        polynomial_multiply(a, key.key), plaintext.message
    )
    b = polynomial_add(b, noise)

    return RlweCiphertext(config=key.config, a=a, b=b)


def rlwe_decrypt(
    ciphertext: RlweCiphertext, key: RlweEncryptionKey
) -> RlwePlaintext:
    message = polynomial_subtract(
        ciphertext.b, polynomial_multiply(ciphertext.a, key.key)
    )
    return RlwePlaintext(config=key.config, message=message)

#############################################################################
#Implementing gsw scheme

@dataclasses.dataclass
class GswConfig:
    rlwe_config: RlweConfig
    log_p: int  # Homomorphic multiplication will use the base-2^log_p representation.


@dataclasses.dataclass
class GswPlaintext:
    config: GswConfig
    message: Polynomial


@dataclasses.dataclass
class GswCiphertext:
    config: GswConfig
    rlwe_ciphertexts: Sequence[RlweCiphertext]


@dataclasses.dataclass
class GswEncryptionKey:
    config: GswConfig
    key: Polynomial


def base_p_num_powers(log_p: int):
    """Return the size of a base 2^log_p representation of an int32."""
    return 32 // log_p


def array_to_base_p(a: np.ndarray, log_p: int) -> Sequence[np.ndarray]:
    """Compute the base 2^log_p representation of each element in a.

    a: An array of type int32
    log_p: Compute the representation in base 2^log_p
    """
    num_powers = base_p_num_powers(log_p)
    half_p = np.int32(2 ** (log_p - 1))
    offset = half_p * sum(2 ** (i * log_p) for i in range(num_powers))
    mask = 2 ** (log_p) - 1

    a_offset = (a + offset).astype(np.uint32)

    output = []
    for i in range(num_powers):
        output.append(
            (np.right_shift(a_offset, i * log_p) & mask).astype(np.int32)
            - half_p
        )

    return output


def base_p_to_array(a_base_p: Sequence[np.ndarray], log_p) -> np.ndarray:
    """Reconstruct an array of int32s from its base 2^log_p representation."""
    return sum(2 ** (i * log_p) * x for i, x in enumerate(a_base_p)).astype(
        np.int32
    )


def polynomial_to_base_p(
    f: Polynomial, log_p: int
) -> Sequence[Polynomial]:
    """Compute the base 2^log_p of the polynomial f."""
    return [
        Polynomial(coeff=v, N=f.N)
        for v in array_to_base_p(f.coeff, log_p=log_p)
    ]


def base_p_to_polynomial(
    f_base_p: Sequence[Polynomial], log_p: int
) -> Polynomial:
    """Recover the polynomial f from its base 2^log_p representation."""
    f = zero_polynomial(f_base_p[0].N)

    for i, level in enumerate(f_base_p):
        p_i = 2 ** (i * log_p)
        f = polynomial_add(
            f, polynomial_constant_multiply(p_i, level)
        )

    return f


def convert_lwe_key_to_gsw(
    lwe_key: lwe.LweEncryptionKey, gsw_config: GswConfig
) -> GswEncryptionKey:
    return GswEncryptionKey(
        config=gsw_config,
        key=Polynomial(
            N=gsw_config.rlwe_config.degree, coeff=lwe_key.key
        ),
    )


def convert_rlwe_key_to_gsw(
    rlwe_key: RlweEncryptionKey, gsw_config: GswConfig
) -> GswEncryptionKey:
    return GswEncryptionKey(config=gsw_config, key=rlwe_key.key)


def convert_gws_key_to_rlwe(
    gsw_key: GswEncryptionKey,
) -> RlweEncryptionKey:
    return RlweEncryptionKey(
        config=gsw_key.config.rlwe_config, key=gsw_key.key
    )


def gsw_encrypt(
    plaintext: GswPlaintext, key: GswEncryptionKey
) -> GswCiphertext:
    gsw_config = key.config
    num_powers = base_p_num_powers(log_p=gsw_config.log_p)

    # Create 2 RLWE encryptions of 0 for each element of a base-p representation.
    rlwe_key = convert_gws_key_to_rlwe(key)
    rlwe_plaintext_zero = build_zero_rlwe_plaintext(gsw_config.rlwe_config)
    rlwe_ciphertexts = [
        rlwe_encrypt(rlwe_plaintext_zero, rlwe_key)
        for _ in range(2 * num_powers)
    ]

    # Add multiples p^i * message to the rlwe ciphertexts
    for i in range(num_powers):
        p_i = 2 ** (i * gsw_config.log_p)
        scaled_message = polynomial_constant_multiply(
            p_i, plaintext.message
        )

        rlwe_ciphertexts[i].a = polynomial_add(
            rlwe_ciphertexts[i].a, scaled_message
        )

        b_idx = i + num_powers
        rlwe_ciphertexts[b_idx].b = polynomial_add(
            rlwe_ciphertexts[b_idx].b, scaled_message
        )

    return GswCiphertext(gsw_config, rlwe_ciphertexts)


def gsw_multiply(
    gsw_ciphertext: GswCiphertext, rlwe_ciphertext: RlweCiphertext
) -> RlweCiphertext:
    gsw_config = gsw_ciphertext.config
    rlwe_config = rlwe_ciphertext.config

    # Concatenate the base-p representations of rlwe_ciphertext.a and rlwe_ciphertext.b
    rlwe_base_p = polynomial_to_base_p(
        rlwe_ciphertext.a, log_p=gsw_config.log_p
    ) + polynomial_to_base_p(rlwe_ciphertext.b, log_p=gsw_config.log_p)

    # Multiply the row vector rlwe_base_p with the
    # len(rlwe_base_p)x2 matrix gsw_ciphertext.rlwe_ciphertexts.
    rlwe_ciphertext = RlweCiphertext(
        config=rlwe_config,
        a=zero_polynomial(rlwe_config.degree),
        b=zero_polynomial(rlwe_config.degree),
    )

    for i, p in enumerate(rlwe_base_p):
        rlwe_ciphertext.a = polynomial_add(
            rlwe_ciphertext.a,
            polynomial_multiply(
                p, gsw_ciphertext.rlwe_ciphertexts[i].a
            ),
        )
        rlwe_ciphertext.b = polynomial_add(
            rlwe_ciphertext.b,
            polynomial_multiply(
                p, gsw_ciphertext.rlwe_ciphertexts[i].b
            ),
        )

    return rlwe_ciphertext

