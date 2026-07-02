import numpy as np
from typing import Sequence


def generate_perlin_noise_2d(shape: Sequence[int], res: Sequence[int]) -> np.ndarray:
    def f(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1) * 0.5 + 0.5


def generate_fractal_noise_2d(
    xSize=20,
    ySize=20,
    xSamples=1600,
    ySamples=1600,
    frequency=10,
    fractalOctaves=2,
    fractalLacunarity=2.0,
    fractalGain=0.25,
    zScale=0.23,
    centering=False,  # If True, the noise will be centered around 0
) -> np.ndarray:
    xScale = int(frequency * xSize)
    yScale = int(frequency * ySize)
    amplitude = 1

    # check to make sure the sample shape is the multiple of scale shape
    expected_xSamples = int(xScale * (fractalLacunarity**fractalOctaves))
    expected_ySamples = int(yScale * (fractalLacunarity**fractalOctaves))

    if xSamples > expected_xSamples or ySamples > expected_ySamples:
        raise RuntimeError(
            "Situation not checked, using expected_*Samples is in case the *Samples is not the multiple of *Size"
        )

    noise = np.zeros((expected_xSamples, expected_ySamples))
    for _ in range(fractalOctaves):
        noise += amplitude * generate_perlin_noise_2d(noise.shape, (xScale, yScale)) * zScale
        amplitude *= fractalGain
        xScale, yScale = int(fractalLacunarity * xScale), int(fractalLacunarity * yScale)

    if centering:
        noise -= np.mean(noise)

    return noise[:xSamples, :ySamples].copy()
