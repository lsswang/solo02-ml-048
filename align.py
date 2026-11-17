"""Ordinary point-based translation registration; no projective fit yet."""
import numpy as np


def fit(source, destination, weights):
    source = np.asarray(source, dtype=float)
    destination = np.asarray(destination, dtype=float)
    weights = np.asarray(weights, dtype=float)
    shift = np.average(destination - source, axis=0, weights=weights)
    matrix = np.eye(3)
    matrix[:2, 2] = shift
    return {"status": "ok", "matrix": matrix.tolist()}


def project(matrix, points):
    points = np.asarray(points, dtype=float)
    homogeneous = np.c_[points, np.ones(len(points))] @ np.asarray(matrix).T
    return homogeneous[:, :2] / homogeneous[:, 2, None]
