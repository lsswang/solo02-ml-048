"""Nearest sampling and independent per-pixel source choice for ordinary mosaics."""
import numpy as np


def warp(image, mask, matrix, shape):
    image = np.asarray(image, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    inverse = np.linalg.inv(np.asarray(matrix, dtype=float))
    values = np.zeros(shape)
    valid = np.zeros(shape, dtype=bool)
    for y, x in np.ndindex(shape):
        point = inverse @ [x, y, 1.0]
        if abs(point[2]) < 1e-9:
            continue
        sx, sy = np.rint(point[:2] / point[2]).astype(int)
        if 0 <= sy < image.shape[0] and 0 <= sx < image.shape[1] and mask[sy, sx]:
            values[y, x] = image[sy, sx]
            valid[y, x] = True
    return values, valid


def compose(request, matrices):
    reference = np.asarray(request["reference"], dtype=float)
    warped = [warp(item["image"], item["mask"], matrix, reference.shape)
              for item, matrix in zip(request["sources"], matrices)]
    labels = np.full(reference.shape, -1, dtype=int)
    output = np.full(reference.shape, np.nan)
    for y, x in np.ndindex(reference.shape):
        candidates = [i for i, (_, valid) in enumerate(warped) if valid[y, x]]
        if candidates:
            i = min(candidates, key=lambda i: request["sources"][i]["confidence"] *
                    (warped[i][0][y, x] - reference[y, x]) ** 2)
            labels[y, x] = i
            output[y, x] = warped[i][0][y, x]
    return {"labels": labels.tolist(), "image": output.tolist()}
