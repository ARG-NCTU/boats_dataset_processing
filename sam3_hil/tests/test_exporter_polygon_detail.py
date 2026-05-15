"""
Regression checks for high-detail polygon export.
"""

import numpy as np

from src.core.exporter import mask_to_polygon


def test_unsimplified_polygon_keeps_dense_contour_points():
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[4:20, 6:18] = 1

    simplified = mask_to_polygon(mask, simplify=True)
    unsimplified = mask_to_polygon(mask, simplify=False)

    assert len(simplified) == 1
    assert len(unsimplified) == 1
    assert len(unsimplified[0]) > len(simplified[0])
    assert len(unsimplified[0]) > 20
