"""
Utility functions for spatial calculations, coordinate scaling, and pyvroom matrix formatting.
"""

from typing import Any, Iterable, List, Sequence, Tuple, Union
import numpy as np

# Default integer scaling factor for spatial coordinates and temporal metrics
DEFAULT_SCALE_FACTOR: int = 100


def scale_value(val: Union[int, float], scale_factor: int = DEFAULT_SCALE_FACTOR) -> int:
    """
    Scale a floating-point metric by the integer scaling factor and round to nearest integer.
    
    Args:
        val: Numeric value (coordinate, time window bound, or service duration).
        scale_factor: Multiplier to preserve precision in integer domain (default: 100).
        
    Returns:
        Integer scaled representation.
    """
    return int(round(float(val) * scale_factor))


def unscale_value(val: Union[int, float], scale_factor: int = DEFAULT_SCALE_FACTOR) -> float:
    """
    Convert an integer-scaled metric back to original continuous units for human display.
    
    Args:
        val: Scaled integer value.
        scale_factor: Scaling factor previously applied (default: 100).
        
    Returns:
        Original metric in floating-point format.
    """
    return float(val) / float(scale_factor)


def compute_euclidean_distance(
    p1: Tuple[Union[int, float], Union[int, float]],
    p2: Tuple[Union[int, float], Union[int, float]],
    scale_factor: int = DEFAULT_SCALE_FACTOR,
    already_scaled: bool = False,
) -> int:
    """
    Compute scaled Euclidean distance between two points, rounded to the nearest integer.
    
    Args:
        p1: (x, y) coordinate pair for point 1.
        p2: (x, y) coordinate pair for point 2.
        scale_factor: Integer scaling factor (default: 100).
        already_scaled: If True, coordinates are assumed already scaled by scale_factor.
        
    Returns:
        Integer Euclidean distance.
    """
    dx = float(p1[0]) - float(p2[0])
    dy = float(p1[1]) - float(p2[1])
    raw_dist = np.hypot(dx, dy)
    
    if already_scaled:
        return int(round(raw_dist))
    return int(round(raw_dist * scale_factor))


def compute_distance_matrix(
    coordinates: Sequence[Tuple[Union[int, float], Union[int, float]]],
    scale_factor: int = DEFAULT_SCALE_FACTOR,
    already_scaled: bool = False,
) -> np.ndarray:
    """
    Compute a 2D integer distance matrix using scaled Euclidean distances.
    
    This matrix serves as both the spatial distance matrix and the travel duration
    matrix (under standard unit-speed assumptions in Solomon VRPTW benchmarks).
    
    Args:
        coordinates: Ordered sequence of (x, y) coordinates where index 0 is depot
                     and indices 1..N correspond to customer nodes.
        scale_factor: Scaling factor (default: 100).
        already_scaled: Whether the passed coordinates are already scaled integers.
        
    Returns:
        2D NumPy array of shape (N, N) with dtype np.uintc (compatible with pyvroom).
    """
    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coordinates array of shape (N, 2), got {coords.shape}")

    n = coords.shape[0]
    
    # Compute pairwise Euclidean differences
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    raw_dist = np.hypot(diff[:, :, 0], diff[:, :, 1])
    
    if not already_scaled:
        scaled_dist = np.round(raw_dist * scale_factor)
    else:
        scaled_dist = np.round(raw_dist)

    # Ensure diagonal is strictly zero
    np.fill_diagonal(scaled_dist, 0)
    
    # Use np.uintc for cross-platform pybind11 buffer alignment (MSVC / GCC)
    return scaled_dist.astype(np.uintc)


def to_vroom_matrix(matrix: Union[np.ndarray, Sequence[Sequence[int]]]) -> Any:
    """
    Safely wrap a 2D integer distance/duration matrix into a pyvroom Matrix buffer object.
    
    This function avoids the 'RuntimeError: Incompatible buffer format!' on 64-bit Windows
    systems running NumPy 2.x by explicitly coercing to np.uintc ('I') before pybind11 ingestion.
    
    Args:
        matrix: 2D square matrix or sequence of sequences.
        
    Returns:
        vroom._vroom.Matrix instance ready for problem.set_durations_matrix().
    """
    import vroom  # lazy import
    
    arr = np.asarray(matrix, dtype=np.uintc)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Matrix must be 2D square array, got shape {arr.shape}")
        
    return vroom._vroom.Matrix(arr)


def resolve_location_index(loc: Any) -> int:
    """
    Safely extract integer location index from either an int, numpy scalar, or pyvroom LocationIndex object.
    
    Args:
        loc: Location index or pyvroom.LocationIndex instance.
        
    Returns:
        Integer location index.
    """
    if hasattr(loc, "index"):
        return int(loc.index)
    return int(loc)
