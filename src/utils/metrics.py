"""Documentation string."""

from __future__ import annotations

import numpy as np

from src.utils.quality_selection import HEX_METRICS, metric_names

__all__ = [
    "SignedPolygonArea",
    "ComputeNodeJacobian",
    "ComputeQuadJacobian",
    "CheckLineSegmentsIntersect",
    "CheckSelfIntersection",
    "CheckConcave",
    "ComputeQuadAngle",
    "ComputeQuadSkewness",
    "ComputeQuadOrthogonality",
    "ComputeQuadAspectRatio",
    "ComputeQuadParallelDeviation",
    "QuadMeshQualityEvaluator",
    "test_public_quality_functions",
    "test_quad_mesh_quality_evaluator",
]

_GEOMETRY_TOLERANCE = 1e-12


def _as_points(points, *, name="points", minimum_points=1) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("""Documentation string.""")
    if len(array) < minimum_points:
        raise ValueError("""Documentation string.""")
    if not np.isfinite(array).all():
        raise ValueError("""Documentation string.""")
    return array


def _as_quad(quad) -> np.ndarray:
    array = _as_points(quad, name="quad", minimum_points=4)
    if array.shape != (4, 2):
        raise ValueError("""Documentation string.""")
    return array


def _safe_unit(vector: np.ndarray, *, name="vector") -> np.ndarray:
    length = np.linalg.norm(vector)
    if length <= _GEOMETRY_TOLERANCE:
        raise ValueError("""Documentation string.""")
    return vector / length


# ============================================================
# Implementation comment.
# ============================================================
def SignedPolygonArea(polygon) -> float:
    """Documentation string."""
    points = _as_points(polygon, name="polygon", minimum_points=3)
    x, y = points[:, 0], points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def ComputeNodeJacobian(polygon) -> np.ndarray:
    """Documentation string."""
    points = np.asarray(polygon, dtype=float)
    if points.ndim < 2 or points.shape[-1] != 2 or points.shape[-2] < 3:
        raise ValueError("""Documentation string.""")
    if not np.isfinite(points).all():
        raise ValueError("""Documentation string.""")
    backward = points - np.roll(points, 1, axis=-2)
    forward = np.roll(points, -1, axis=-2) - points
    return backward[..., 0] * forward[..., 1] - backward[..., 1] * forward[..., 0]


def ComputeQuadJacobian(quad, xi=0.0, eta=0.0):
    """Documentation string."""
    points = _as_quad(quad)
    if not np.isfinite([xi, eta]).all() or not (-1 <= xi <= 1 and -1 <= eta <= 1):
        raise ValueError("""Documentation string.""")
    dN_dxi = 0.25 * np.array(
        [
            -(1 - eta),
            1 - eta,
            1 + eta,
            -(1 + eta),
        ]
    )
    dN_deta = 0.25 * np.array(
        [
            -(1 - xi),
            -(1 + xi),
            1 + xi,
            1 - xi,
        ]
    )
    jacobian = np.column_stack((dN_dxi @ points, dN_deta @ points))
    return float(np.linalg.det(jacobian)), jacobian


def _orientation(a, b, c, tolerance) -> int:
    first = b - a
    second = c - a
    cross = first[0] * second[1] - first[1] * second[0]
    if abs(cross) <= tolerance:
        return 0
    return 1 if cross > 0 else -1


def _point_on_segment(point, start, end, tolerance) -> bool:
    return (
        _orientation(start, end, point, tolerance) == 0
        and np.all(point >= np.minimum(start, end) - tolerance)
        and np.all(point <= np.maximum(start, end) + tolerance)
    )


def CheckLineSegmentsIntersect(A1, A2, B1, B2, tolerance=1e-12) -> bool:
    """Documentation string."""
    values = np.asarray([A1, A2, B1, B2], dtype=float)
    if values.shape != (4, 2) or not np.isfinite(values).all():
        raise ValueError("""Documentation string.""")
    if tolerance < 0:
        raise ValueError("""Documentation string.""")
    a1, a2, b1, b2 = values
    o1 = _orientation(a1, a2, b1, tolerance)
    o2 = _orientation(a1, a2, b2, tolerance)
    o3 = _orientation(b1, b2, a1, tolerance)
    o4 = _orientation(b1, b2, a2, tolerance)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    return (
        _point_on_segment(b1, a1, a2, tolerance)
        or _point_on_segment(b2, a1, a2, tolerance)
        or _point_on_segment(a1, b1, b2, tolerance)
        or _point_on_segment(a2, b1, b2, tolerance)
    )


def CheckSelfIntersection(polygon, tolerance=1e-12) -> bool:
    """Documentation string."""
    points = _as_points(polygon, name="polygon", minimum_points=4)
    count = len(points)
    for first in range(count):
        first_next = (first + 1) % count
        for second in range(first + 1, count):
            second_next = (second + 1) % count
            if first == second or first_next == second or second_next == first:
                continue
            if CheckLineSegmentsIntersect(
                points[first],
                points[first_next],
                points[second],
                points[second_next],
                tolerance,
            ):
                return True
    return False


def CheckConcave(polygon, tolerance=1e-12) -> bool:
    """Documentation string."""
    points = _as_points(polygon, name="polygon", minimum_points=4)
    if CheckSelfIntersection(points, tolerance):
        return False
    jacobians = ComputeNodeJacobian(points)
    nonzero = jacobians[np.abs(jacobians) > tolerance]
    if len(nonzero) < len(points):
        return False
    return bool(np.any(nonzero > 0) and np.any(nonzero < 0))


# ============================================================
# Implementation comment.
# ============================================================
def ComputeQuadAngle(quad) -> np.ndarray:
    """Documentation string."""
    points = _as_quad(quad)
    angles = np.empty(4)
    orientation = np.sign(SignedPolygonArea(points))
    for index in range(4):
        previous = points[(index - 1) % 4] - points[index]
        following = points[(index + 1) % 4] - points[index]
        previous = _safe_unit(previous, name="""Documentation string.""")
        following = _safe_unit(following, name="""Documentation string.""")
        cosine = np.clip(np.dot(previous, following), -1.0, 1.0)
        smaller_angle = np.degrees(np.arccos(cosine))
        backward = -previous
        corner_cross = backward[0] * following[1] - backward[1] * following[0]
        if orientation != 0 and corner_cross * orientation < 0:
            angles[index] = 360.0 - smaller_angle
        else:
            angles[index] = smaller_angle
    return angles


def ComputeQuadSkewness(quad) -> float:
    """Documentation string."""
    angles = ComputeQuadAngle(quad)
    skewness = max((angles.max() - 90.0) / 90.0, (90.0 - angles.min()) / 90.0)
    return float(np.clip(skewness, 0.0, 1.0))


def ComputeQuadOrthogonality(quad) -> float:
    """Documentation string."""
    points = _as_quad(quad)
    if abs(SignedPolygonArea(points)) <= _GEOMETRY_TOLERANCE:
        return 0.0
    center = np.mean(points, axis=0)
    qualities = []
    for index in range(4):
        start, end = points[index], points[(index + 1) % 4]
        edge = end - start
        edge_unit = _safe_unit(edge, name="""Documentation string.""")
        normal = np.array([-edge_unit[1], edge_unit[0]])
        center_to_face = 0.5 * (start + end) - center
        distance = np.linalg.norm(center_to_face)
        if distance <= _GEOMETRY_TOLERANCE:
            return 0.0
        qualities.append(abs(np.dot(center_to_face / distance, normal)))
    return float(np.clip(min(qualities), 0.0, 1.0))


# ============================================================
# Implementation comment.
# ============================================================
def ComputeQuadAspectRatio(quad) -> float:
    """Documentation string."""
    points = _as_quad(quad)
    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    if lengths.min() <= _GEOMETRY_TOLERANCE:
        raise ValueError("""Documentation string.""")
    return float(lengths.max() / lengths.min())


def ComputeQuadParallelDeviation(quad) -> float:
    """Documentation string."""
    points = _as_quad(quad)
    edges = np.roll(points, -1, axis=0) - points
    units = np.array(
        [_safe_unit(edge, name="""Documentation string.""") for edge in edges]
    )
    deviations = []
    for first, second in ((0, 2), (1, 3)):
        cosine = np.clip(abs(np.dot(units[first], units[second])), 0.0, 1.0)
        deviations.append(np.degrees(np.arccos(cosine)))
    return float(max(deviations))


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    inside = False
    x, y = point
    for index in range(len(polygon)):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % len(polygon)]
        if (y1 > y) != (y2 > y):
            intersection_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < intersection_x:
                inside = not inside
    return inside


def _polygons_overlap(first: np.ndarray, second: np.ndarray) -> bool:
    """Detect crossings/overlap without counting a shared endpoint as overlap.

    Cut boundaries may repeat a physical endpoint under different node IDs.
    Ignore endpoint contact and positive-length collinear shared edges, including
    duplicated C-grid seam edges. Retain T-junctions, proper crossings, identical
    cells and strict containment.
    """
    tolerance = 1e-12
    if len(first) == len(second) and all(
        any(np.max(np.abs(point - other)) <= tolerance for other in second)
        for point in first
    ):
        return True
    for i in range(len(first)):
        a, b = first[i], first[(i + 1) % len(first)]
        for j in range(len(second)):
            c, d = second[j], second[(j + 1) % len(second)]
            if not CheckLineSegmentsIntersect(a, b, c, d, tolerance):
                continue
            shared_endpoint = any(
                np.max(np.abs(x - y)) <= tolerance for x in (a, b) for y in (c, d)
            )
            if not shared_endpoint:
                return True
            # A positive-length collinear overlap is a shared physical edge.
            # Structured C-grid seams may duplicate that edge with different node
            # IDs, so boundary-only contact must not be reported as cell overlap.
            # Genuine area overlap is still detected below by strict containment,
            # while proper crossings and T-junctions return above.

    # Checking all vertices also catches containment when vertex zero happens
    # to be the shared endpoint. Boundary points are not strictly interior.
    for vertices, polygon in ((first, second), (second, first)):
        for point in vertices:
            on_boundary = any(
                _point_on_segment(
                    point, polygon[i], polygon[(i + 1) % len(polygon)], tolerance
                )
                for i in range(len(polygon))
            )
            if not on_boundary and _point_in_polygon(point, polygon):
                return True
    return False


def _segments_intersect_many(a, b, c, d, tolerance=_GEOMETRY_TOLERANCE):
    """Vectorized closed-segment intersection for equally sized point arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)

    def cross(first, second):
        return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]

    def orientation(start, end, point):
        return cross(end - start, point - start)

    def on_segment(point, start, end, value):
        return (
            (np.abs(value) <= tolerance)
            & np.all(point >= np.minimum(start, end) - tolerance, axis=-1)
            & np.all(point <= np.maximum(start, end) + tolerance, axis=-1)
        )

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    proper = (
        (((o1 > tolerance) & (o2 < -tolerance)) | ((o1 < -tolerance) & (o2 > tolerance)))
        & (((o3 > tolerance) & (o4 < -tolerance)) | ((o3 < -tolerance) & (o4 > tolerance)))
    )
    return (
        proper
        | on_segment(c, a, b, o1)
        | on_segment(d, a, b, o2)
        | on_segment(a, c, d, o3)
        | on_segment(b, c, d, o4)
    )


def _spatial_aabb_pairs(minimum, maximum):
    """Return possible AABB overlaps using a conservative uniform spatial hash."""
    minimum = np.asarray(minimum, dtype=float)
    maximum = np.asarray(maximum, dtype=float)
    count = len(minimum)
    if count < 2:
        return np.empty((0, 2), dtype=int)
    lower = np.min(minimum, axis=0)
    upper = np.max(maximum, axis=0)
    span = np.maximum(upper - lower, np.finfo(float).eps)
    base = max(1, int(np.ceil(np.sqrt(count))))
    aspect = float(np.sqrt(span[0] / span[1]))
    bins_x = int(np.clip(round(base * aspect), 1, 256))
    bins_y = int(np.clip(round(base / max(aspect, np.finfo(float).eps)), 1, 256))
    bins = np.array([bins_x, bins_y], dtype=int)
    first_bin = np.floor((minimum - lower) / span * bins).astype(int)
    last_bin = np.floor((maximum - lower) / span * bins).astype(int)
    first_bin = np.clip(first_bin, 0, bins - 1)
    last_bin = np.clip(last_bin, 0, bins - 1)
    buckets = {}
    for cell in range(count):
        for column in range(first_bin[cell, 0], last_bin[cell, 0] + 1):
            for row in range(first_bin[cell, 1], last_bin[cell, 1] + 1):
                buckets.setdefault((column, row), []).append(cell)
    pairs = set()
    for cells in buckets.values():
        for offset, first in enumerate(cells[:-1]):
            for second in cells[offset + 1 :]:
                pairs.add((first, second) if first < second else (second, first))
    if not pairs:
        return np.empty((0, 2), dtype=int)
    result = np.asarray(sorted(pairs), dtype=int)
    first, second = result[:, 0], result[:, 1]
    overlaps = np.all(
        (minimum[first] <= maximum[second] + _GEOMETRY_TOLERANCE)
        & (minimum[second] <= maximum[first] + _GEOMETRY_TOLERANCE),
        axis=1,
    )
    return result[overlaps]


def _polygons_overlap_many(first, second, tolerance=_GEOMETRY_TOLERANCE):
    """Vectorized form of ``_polygons_overlap`` for batches of quadrilaterals."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    count = len(first)
    if count == 0:
        return np.empty(0, dtype=bool)

    first_end = np.roll(first, -1, axis=1)
    second_end = np.roll(second, -1, axis=1)
    a = np.broadcast_to(first[:, :, None, :], (count, 4, 4, 2))
    b = np.broadcast_to(first_end[:, :, None, :], (count, 4, 4, 2))
    c = np.broadcast_to(second[:, None, :, :], (count, 4, 4, 2))
    d = np.broadcast_to(second_end[:, None, :, :], (count, 4, 4, 2))
    intersects = _segments_intersect_many(
        a.reshape(-1, 2),
        b.reshape(-1, 2),
        c.reshape(-1, 2),
        d.reshape(-1, 2),
        tolerance,
    ).reshape(count, 4, 4)
    shared_endpoint = (
        np.max(np.abs(a - c), axis=-1) <= tolerance
    ) | (
        np.max(np.abs(a - d), axis=-1) <= tolerance
    ) | (
        np.max(np.abs(b - c), axis=-1) <= tolerance
    ) | (
        np.max(np.abs(b - d), axis=-1) <= tolerance
    )
    crossing = np.any(intersects & ~shared_endpoint, axis=(1, 2))

    def strictly_inside(points, polygons):
        x = points[..., 0, None]
        y = points[..., 1, None]
        start = polygons[:, None, :, :]
        end = np.roll(polygons, -1, axis=1)[:, None, :, :]
        x1, y1 = start[..., 0], start[..., 1]
        x2, y2 = end[..., 0], end[..., 1]
        crosses_ray = (y1 > y) != (y2 > y)
        intersection_x = (x2 - x1) * (y - y1) / np.where(
            np.abs(y2 - y1) > tolerance, y2 - y1, 1.0
        ) + x1
        inside = np.count_nonzero(crosses_ray & (x < intersection_x), axis=-1) % 2 == 1
        edge = end - start
        relative = points[:, :, None, :] - start
        cross = edge[..., 0] * relative[..., 1] - edge[..., 1] * relative[..., 0]
        on_boundary = np.any(
            (np.abs(cross) <= tolerance)
            & np.all(points[:, :, None, :] >= np.minimum(start, end) - tolerance, axis=-1)
            & np.all(points[:, :, None, :] <= np.maximum(start, end) + tolerance, axis=-1),
            axis=-1,
        )
        return inside & ~on_boundary

    containment = np.any(strictly_inside(first, second), axis=1) | np.any(
        strictly_inside(second, first), axis=1
    )
    matches = np.max(
        np.abs(first[:, :, None, :] - second[:, None, :, :]), axis=-1
    ) <= tolerance
    identical = np.all(np.any(matches, axis=2), axis=1)
    return crossing | containment | identical


class QuadMeshQualityEvaluator:
    """Evaluate quadrilateral quality with cached vectorized geometry kernels."""

    def __init__(self, points: np.ndarray, quads: np.ndarray) -> None:
        self.points = _as_points(points, name="points", minimum_points=4)
        indices = np.asarray(quads)
        if indices.ndim != 2 or indices.shape[1] != 4:
            raise ValueError("""Documentation string.""")
        if not np.issubdtype(indices.dtype, np.integer):
            if not np.all(np.equal(indices, np.floor(indices))):
                raise ValueError("""Documentation string.""")
            indices = indices.astype(int)
        self.quads = indices.astype(int, copy=False)
        if len(self.quads) == 0:
            raise ValueError("""Documentation string.""")
        if self.quads.min() < 0 or self.quads.max() >= len(self.points):
            raise IndexError("""Documentation string.""")
        if any(len(set(cell)) != 4 for cell in self.quads):
            raise ValueError("""Documentation string.""")
        self.cells = self.points[self.quads]
        self._angles = None
        self._signed_area = None
        self._node_jacobian = None
        self._center_jacobian = None
        self._center_jacobian_matrix = None
        self._self_intersections = None
        self._concave_cells = None
        self._concave_vertex_count = None
        self._degenerate_cells = None
        self._flipping_cells = None
        self._intersecting_cells = None

    def SignedArea(self) -> np.ndarray:
        if self._signed_area is None:
            x = self.cells[..., 0]
            y = self.cells[..., 1]
            self._signed_area = 0.5 * np.sum(
                x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y,
                axis=1,
            )
        return self._signed_area.copy()

    def Area(self) -> np.ndarray:
        return np.abs(self.SignedArea())

    def NodeJacobian(self) -> np.ndarray:
        if self._node_jacobian is None:
            self._node_jacobian = ComputeNodeJacobian(self.cells)
        return self._node_jacobian.copy()

    def Jacobian(self, xi=0.0, eta=0.0):
        if not np.isfinite([xi, eta]).all() or not (-1 <= xi <= 1 and -1 <= eta <= 1):
            raise ValueError("xi and eta must be finite and lie in [-1, 1]")
        if xi == 0.0 and eta == 0.0 and self._center_jacobian is not None:
            return self._center_jacobian.copy(), self._center_jacobian_matrix.copy()
        dN_dxi = 0.25 * np.array(
            [-(1 - eta), 1 - eta, 1 + eta, -(1 + eta)]
        )
        dN_deta = 0.25 * np.array(
            [-(1 - xi), -(1 + xi), 1 + xi, 1 - xi]
        )
        derivative_xi = np.einsum("i,nij->nj", dN_dxi, self.cells)
        derivative_eta = np.einsum("i,nij->nj", dN_deta, self.cells)
        matrices = np.stack((derivative_xi, derivative_eta), axis=-1)
        determinants = (
            matrices[:, 0, 0] * matrices[:, 1, 1]
            - matrices[:, 0, 1] * matrices[:, 1, 0]
        )
        if xi == 0.0 and eta == 0.0:
            self._center_jacobian = determinants
            self._center_jacobian_matrix = matrices
        return determinants.copy(), matrices.copy()

    def Angle(self) -> np.ndarray:
        if self._angles is None:
            previous = np.roll(self.cells, 1, axis=1) - self.cells
            following = np.roll(self.cells, -1, axis=1) - self.cells
            previous_length = np.linalg.norm(previous, axis=2)
            following_length = np.linalg.norm(following, axis=2)
            if np.any(previous_length <= _GEOMETRY_TOLERANCE) or np.any(
                following_length <= _GEOMETRY_TOLERANCE
            ):
                raise ValueError("Cannot compute angles for zero-length edges")
            previous /= previous_length[..., None]
            following /= following_length[..., None]
            cosine = np.clip(np.sum(previous * following, axis=2), -1.0, 1.0)
            smaller = np.degrees(np.arccos(cosine))
            backward = -previous
            corner_cross = (
                backward[..., 0] * following[..., 1]
                - backward[..., 1] * following[..., 0]
            )
            orientation = np.sign(self.SignedArea())[:, None]
            self._angles = np.where(
                (orientation != 0) & (corner_cross * orientation < 0),
                360.0 - smaller,
                smaller,
            )
        return self._angles.copy()

    def MaxAngle(self) -> np.ndarray:
        return np.max(self.Angle(), axis=1)

    def MinAngle(self) -> np.ndarray:
        return np.min(self.Angle(), axis=1)

    def Skewness(self) -> np.ndarray:
        angles = self.Angle()
        values = np.maximum(
            (np.max(angles, axis=1) - 90.0) / 90.0,
            (90.0 - np.min(angles, axis=1)) / 90.0,
        )
        return np.clip(values, 0.0, 1.0)

    def Orthogonality(self) -> np.ndarray:
        edges = np.roll(self.cells, -1, axis=1) - self.cells
        edge_length = np.linalg.norm(edges, axis=2)
        safe_edge_length = np.maximum(edge_length, np.finfo(float).eps)
        edge_unit = edges / safe_edge_length[..., None]
        normals = np.stack((-edge_unit[..., 1], edge_unit[..., 0]), axis=-1)
        centers = np.mean(self.cells, axis=1, keepdims=True)
        face_vectors = 0.5 * (self.cells + np.roll(self.cells, -1, axis=1)) - centers
        face_distance = np.linalg.norm(face_vectors, axis=2)
        safe_face_distance = np.maximum(face_distance, np.finfo(float).eps)
        face_unit = face_vectors / safe_face_distance[..., None]
        qualities = np.abs(np.sum(face_unit * normals, axis=2))
        result = np.clip(np.min(qualities, axis=1), 0.0, 1.0)
        invalid = (
            np.abs(self.SignedArea()) <= _GEOMETRY_TOLERANCE
        ) | np.any(
            (edge_length <= _GEOMETRY_TOLERANCE)
            | (face_distance <= _GEOMETRY_TOLERANCE),
            axis=1,
        )
        result[invalid] = 0.0
        return result

    def AspectRatio(self) -> np.ndarray:
        lengths = np.linalg.norm(
            np.roll(self.cells, -1, axis=1) - self.cells, axis=2
        )
        minimum = np.min(lengths, axis=1)
        if np.any(minimum <= _GEOMETRY_TOLERANCE):
            raise ValueError("Cannot compute aspect ratio for zero-length edges")
        return np.max(lengths, axis=1) / minimum

    def ParallelDeviations(self) -> np.ndarray:
        edges = np.roll(self.cells, -1, axis=1) - self.cells
        lengths = np.linalg.norm(edges, axis=2)
        if np.any(lengths <= _GEOMETRY_TOLERANCE):
            raise ValueError("Cannot compute parallel deviation for zero-length edges")
        units = edges / lengths[..., None]
        first = np.abs(np.sum(units[:, 0] * units[:, 2], axis=1))
        second = np.abs(np.sum(units[:, 1] * units[:, 3], axis=1))
        deviations = np.degrees(
            np.arccos(np.clip(np.stack((first, second), axis=1), 0.0, 1.0))
        )
        return np.max(deviations, axis=1)

    def SelfIntersectingCells(self) -> np.ndarray:
        if self._self_intersections is None:
            cells = self.cells
            self._self_intersections = _segments_intersect_many(
                cells[:, 0], cells[:, 1], cells[:, 2], cells[:, 3]
            ) | _segments_intersect_many(
                cells[:, 1], cells[:, 2], cells[:, 3], cells[:, 0]
            )
        return self._self_intersections.copy()

    def ConcaveCells(self) -> np.ndarray:
        if self._concave_cells is None:
            jacobians = self.NodeJacobian()
            nonzero = np.all(np.abs(jacobians) > _GEOMETRY_TOLERANCE, axis=1)
            mixed_sign = np.any(jacobians > 0, axis=1) & np.any(
                jacobians < 0, axis=1
            )
            self._concave_cells = (
                ~self.SelfIntersectingCells() & nonzero & mixed_sign
            )
        return self._concave_cells.copy()

    def NumConcaveVertex(self) -> np.ndarray:
        """Return the number of inward corners for every simple cell."""
        if self._concave_vertex_count is None:
            jacobians = self.NodeJacobian()
            orientation = np.sign(self.SignedArea())[:, None]
            counts = np.sum(
                jacobians * orientation < -_GEOMETRY_TOLERANCE,
                axis=1,
            )
            simple = ~self.SelfIntersectingCells() & ~self.DegenerateCells()
            self._concave_vertex_count = np.where(simple, counts, 0)
        return self._concave_vertex_count.copy()

    def FlippingCells(self) -> np.ndarray:
        if self._flipping_cells is None:
            self._flipping_cells = (
                (self.SignedArea() <= _GEOMETRY_TOLERANCE)
                | self.SelfIntersectingCells()
                | (self.NumConcaveVertex() > 0)
            )
        return self._flipping_cells.copy()

    def DegenerateCells(self) -> np.ndarray:
        if self._degenerate_cells is None:
            self._degenerate_cells = self.Area() <= _GEOMETRY_TOLERANCE
        return self._degenerate_cells.copy()

    def ValidCells(self) -> np.ndarray:
        return ~self.FlippingCells()

    def Num_flipping_cells(self) -> int:
        return int(np.sum(self.FlippingCells()))

    def Num_concave_cells(self) -> int:
        return int(np.sum(self.ConcaveCells()))

    def Num_self_intersecting_cells(self) -> int:
        return int(np.sum(self.SelfIntersectingCells()))

    def IntersectingCells(self) -> np.ndarray:
        """Find overlapping non-neighbor cells with a spatial-hash broad phase."""
        if self._intersecting_cells is not None:
            return self._intersecting_cells.copy()
        result = self.SelfIntersectingCells()
        minimum = self.cells.min(axis=1)
        maximum = self.cells.max(axis=1)
        pairs = _spatial_aabb_pairs(minimum, maximum)
        if len(pairs):
            first, second = pairs[:, 0], pairs[:, 1]
            shares_node = np.any(
                self.quads[first, :, None] == self.quads[second, None, :],
                axis=(1, 2),
            )
            candidates = pairs[~shares_node]
            for start in range(0, len(candidates), 4096):
                chunk = candidates[start : start + 4096]
                overlap = _polygons_overlap_many(
                    self.cells[chunk[:, 0]], self.cells[chunk[:, 1]]
                )
                overlapping = chunk[overlap]
                if len(overlapping):
                    result[overlapping.reshape(-1)] = True
        self._intersecting_cells = result
        return result.copy()

    def Num_intersecting_cells(self) -> int:
        return int(np.sum(self.IntersectingCells()))

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(points_shape={self.points.shape}, "
            f"quads_shape={self.quads.shape})"
        )

    def __len__(self) -> int:
        return len(self.quads)


def test_public_quality_functions() -> dict:
    """Documentation string."""
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    rectangle = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=float)
    parallelogram = np.array([[0, 0], [2, 0], [2.5, 1], [0.5, 1]], dtype=float)
    concave = np.array([[0, 0], [1, 0], [0.4, 0.4], [0, 1]], dtype=float)
    bow_tie = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=float)

    square_det, _ = ComputeQuadJacobian(square)
    results = {
        "square_area": SignedPolygonArea(square),
        "square_node_jacobian": ComputeNodeJacobian(square),
        "square_center_jacobian": square_det,
        "square_angles": ComputeQuadAngle(square),
        "square_skewness": ComputeQuadSkewness(square),
        "square_orthogonality": ComputeQuadOrthogonality(square),
        "square_aspect_ratio": ComputeQuadAspectRatio(square),
        "square_parallel_deviation": ComputeQuadParallelDeviation(square),
        "rectangle_aspect_ratio": ComputeQuadAspectRatio(rectangle),
        "parallelogram_skewness": ComputeQuadSkewness(parallelogram),
        "parallelogram_orthogonality": ComputeQuadOrthogonality(parallelogram),
        "parallelogram_parallel_deviation": ComputeQuadParallelDeviation(
            parallelogram,
        ),
        "concave_angles": ComputeQuadAngle(concave),
        "concave": CheckConcave(concave),
        "self_intersection": CheckSelfIntersection(bow_tie),
        "segment_intersection": CheckLineSegmentsIntersect(
            [0, 0],
            [1, 1],
            [0, 1],
            [1, 0],
        ),
    }
    assert np.isclose(results["square_area"], 1.0)
    assert np.allclose(results["square_node_jacobian"], 1.0)
    assert np.isclose(results["square_center_jacobian"], 0.25)
    assert np.allclose(results["square_angles"], 90.0)
    assert np.isclose(results["square_skewness"], 0.0)
    assert np.isclose(results["square_orthogonality"], 1.0)
    assert np.isclose(results["square_aspect_ratio"], 1.0)
    assert np.isclose(results["square_parallel_deviation"], 0.0)
    assert np.isclose(results["rectangle_aspect_ratio"], 2.0)
    assert results["parallelogram_skewness"] > 0
    assert 0 < results["parallelogram_orthogonality"] < 1
    assert np.isclose(
        results["parallelogram_parallel_deviation"],
        0.0,
        atol=1e-6,
    )
    assert np.max(results["concave_angles"]) > 180
    assert results["concave"] and results["self_intersection"]
    assert results["segment_intersection"]

    try:
        ComputeQuadAspectRatio([[0, 0], [0, 0], [1, 1], [0, 1]])
    except ValueError:
        results["degenerate_rejected"] = True
    else:
        raise AssertionError("""Documentation string.""")
    return results


def test_quad_mesh_quality_evaluator() -> dict:
    """Documentation string."""
    from src.data.geometry import boundary_generator_2d as boundary
    from src.data.mesh import topology as quad_mesh
    from scripts.pde.vectorized_winslow_2d import solve_winslow_2d_vectorized

    Nx = Ny = 20
    curves = boundary.GenerateRectangleWithTopSemiEllipseBoundary(
        length=1.0,
        height=1.0,
        semi_minor_axis=0.25,
    )
    points = solve_winslow_2d_vectorized(
        curves,
        nx=Nx,
        ny=Ny,
        max_iter=100,
        tolerance=1e-8,
    )
    quads = np.asarray(quad_mesh.GenerateQuadMeshIndex(Ny=Ny, Nx=Nx))
    evaluator = QuadMeshQualityEvaluator(points, quads)

    det_jacobian, jacobian = evaluator.Jacobian()
    results = {
        "points_shape": points.shape,
        "quads_shape": quads.shape,
        "cell_count": len(evaluator),
        "area_min_mean_max": (
            evaluator.Area().min(),
            evaluator.Area().mean(),
            evaluator.Area().max(),
        ),
        "angle_min_mean_max": (
            evaluator.MinAngle().min(),
            evaluator.Angle().mean(),
            evaluator.MaxAngle().max(),
        ),
        "skewness_min_mean_max": (
            evaluator.Skewness().min(),
            evaluator.Skewness().mean(),
            evaluator.Skewness().max(),
        ),
        "orthogonality_min_mean_max": (
            evaluator.Orthogonality().min(),
            evaluator.Orthogonality().mean(),
            evaluator.Orthogonality().max(),
        ),
        "aspect_ratio_min_mean_max": (
            evaluator.AspectRatio().min(),
            evaluator.AspectRatio().mean(),
            evaluator.AspectRatio().max(),
        ),
        "parallel_deviation_min_mean_max": (
            evaluator.ParallelDeviations().min(),
            evaluator.ParallelDeviations().mean(),
            evaluator.ParallelDeviations().max(),
        ),
        "center_jacobian_min_max": (
            det_jacobian.min(),
            det_jacobian.max(),
        ),
        "jacobian_shape": jacobian.shape,
        "concave_vertex_max": evaluator.NumConcaveVertex().max(),
        "flipping_cells": evaluator.Num_flipping_cells(),
        "concave_cells": evaluator.Num_concave_cells(),
        "self_intersecting_cells": evaluator.Num_self_intersecting_cells(),
        "intersecting_cells": evaluator.Num_intersecting_cells(),
        "valid_cells": int(np.sum(evaluator.ValidCells())),
        "degenerate_cells": int(np.sum(evaluator.DegenerateCells())),
        "description": str(evaluator),
    }
    assert results["points_shape"] == (Nx * Ny, 2)
    assert results["quads_shape"] == ((Nx - 1) * (Ny - 1), 4)
    assert results["cell_count"] == (Nx - 1) * (Ny - 1)
    assert np.isfinite(points).all()
    assert np.all(det_jacobian > 0)
    assert results["flipping_cells"] == 0
    assert results["concave_cells"] == 0
    assert results["self_intersecting_cells"] == 0
    assert results["intersecting_cells"] == 0
    assert results["valid_cells"] == len(evaluator)
    assert results["degenerate_cells"] == 0
    return results


# ============================ 3D hexahedral metrics ============================
def evaluate_hexahedral_grid_quality(grid, tolerance=1e-12, metrics=None):
    """Return per-cell industrial metrics for ``grid[nz,ny,nx,3]``.

    ``min_angle`` and ``max_angle`` are the extrema of the three pairwise
    interior edge angles at all eight hexahedron corners (24 angles per cell).
    An orthogonal hexahedron therefore has both values equal to 90 degrees.
    ``skewness`` is the largest normalized deviation from 90 degrees and
    ``orthogonality`` is one minus the largest absolute adjacent-edge cosine.
    The result also includes minimum corner scaled Jacobian, centre
    determinant/condition, edge aspect ratio, volume and validity flags.
    """
    grid = np.asarray(grid, dtype=float)
    if (
        grid.ndim != 4
        or grid.shape[-1] != 3
        or min(grid.shape[:3]) < 2
        or not np.isfinite(grid).all()
    ):
        raise ValueError("grid must be finite with shape (Nz,Ny,Nx,3)")
    selected = set(metric_names(metrics, HEX_METRICS))
    if not selected:
        return {}
    result = {name: [] for name in HEX_METRICS if name in selected}
    need_scaled = bool(selected & {"minimum_scaled_jacobian", "flipped_cell"})
    need_lengths = bool(selected & {"edge_aspect_ratio", "degenerate_cell"})
    need_angles = bool(selected & {"max_angle", "min_angle", "skewness"})
    need_cosines = need_angles or "orthogonality" in selected
    need_center = bool(
        selected
        & {
            "center_jacobian",
            "center_condition_number",
            "cell_volume",
            "degenerate_cell",
        }
    )
    need_det = bool(selected & {"center_jacobian", "cell_volume", "degenerate_cell"})
    for k in range(grid.shape[0] - 1):
        for j in range(grid.shape[1] - 1):
            for i in range(grid.shape[2] - 1):
                c = grid[k : k + 2, j : j + 2, i : i + 2]
                scaled = []
                lengths = []
                corner_angles = []
                adjacent_cosines = []
                if need_scaled or need_cosines:
                    for ck in (0, 1):
                        for cj in (0, 1):
                            for ci in (0, 1):
                                if need_scaled:
                                    eu = c[ck, cj, 1] - c[ck, cj, 0]
                                    ev = c[ck, 1, ci] - c[ck, 0, ci]
                                    ew = c[1, cj, ci] - c[0, cj, ci]
                                    den = (
                                        np.linalg.norm(eu)
                                        * np.linalg.norm(ev)
                                        * np.linalg.norm(ew)
                                    )
                                    scaled.append(
                                        np.linalg.det(np.column_stack((eu, ev, ew)))
                                        / den
                                        if den > tolerance
                                        else -np.inf
                                    )
                                if need_cosines:
                                    outgoing = (
                                        c[ck, cj, 1 - ci] - c[ck, cj, ci],
                                        c[ck, 1 - cj, ci] - c[ck, cj, ci],
                                        c[1 - ck, cj, ci] - c[ck, cj, ci],
                                    )
                                    for first, second in (
                                        (outgoing[0], outgoing[1]),
                                        (outgoing[0], outgoing[2]),
                                        (outgoing[1], outgoing[2]),
                                    ):
                                        denominator = np.linalg.norm(
                                            first
                                        ) * np.linalg.norm(second)
                                        if denominator <= tolerance:
                                            cosine = np.nan
                                            angle = np.nan
                                        else:
                                            cosine = float(
                                                np.clip(
                                                    np.dot(first, second) / denominator,
                                                    -1.0,
                                                    1.0,
                                                )
                                            )
                                            angle = (
                                                float(np.degrees(np.arccos(cosine)))
                                                if need_angles
                                                else np.nan
                                            )
                                        adjacent_cosines.append(cosine)
                                        corner_angles.append(angle)
                if need_lengths:
                    for a, b in (
                        (c[:, :, 1], c[:, :, 0]),
                        (c[:, 1, :], c[:, 0, :]),
                        (c[1, :, :], c[0, :, :]),
                    ):
                        lengths.extend(np.linalg.norm(a - b, axis=-1).ravel())
                if need_center:
                    eu = np.mean(c[:, :, 1] - c[:, :, 0], axis=(0, 1))
                    ev = np.mean(c[:, 1, :] - c[:, 0, :], axis=(0, 1))
                    ew = np.mean(c[1, :, :] - c[0, :, :], axis=(0, 1))
                    jac = np.column_stack((eu, ev, ew))
                if need_det:
                    det = float(np.linalg.det(jac))
                if need_scaled:
                    sj = float(min(scaled))
                if need_angles:
                    finite_angles = np.asarray(corner_angles, dtype=float)
                    finite_angles = finite_angles[np.isfinite(finite_angles)]
                if "orthogonality" in selected:
                    finite_cosines = np.asarray(adjacent_cosines, dtype=float)
                    finite_cosines = finite_cosines[np.isfinite(finite_cosines)]
                # Lambdas defer every metric-specific reduction until selected.
                calculations = {
                    "minimum_scaled_jacobian": lambda: sj,
                    "center_jacobian": lambda: det,
                    "center_condition_number": lambda: float(np.linalg.cond(jac)),
                    "edge_aspect_ratio": lambda: float(
                        max(lengths) / max(min(lengths), tolerance)
                    ),
                    "max_angle": lambda: (
                        float(finite_angles.max()) if finite_angles.size else np.nan
                    ),
                    "min_angle": lambda: (
                        float(finite_angles.min()) if finite_angles.size else np.nan
                    ),
                    "skewness": lambda: (
                        float(np.max(np.abs(finite_angles - 90.0)) / 90.0)
                        if finite_angles.size
                        else np.nan
                    ),
                    "orthogonality": lambda: (
                        float(1.0 - np.max(np.abs(finite_cosines)))
                        if finite_cosines.size
                        else np.nan
                    ),
                    "cell_volume": lambda: abs(det),
                    "flipped_cell": lambda: float(sj <= 0),
                    "degenerate_cell": lambda: float(
                        abs(det) <= tolerance or min(lengths) <= tolerance
                    ),
                }
                for name in result:
                    result[name].append(calculations[name]())
    return {key: np.asarray(value) for key, value in result.items()}


if __name__ == "__main__":
    public_results = test_public_quality_functions()
    print("Public quality function tests passed:")
    for name, value in public_results.items():
        print(f"  {name}: {value}")

    evaluator_results = test_quad_mesh_quality_evaluator()
    print("QuadMeshQualityEvaluator tests passed:")
    for name, value in evaluator_results.items():
        print(f"  {name}: {value}")

