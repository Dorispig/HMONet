"""Low-level geometric predicates used by mesh-quality evaluators."""

from .quality import (
    CheckConcave,
    CheckLineSegmentsIntersect,
    CheckSelfIntersection,
    SignedPolygonArea,
)

signed_polygon_area = SignedPolygonArea
segments_intersect = CheckLineSegmentsIntersect
polygon_self_intersects = CheckSelfIntersection
polygon_is_concave = CheckConcave

__all__ = [
    "signed_polygon_area",
    "segments_intersect",
    "polygon_self_intersects",
    "polygon_is_concave",
]
