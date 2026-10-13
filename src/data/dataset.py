"""Load only boundary points, target meshes and optional parametric boundaries."""

import json
import copy
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from src.data.mesh.topology import GenerateQuadMeshBoundaryIndex
from src.data.geometry.parametric_curve_2d import BoundaryCurveSet2D
from src.data.geometry.parametric_geometry_3d import BoundarySurfaceSet3D


NUMERICAL_DATA_FILENAME = "dataset.npy"
BOUNDARY_GEOMETRY_FILENAME = "boundary_geometry.npy"
PAIRED_FORMAT = "parametric_2d_pair_v1"


def load_paired_records(path, role):
    """Validate file identity and row order without retaining them in the dataset."""
    records = np.load(Path(path), allow_pickle=True)
    if records.ndim != 1 or len(records) == 0:
        raise ValueError("A paired dataset file must be a nonempty 1-D object array")
    if not isinstance(records[0], dict):
        raise ValueError("Dataset records must be dictionaries")
    metadata = records[0].get("metadata", {})
    if metadata.get("format") != PAIRED_FORMAT or metadata.get("data_role") != role:
        raise ValueError(f"Invalid {role} dataset header")
    if metadata.get("dimension") != 2 or metadata.get("size") != len(records):
        raise ValueError("Dataset dimension or record count does not match the header")
    identity = metadata.get("dataset_id")
    if not isinstance(identity, str) or not identity:
        raise ValueError("Dataset header must contain a dataset_id")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Sample {index} is not a dictionary")
        if record.get("sample_id") != index:
            raise ValueError(f"sample_id mismatch at index {index}")
        if record.get("dataset_id") != identity:
            raise ValueError(f"dataset_id mismatch at index {index}")
    return records, metadata


def read_dataset_metadata(root):
    """Read generation settings for catalogs, separately from training samples."""
    root = Path(root)
    if (root / BOUNDARY_GEOMETRY_FILENAME).is_file():
        if not (root / NUMERICAL_DATA_FILENAME).is_file():
            raise FileNotFoundError(root / NUMERICAL_DATA_FILENAME)
        _, metadata = load_paired_records(
            root / BOUNDARY_GEOMETRY_FILENAME, "geometry"
        )
        return metadata
    return json.loads((root / "metadata.json").read_text(encoding="utf-8"))


def read_sample_parameters(root, index):
    """Read geometric parameter values for evaluation labels, without caching them."""
    root = Path(root)
    if (root / BOUNDARY_GEOMETRY_FILENAME).is_file():
        records, _ = load_paired_records(
            root / BOUNDARY_GEOMETRY_FILENAME, "geometry"
        )
    else:
        data = json.loads((root / "geometry.json").read_text(encoding="utf-8"))
        records = data["samples"]
    return dict(records[index]["parameters"])


class ParametricDataset(Dataset):
    """Return two numeric fields and optionally the parametric boundary class.

    root identifies the dataset directory. load_boundary controls whether
    parameter boundaries are loaded. grid_shape is the saved logical grid size.
    boundary_points and mesh_points hold the numeric arrays. When load_boundary
    is True, boundary holds the restored parametric boundary objects.
    """

    dimension = 2

    def __init__(self, root, load_boundary=True):
        if not isinstance(load_boundary, bool):
            raise TypeError("load_boundary must be True or False")
        requested = Path(root)
        self.root = requested.parent if requested.is_file() else requested
        self.load_boundary = load_boundary
        self.boundary_points, self.mesh_points, metadata = self._read_mesh_data()
        shape = metadata["grid_shape"]
        if (
            metadata["dimension"] != self.dimension
            or len(shape) != self.dimension
            or any(int(value) != value or value < 2 for value in shape)
        ):
            raise ValueError("Invalid dataset dimension or grid_shape")
        self.grid_shape = tuple(int(value) for value in shape)
        size = metadata["size"]
        point_count = int(np.prod(self.grid_shape))
        boundary_count = point_count - int(np.prod(np.asarray(self.grid_shape) - 2))
        if (
            self.boundary_points.shape != (size, boundary_count, self.dimension)
            or self.mesh_points.shape != (size, point_count, self.dimension)
        ):
            raise ValueError("Numeric array shapes do not match the dataset grid")
        if (
            not np.isfinite(self.boundary_points).all()
            or not np.isfinite(self.mesh_points).all()
        ):
            raise ValueError("Non-finite coordinates in dataset")
        if self.dimension == 2:
            nx, ny = self.grid_shape
            indices = GenerateQuadMeshBoundaryIndex(ny, nx)
            if not np.array_equal(self.mesh_points[:, indices], self.boundary_points):
                raise ValueError("Stored mesh boundaries do not match boundary_points")
        if self.load_boundary:
            self.boundary = self._read_boundaries(metadata)

    def _read_mesh_data(self):
        """Extract numeric arrays; discard IDs and configuration after validation."""
        records, metadata = load_paired_records(
            self.root / NUMERICAL_DATA_FILENAME, "mesh"
        )
        points = np.stack([record["boundary_points"] for record in records])
        meshes = np.stack([record["mesh_points"] for record in records])
        return points, meshes, metadata

    def _read_boundaries(self, metadata):
        """Match the two files and restore only the boundary classes."""
        records, geometry_metadata = load_paired_records(
            self.root / BOUNDARY_GEOMETRY_FILENAME, "geometry"
        )
        for field in ("dataset_id", "size", "dimension", "grid_shape"):
            if geometry_metadata.get(field) != metadata.get(field):
                raise ValueError(f"Geometry/numerical file mismatch: {field}")
        return [
            BoundaryCurveSet2D.from_dict(record["boundary"])
            for record in records
        ]

    def __len__(self):
        return len(self.boundary_points)

    def __getitem__(self, index):
        """Return numeric fields and include boundary only when loading is enabled."""
        sample = {
            "boundary_points": self.boundary_points[index],
            "mesh_points": self.mesh_points[index],
        }
        if self.load_boundary:
            sample["boundary"] = self.boundary[index]
        return sample


class ParametricDataset2D(ParametricDataset):
    """Read the paired 2-D NumPy files."""

    dimension = 2


class ParametricDataset3D(ParametricDataset):
    """Read required arrays and surfaces from the existing 3-D file layout."""

    dimension = 3

    def _read_mesh_data(self):
        metadata = json.loads(
            (self.root / "metadata.json").read_text(encoding="utf-8")
        )
        points = np.load(self.root / "boundary_samples.npy")
        meshes = np.load(self.root / "pde_mesh.npy").reshape(len(points), -1, 3)
        return points, meshes, metadata

    def _read_boundaries(self, metadata):
        data = json.loads((self.root / "geometry.json").read_text(encoding="utf-8"))
        records = data["samples"]
        if len(records) != metadata["size"]:
            raise ValueError("Geometry and numerical sample counts differ")
        for index, record in enumerate(records):
            if record["sample_id"] != index:
                raise ValueError(f"sample_id mismatch at index {index}")
        return [
            BoundarySurfaceSet3D.from_dict(record["boundary"])
            for record in records
        ]


def uniformly_spaced_subset(dataset, sample_count):
    """Return an in-memory, order-preserving subset spanning the full dataset.

    No dataset files are created.  For parameter-ordered generated datasets,
    the rounded linspace indices retain both endpoints and cover the complete
    parameter interval approximately uniformly.
    """
    sample_count = int(sample_count)
    total = len(dataset)
    if sample_count < 1 or sample_count > total:
        raise ValueError(
            f"sample_count must be between 1 and {total}, received {sample_count}"
        )
    indices = np.rint(np.linspace(0, total - 1, sample_count)).astype(np.int64)
    if len(np.unique(indices)) != sample_count:
        raise RuntimeError("Uniform subset selection produced duplicate indices")
    subset = copy.copy(dataset)
    subset.boundary_points = (
        dataset.boundary_points
        if sample_count == total
        else dataset.boundary_points[indices]
    )
    subset.mesh_points = (
        dataset.mesh_points if sample_count == total else dataset.mesh_points[indices]
    )
    if dataset.load_boundary:
        subset.boundary = (
            dataset.boundary
            if sample_count == total
            else [dataset.boundary[index] for index in indices]
        )
    subset.subset_indices = indices
    subset.original_size = total
    return subset
