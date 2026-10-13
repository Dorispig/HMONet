"""Orchestrate selected-method evaluation without sharing model result labels."""

from pathlib import Path
import shutil
import time

import numpy as np

from src.data.mesh.sampling import boundary_indices, logical_grid, sample_boundary
from src.evaluation.artifacts import (
    array_digest,
    file_digest,
    digest,
    read_json,
    write_json,
    write_csv,
    manifest_valid,
    save_manifest,
    clear_managed_files,
    boundary_error,
    save_boundary_errors,
    save_timing_tables,
)
from src.evaluation.configuration import jobs, evaluation_options, resolve_path
from src.evaluation.generation import (
    generate,
    implementation_version,
    solve_pde,
    execute_prediction_batches,
)
from src.evaluation.inference import find_checkpoint, postprocess
from src.evaluation.analysis import (
    save_method,
    load_quality,
    save_comparison,
    save_point_errors,
)
from src.evaluation.local_views import run_local_views
from src.utils.quality_selection import quality_settings


def selected_mapping(method, mapping, options):
    """Keep PI-HMONet's mapping independent from supervised HMONet."""
    if method == "PI-HMONet":
        return options["pihmonet_mapping_mode"]
    if method == "Attention-PI-HMONet":
        return options["attentionpihmonet_mapping_mode"]
    return mapping


def mapping_combinations(options):
    """Expand requested hard-constraint variants without mixing checkpoint types."""

    def variants(key, method):
        value = options[key]
        if value == "both":
            return (
                ("current", "coons_residual")
                if method in options["methods"]
                else ("current",)
            )
        return (value,)

    return [
        (
            h,
            {
                **options,
                "hmonet_mapping_mode": h,
                "pihmonet_mapping_mode": pi,
                "attentionpihmonet_mapping_mode": attention,
            },
        )
        for h in variants("hmonet_mapping_mode", "HMONet")
        for pi in variants("pihmonet_mapping_mode", "PI-HMONet")
        for attention in variants(
            "attentionpihmonet_mapping_mode", "Attention-PI-HMONet"
        )
    ]


def method_directory(root, method, mapping, projection=None):
    """Keep weights' internal mapping and external boundary processing separate."""
    path = root / method
    if method.startswith("HMONet"):
        path = root / "HMONet" / mapping
    if method == "PI-HMONet":
        path /= mapping
    if method == "Attention-PI-HMONet":
        path /= mapping
    if projection is not None:
        path /= projection
    if method == "HMONet_PDE_refined":
        path /= "pde_refined"
    return path


def comparison_directory(root, mapping, options, names):
    tags = {"inverse_distance": "idw", "direct": "direct", "none": "none"}
    parent = root / "comparisons"
    if "HMONet" in names:
        parent /= mapping
        parent /= f"hmonet_{tags[options['hmonet_boundary_projection']]}__meshonet_{tags[options['meshonet_boundary_projection']]}"
    elif "MeshONet" in names:
        parent /= f"meshonet_{tags[options['meshonet_boundary_projection']]}"
    if "PI-MeshONet" in names:
        parent /= f"pimeshonet_{tags[options['pimeshonet_boundary_projection']]}"
    if "PI-HMONet" in names:
        parent /= "pihmonet_" + options["pihmonet_mapping_mode"]
        parent /= f"pihmonet_{tags[options['pihmonet_boundary_projection']]}"
    if "Attention-PI-HMONet" in names:
        parent /= "attention_pihmonet_" + options[
            "attentionpihmonet_mapping_mode"
        ]
    slugs = {
        "TFI": "tfi",
        "PDE_Winslow": "pde",
        "MeshONet": "meshonet",
        "PI-MeshONet": "pimeshonet",
        "PI-HMONet": "pihmonet",
        "Attention-PI-HMONet": "attention_pihmonet",
        "HMONet": "hmonet",
        "HMONet_PDE_refined": "hmonet_pde_refined",
    }
    return parent / ("01_" + "_".join(slugs[name] for name in names))


def analyze_method(
    directory,
    method,
    points,
    shape,
    signature,
    exact,
    options,
    raw=False,
    clone_from=None,
):
    """Reuse complete analysis; repair missing files without repeating inference."""
    plot_settings = {
        key: options[key]
        for key in ("save_plots", "save_png", "save_pdf", "save_distribution_plots")
    }
    signature = digest(
        {
            "generation": signature,
            "quality_revision": "endpoint-contact-v2",
            "plots": plot_settings,
            "quality": quality_settings(options, len(shape)),
            "stage": "raw" if raw else "postprocessed",
        }
    )
    reusable = (
        options["reuse_analysis_results"]
        and not options["force_reanalyze"]
        and manifest_valid(directory, signature)
    )
    if reusable:
        print(
            f"  reuse analysis {method} ({'raw' if raw else 'processed'})", flush=True
        )
        return load_quality(directory / "grid_quality", points, shape, options)
    if clone_from is not None and Path(clone_from).resolve() != Path(directory).resolve():
        print(f"  clone identical analysis {method} (raw -> processed)", flush=True)
        clear_managed_files(directory)
        Path(directory).mkdir(parents=True, exist_ok=True)
        shutil.copytree(clone_from, directory, dirs_exist_ok=True)
        raw_metadata = Path(directory) / "raw_output_metadata.json"
        if raw_metadata.is_file():
            raw_metadata.unlink()
        save_boundary_errors(
            directory,
            points,
            exact,
            boundary_indices(shape),
            method,
            "postprocessed",
            {
                **options,
                "save_plots": options["save_plots"]
                and bool(
                    quality_settings(options, len(shape))["metrics"]
                    or quality_settings(options, len(shape))["surface_metrics"]
                ),
            },
        )
        write_json(
            Path(directory) / "mesh_metadata.json",
            {
                "method": method,
                "grid_shape": shape,
                "stage": "postprocessed",
                "signature": signature,
                "analysis_cloned_from": "raw",
            },
        )
        save_manifest(directory, signature)
        return load_quality(Path(directory) / "grid_quality", points, shape, options)
    print(f"  analyze {method} ({'raw' if raw else 'processed'})", flush=True)
    clear_managed_files(directory)
    quality = save_method(directory, method, points, shape, options)
    if method in ("HMONet", "MeshONet", "PI-MeshONet", "PI-HMONet", "Attention-PI-HMONet"):
        save_boundary_errors(
            directory,
            points,
            exact,
            boundary_indices(shape),
            method,
            "raw" if raw else "postprocessed",
            {
                **options,
                "save_plots": options["save_plots"]
                and bool(
                    quality_settings(options, len(shape))["metrics"]
                    or quality_settings(options, len(shape))["surface_metrics"]
                ),
            },
        )
    write_json(
        directory / "mesh_metadata.json",
        {
            "method": method,
            "grid_shape": shape,
            "stage": "raw" if raw else "postprocessed",
            "signature": signature,
        },
    )
    if raw:
        write_json(
            directory / "raw_output_metadata.json",
            {
                "method": method,
                "boundary_projection": "none",
                "comparison_eligible": False,
                "inference_reused_for_projected_output": True,
                "grid_shape": shape,
                "signature": signature,
            },
        )
    save_manifest(directory, signature)
    return quality


def evaluate_object(
    name,
    case,
    tag,
    shape,
    geometry,
    parameters,
    options,
    args,
    mapping,
    version,
    models,
):
    started = time.perf_counter()
    output = resolve_path(options["output_root"])
    dimension = len(shape)
    if output.name != f"{dimension}D":
        output /= f"{dimension}D"
    root = output / name / "x".join(map(str, shape)) / tag
    root.mkdir(parents=True, exist_ok=True)
    exact = sample_boundary(geometry, shape)
    ids = boundary_indices(shape)
    selected = list(options["methods"])
    checkpoints = {}
    skipped = []
    for method in selected:
        if method not in ("HMONet", "MeshONet", "PI-MeshONet", "PI-HMONet", "Attention-PI-HMONet"):
            continue
        path = find_checkpoint(
            dimension,
            name,
            method,
            selected_mapping(method, mapping, options),
            options,
            case.get("split"),
        )
        if path is None:
            if not options["skip_missing_checkpoints"]:
                raise FileNotFoundError(
                    f"No matching {method}/{mapping} checkpoint for {name}"
                )
            skipped.append(
                {"method": method, "reason": "matching checkpoint not found"}
            )
            print(
                f"  skip {method}: no matching checkpoint for {name}/{mapping}",
                flush=True,
            )
        else:
            checkpoints[method] = path
    selected = [m for m in selected if not any(s["method"] == m for s in skipped)]
    if not selected:
        raise RuntimeError("None of the requested methods is available")
    meshes, raw_meshes, timings, signatures, directories = {}, {}, {}, {}, {}
    identical_raw_analyses = {}
    error_records = []
    for method in selected:
        cache = root / ".cache" / method
        if method in ("HMONet", "PI-HMONet", "Attention-PI-HMONet"):
            cache /= selected_mapping(method, mapping, options)
        points, timing, source_signature = generate(
            cache,
            method,
            geometry,
            shape,
            selected_mapping(method, mapping, options),
            options,
            checkpoints.get(method),
            args.train_config,
            version,
            models,
        )
        if method in ("HMONet", "MeshONet", "PI-MeshONet", "PI-HMONet", "Attention-PI-HMONet"):
            raw_meshes[method] = points.copy()
            raw_summary, _ = boundary_error(points, exact, ids)
            error_records.append(dict(method=method, stage="raw", **raw_summary))
            raw_directory = method_directory(
                root, method, selected_mapping(method, mapping, options), "raw"
            )
            if options["generation_only"]:
                raw_directory.mkdir(parents=True, exist_ok=True)
                np.save(raw_directory / "mesh.npy", points)
                save_boundary_errors(
                    raw_directory,
                    points,
                    exact,
                    ids,
                    method,
                    "raw",
                    {**options, "save_plots": False},
                )
            else:
                analyze_method(
                    raw_directory,
                    method,
                    points,
                    shape,
                    source_signature,
                    exact,
                    options,
                    raw=True,
                )
            mode = options[f"{method.lower().replace('-', '')}_boundary_projection"]
            if mode == "none":
                # No correction or copy is performed; report zero postprocessing
                # time instead of timing an otherwise unnecessary array copy.
                timing["boundary_projection_seconds"] = 0.0
                identical_raw_analyses[method] = raw_directory
            else:
                projection_started = time.perf_counter()
                points = postprocess(
                    points, exact, ids, mode, options["inverse_distance_power"]
                )
                timing["boundary_projection_seconds"] = (
                    time.perf_counter() - projection_started
                )
            timing["total_model_seconds"] = (
                timing["network_inference_seconds"]
                + timing["boundary_projection_seconds"]
            )
            post_summary, _ = boundary_error(points, exact, ids)
            error_records.append(
                dict(method=method, stage="postprocessed", **post_summary)
            )
            signatures[method] = digest(
                {
                    "raw": source_signature,
                    "projection": mode,
                    "power": options["inverse_distance_power"],
                }
            )
            directories[method] = method_directory(
                root, method, selected_mapping(method, mapping, options), mode
            )
        else:
            signatures[method] = source_signature
            directories[method] = method_directory(root, method, mapping)
        meshes[method] = points
        timings[method] = timing
        network_seconds = timing.get(
            "network_inference_seconds", timing.get("total_seconds")
        )
        print(
            f"  {method}: generation={network_seconds:.6f}s, "
            f"postprocess={timing.get('boundary_projection_seconds', 0):.6f}s, "
            f"reused={timing['cache_hit']}",
            flush=True,
        )
    if options["hmonet_pde_iterations"] and "HMONet" in meshes:
        method = "HMONet_PDE_refined"
        refined_options = {
            **options,
            "winslow_max_iter": options["hmonet_pde_iterations"],
            "require_pde_convergence": False,
        }
        refine_sig = digest(
            {
                "initial": signatures["HMONet"],
                "iterations": options["hmonet_pde_iterations"],
                "tolerance": options["winslow_tolerance"],
            }
        )
        cache = root / ".cache/HMONet" / mapping / "refinement"
        meta = read_json(cache / "metadata.json", {})
        if (
            not options["force_regenerate"]
            and options["reuse_generated_results"]
            and meta.get("signature") == refine_sig
            and (cache / "mesh.npy").is_file()
            and meta.get("file_sha256") == file_digest(cache / "mesh.npy")
        ):
            meshes[method] = np.load(cache / "mesh.npy")
            timings[method] = {**meta["timing"], "cache_hit": True}
        else:
            if options["analysis_only"]:
                raise RuntimeError(
                    "analysis-only requires matching HMONet PDE refinement cache"
                )
            start = time.perf_counter()
            meshes[method], info = solve_pde(
                geometry,
                shape,
                refined_options,
                meshes["HMONet"],
                options["hmonet_pde_iterations"],
            )
            timings[method] = {
                "total_seconds": time.perf_counter() - start,
                "pde": info,
                "cache_hit": False,
            }
            cache.mkdir(parents=True, exist_ok=True)
            np.save(cache / "mesh.npy", meshes[method])
            write_json(
                cache / "metadata.json",
                {
                    "signature": refine_sig,
                    "timing": timings[method],
                    "file_sha256": file_digest(cache / "mesh.npy"),
                },
            )
        signatures[method] = digest(
            {"input": refine_sig, "mesh": array_digest(meshes[method])}
        )
        directories[method] = method_directory(
            root, method, mapping, options["hmonet_boundary_projection"]
        )
    metadata = {
        "case": name,
        "split": case.get("split"),
        "parameter": case.get("parameter"),
        "value": parameters.get(case.get("parameter"), tag),
        "parameters": parameters,
        "grid_shape": shape,
        "nx": shape[0],
        "ny": shape[1],
        "quality_evaluation": quality_settings(options, dimension),
        "method_order": list(meshes),
        "requested_methods": options["methods"],
        "skipped_methods": skipped,
        "hmonet_mapping_mode": mapping,
        "pihmonet_mapping_mode": options["pihmonet_mapping_mode"],
        "attentionpihmonet_mapping_mode": options[
            "attentionpihmonet_mapping_mode"
        ],
        "boundary_projection": {
            "HMONet": options["hmonet_boundary_projection"],
            "MeshONet": options["meshonet_boundary_projection"],
            "PI-MeshONet": options["pimeshonet_boundary_projection"],
            "PI-HMONet": options["pihmonet_boundary_projection"],
            "Attention-PI-HMONet": options["attentionpihmonet_boundary_projection"],
        },
        "checkpoints": {m: str(p) for m, p in checkpoints.items()},
    }
    if dimension == 3:
        metadata["nz"] = shape[2]
    comparison = comparison_directory(root, mapping, options, list(meshes))
    if options["generation_only"]:
        for method, points in meshes.items():
            directories[method].mkdir(parents=True, exist_ok=True)
            np.save(directories[method] / "mesh.npy", points)
    else:
        qualities = {
            m: analyze_method(
                directories[m],
                m,
                p,
                shape,
                signatures[m],
                exact,
                options,
                clone_from=identical_raw_analyses.get(m),
            )
            for m, p in meshes.items()
        }
        analysis_signature = digest(
            {
                "sources": signatures,
                "quality_revision": "endpoint-contact-v2",
                "metadata": metadata,
                "plots": {
                    k: options[k]
                    for k in (
                        "save_plots",
                        "save_png",
                        "save_pdf",
                        "save_distribution_plots",
                    )
                },
            }
        )
        analysis_reused = (
            options["reuse_analysis_results"]
            and not options["force_reanalyze"]
            and manifest_valid(comparison, analysis_signature)
        )
        if not analysis_reused:
            clear_managed_files(comparison)
            print(f"  compare {' / '.join(meshes)}", flush=True)
            save_comparison(comparison, meshes, qualities, shape, metadata, options)
            save_point_errors(
                comparison,
                meshes,
                raw_meshes,
                shape,
                {
                    **options,
                    "save_plots": options["save_plots"]
                    and bool(
                        metadata["quality_evaluation"]["metrics"]
                        or metadata["quality_evaluation"]["surface_metrics"]
                    ),
                },
            )
        else:
            print(f"  reuse comparison {comparison.name}", flush=True)
        write_json(comparison / "boundary_error_summary.json", error_records)
        write_csv(
            comparison / "boundary_error_summary.csv",
            error_records,
            [
                "method",
                "stage",
                "boundary_point_count",
                "boundary_max_error",
                "boundary_mean_error",
                "boundary_rms_error",
            ],
        )
        save_timing_tables(comparison, timings)
        save_manifest(comparison, analysis_signature)
        run_local_views(comparison, dimension, options)
        # Root-level records describe the last invocation; each group retains
        # its own complete metadata, so different model/projection runs coexist.
    record = {
        **metadata,
        "timings": timings,
        "boundary_errors": error_records,
        "metrics": (
            {m: q["summary"] for m, q in qualities.items()}
            if not options["generation_only"]
            else {}
        ),
        "raw_none_metrics": {
            m: next(
                r for r in error_records if r["method"] == m and r["stage"] == "raw"
            )
            for m in raw_meshes
        },
        "comparison_directory": (
            str(comparison) if not options["generation_only"] else None
        ),
        "generation_only": options["generation_only"],
        "evaluation_seconds": time.perf_counter() - started,
    }
    write_json(root / "evaluation_parameters.json", {**metadata, "evaluation": options})
    write_json(
        root / "evaluation_timing.json",
        {"evaluation_seconds": record["evaluation_seconds"], "methods": timings},
    )
    write_json(root / "results.json", record)
    write_json(root / "evaluation_summary.json", [record])
    write_json(root / "boundary_error_summary.json", error_records)
    write_csv(
        root / "boundary_error_summary.csv",
        error_records,
        [
            "method",
            "stage",
            "boundary_point_count",
            "boundary_max_error",
            "boundary_mean_error",
            "boundary_rms_error",
        ],
    )
    save_timing_tables(root, timings)
    np.savez_compressed(root / "generated_meshes.npz", **meshes)
    np.savez_compressed(root / "generated_raw_model_meshes.npz", **raw_meshes)
    write_json(
        root / "generation_metadata.json", {"signatures": signatures, **metadata}
    )
    print(f"results={root}", flush=True)
    return record


def run_evaluation(dimension, args, config):
    version = implementation_version()
    models = {}
    records = []
    failures = []
    pending_jobs = list(jobs(dimension, args, config))
    requests = []
    for name, case, tag, shape, geometry, parameters in pending_jobs:
        options = evaluation_options(config, case, args)
        if options["inference_batch_size"] <= 1 or options["analysis_only"]:
            continue
        mappings = mapping_combinations(options)
        output = resolve_path(options["output_root"])
        if output.name != f"{dimension}D":
            output /= f"{dimension}D"
        root = output / name / "x".join(map(str, shape)) / tag
        for mapping, options in mappings:
            for method in options["methods"]:
                if method not in ("HMONet", "MeshONet", "PI-MeshONet", "PI-HMONet", "Attention-PI-HMONet"):
                    continue
                try:
                    checkpoint = find_checkpoint(
                        dimension,
                        name,
                        method,
                        selected_mapping(method, mapping, options),
                        options,
                        case.get("split"),
                    )
                    if checkpoint is None:
                        continue
                    cache = root / ".cache" / method
                    if method in ("HMONet", "PI-HMONet", "Attention-PI-HMONet"):
                        cache /= selected_mapping(method, mapping, options)
                    request = generate(
                        cache,
                        method,
                        geometry,
                        shape,
                        selected_mapping(method, mapping, options),
                        options,
                        checkpoint,
                        args.train_config,
                        version,
                        models,
                        defer=True,
                    )
                except Exception as error:
                    # Preserve ordinary per-case error reporting/stop_on_error.
                    print(f"  defer {method} batch preparation: {error}", flush=True)
                    continue
                if request is not None:
                    requests.append(request)
    try:
        execute_prediction_batches(requests, models)
    except Exception as error:
        # If one item prevents a batch, evaluate items through the original
        # path so valid cases can still finish under stop_on_error=False.
        print(f"Batch inference fallback to individual cases: {error}", flush=True)
    for index, (name, case, tag, shape, geometry, parameters) in enumerate(
        pending_jobs, 1
    ):
        options = evaluation_options(config, case, args)
        mappings = mapping_combinations(options)
        for mapping, options in mappings:
            print(
                f"[{dimension}D evaluation {index}] {name}, parameter={tag}, grid={'x'.join(map(str, shape))}, mapping={mapping}",
                flush=True,
            )
            try:
                records.append(
                    evaluate_object(
                        name,
                        case,
                        tag,
                        shape,
                        geometry,
                        parameters,
                        options,
                        args,
                        mapping,
                        version,
                        models,
                    )
                )
            except Exception as error:
                failures.append(
                    {
                        "case": name,
                        "parameter": tag,
                        "shape": shape,
                        "mapping": mapping,
                        "error": str(error),
                    }
                )
                print(f"ERROR: {type(error).__name__}: {error}", flush=True)
                if config.get("stop_on_error", True):
                    raise
    if failures:
        raise RuntimeError(f"{len(failures)} evaluation task(s) failed: {failures}")
    if not records:
        raise RuntimeError("No enabled evaluation tasks were selected")
    return records
