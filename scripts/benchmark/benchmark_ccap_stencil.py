"""Verify exact CCAP stencil reuse and time full 3-D training updates.

Pass generated dataset directories to --validate-datasets (one per family).
--dataset supplies the timing batch; small datasets are cycled to batch-size,
with independently restored parameter-surface objects. No checkpoints are saved.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.parametric_dataset import ParametricDataset3D, read_sample_parameters
from src.data.geometry.parametric_geometry_3d import BoundarySurfaceSet3D
from src.data.mesh.sampling import sample_boundary, logical_grid, boundary_indices
from src.engine.training import predict_coordinates, sample_shared_random_points, interpolate_pde_mesh
from src.losses.physics import stencil_queries, elliptic_grid_scale_invariant_pde_loss, elliptic_grid_pde_residual
from src.models import HMONet3D, MeshONet3D
from src.models.pi_meshonet import PIMeshONet3D


def options(config, name):
    result = {**config['model']['common'], **config['model'][name]}
    result.pop('architecture', None)
    result.pop('residual_regularization_weight', None)
    return result


def verify(path, config, device, count):
    dataset = ParametricDataset3D(path)
    selected = [0, len(dataset) - 1]
    geometries = [dataset.boundary[i] for i in selected]
    shape = (41, 41, 11) if path.name.startswith('run_e') else (21, 21, 11)
    result = dict(dataset=str(path), shape=shape,
                  parameters=[read_sample_parameters(path, i) for i in selected], checks=[])
    for dtype in (torch.float32, torch.float64):
        boundary = torch.tensor(np.stack([sample_boundary(g, shape) for g in geometries]),
                                device=device, dtype=dtype)
        torch.manual_seed(20261030)
        model = HMONet3D(*shape, **options(config, 'hmonet')).to(device=device, dtype=dtype)
        for seed in (31, 47, 83):
            torch.manual_seed(seed)
            steps = boundary.new_tensor(config['hmonet_training']['pde_fd_step'])
            centers = sample_shared_random_points(2, count, 3, device, margin=steps).to(dtype)
            queries = stencil_queries(centers, steps)
            collected = []
            for reuse in (False, True):
                model.zero_grad(set_to_none=True)
                pred = predict_coordinates(model, boundary, queries.flatten(-3, -2),
                                           geometries, central_stencil=reuse).reshape(queries.shape)
                loss = elliptic_grid_scale_invariant_pde_loss(pred, steps)
                loss.backward()
                grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
                collected.append((pred.detach(), loss.detach(), grad.detach().clone()))
            differences = [float((a-b).abs().max()) for a, b in zip(*collected)]
            for reference, actual in zip(*collected):
                assert torch.isfinite(actual).all(), path
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            result['checks'].append(dict(dtype=str(dtype), seed=seed,
                output_max_abs=differences[0], loss_abs=differences[1], gradient_max_abs=differences[2]))
        del model, boundary, collected, pred, loss, grad
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return result


def benchmark(path, config, device, batch, count, repeats, warmups):
    dataset = ParametricDataset3D(path)
    shape = dataset.grid_shape
    indices = [i % len(dataset) for i in range(batch)]
    descriptors = [g.to_dict() for g in dataset.boundary]
    geometries = [BoundarySurfaceSet3D.from_dict(descriptors[i]) for i in indices]
    boundary = torch.tensor(dataset.boundary_points[indices], device=device, dtype=torch.float32)
    mesh = torch.tensor(dataset.mesh_points[indices], device=device, dtype=torch.float32)
    boundary_query = torch.tensor(logical_grid(shape)[boundary_indices(shape)], device=device, dtype=torch.float32)
    steps = boundary.new_tensor(config['hmonet_training']['pde_fd_step'])
    pi_steps = boundary.new_tensor(config['pimeshonet_training']['fd_step'])
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def iteration(model, optimizer, name, reuse, seed):
        torch.manual_seed(seed)
        sync()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        if name == 'meshonet':
            centers = sample_shared_random_points(batch, count, 3, device)
        else:
            h = pi_steps if name == 'pimeshonet' else steps
            centers = sample_shared_random_points(batch, count, 3, device, margin=h)
            stencil = stencil_queries(centers, h)
            if name == 'pimeshonet':
                query = torch.cat((boundary_query[None].expand(batch, -1, -1), stencil.flatten(-3, -2)), 1)
        sync(); t1 = time.perf_counter()
        if name == 'hmonet':
            pred = predict_coordinates(model, boundary, stencil.flatten(-3, -2),
                                       geometries, central_stencil=reuse).reshape(stencil.shape)
        elif name == 'pimeshonet':
            pred = predict_coordinates(model, boundary, query)
        else:
            pb = predict_coordinates(model, boundary, boundary_query[None].expand(batch, -1, -1))
            pred = predict_coordinates(model, boundary, centers)
        sync(); t2 = time.perf_counter()
        if name == 'hmonet':
            loss = config['hmonet_training']['physics_pde_loss_weight'] * elliptic_grid_scale_invariant_pde_loss(pred, steps)
        elif name == 'pimeshonet':
            nb = len(boundary_query)
            bl = (pred[:, :nb]-boundary).square().sum(-1).mean()
            pde = elliptic_grid_pde_residual(pred[:, nb:].reshape(batch, count, 19, 3), pi_steps).square().sum(-1).mean()
            loss = config['pimeshonet_training']['pde_loss_weight']*pde + config['pimeshonet_training']['boundary_loss_weight']*bl
        else:
            target = interpolate_pde_mesh(mesh, centers, shape)
            loss = config['meshonet_training']['alpha']*(pred-target).square().mean() + config['meshonet_training']['beta']*(pb-boundary).square().mean()
        optimizer.zero_grad(set_to_none=True)
        assert torch.isfinite(loss)
        sync(); t3 = time.perf_counter()
        loss.backward()
        sync(); t4 = time.perf_counter()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip_norm'])
        assert torch.isfinite(norm)
        optimizer.step()
        sync(); t5 = time.perf_counter()
        return dict(sampling_ms=1000*(t1-started), forward_ms=1000*(t2-t1), loss_ms=1000*(t3-t2),
                    backward_ms=1000*(t4-t3), optimizer_ms=1000*(t5-t4), total_ms=1000*(t5-started),
                    peak_MiB=torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else None)

    results = {}
    # Keep the two HMONet states matched, alternating execution order per pair.
    groups = [[('hmonet_reference', 'hmonet', HMONet3D, False), ('hmonet_reuse', 'hmonet', HMONet3D, True)],
              [('meshonet', 'meshonet', MeshONet3D, False)], [('pimeshonet', 'pimeshonet', PIMeshONet3D, False)]]
    for group in groups:
        models, optimizers = {}, {}
        for label, name, cls, reuse in group:
            torch.manual_seed(20261030)
            models[label] = cls(*shape, **options(config, name)).to(device).train()
            optimizers[label] = torch.optim.Adam(models[label].parameters(), lr=config[name+'_training']['learning_rate'])
            results[label] = dict(parameters=sum(p.numel() for p in models[label].parameters()), iterations=[])
        for i in range(warmups + repeats):
            for label, name, cls, reuse in (group if i % 2 == 0 else group[::-1]):
                row = iteration(models[label], optimizers[label], name, reuse, 20261030+i)
                if i >= warmups:
                    results[label]['iterations'].append(row)
                print(f'{label} {i+1}/{warmups+repeats}: {row["total_ms"]:.2f} ms', flush=True)
        models.clear(); optimizers.clear(); gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    for entry in results.values():
        entry['mean'] = {k:float(np.mean([r[k] for r in entry['iterations']]))
                         for k in entry['iterations'][0] if entry['iterations'][0][k] is not None}
        entry['std_total_ms'] = float(np.std([r['total_ms'] for r in entry['iterations']]))
    return dict(dataset=str(path), shape=shape, batch=batch, centers=count, boundary_points=boundary.shape[1],
                unique_geometries=len(set(indices)), dtype='float32', results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-datasets', nargs='*', type=Path, default=[])
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/3D/train.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--points', type=int, default=400)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--warmups', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    config = yaml.safe_load(args.config.read_text(encoding='utf-8-sig'))
    report = dict(device=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
                  torch_version=torch.__version__, validations=[],
                  timing_scope='Full batch update including sampling, forward, loss, backward, clipping and Adam; excludes IO and diagnostic full-grid MSE',
                  config=config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    try:
        for path in args.validate_datasets:
            result = verify(path, config, device, args.points)
            report['validations'].append(result); save()
            print('VERIFIED', path.name, flush=True)
        if args.dataset:
            report['benchmark'] = benchmark(args.dataset, config, device, args.batch_size, args.points, args.repeats, args.warmups)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
