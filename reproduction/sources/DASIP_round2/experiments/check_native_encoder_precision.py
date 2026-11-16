#!/usr/bin/env python3
"""Small CUDA compatibility probe; no SLAM, dataset run, or timing claim."""
import argparse
import json
from pathlib import Path
import torch
import tinycudann as tcnn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh output filename')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not accessible in this execution context')
    torch.manual_seed(0)
    probes = {}
    for name, settings in (
            ('HashGrid', dict(otype='HashGrid', n_levels=16, n_features_per_level=2,
                             log2_hashmap_size=16, base_resolution=16, per_level_scale=1.2)),
            ('OneBlob', dict(otype='OneBlob', n_bins=16))):
        encoder = tcnn.Encoding(n_input_dims=3, encoding_config=settings, dtype=torch.float)
        inputs = torch.rand((32, 3), device='cuda', dtype=torch.float32, requires_grad=True)
        values = encoder(inputs)
        values.square().mean().backward()
        torch.cuda.synchronize()
        parameters = list(encoder.parameters())
        gradients = [p.grad for p in parameters if p.numel()]
        finite_gradients = all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)
        row = dict(requested_dtype='torch.float32', output_dtype=str(values.dtype),
                   parameter_dtypes=sorted({str(p.dtype) for p in parameters if p.numel()}),
                   finite_output=bool(torch.isfinite(values).all()),
                   finite_parameter_gradients=finite_gradients,
                   finite_input_gradient=inputs.grad is not None and bool(torch.isfinite(inputs.grad).all()),
                   configuration=settings)
        assert row['output_dtype']=='torch.float32' and row['finite_output'], row
        assert row['finite_parameter_gradients'] and row['finite_input_gradient'], row
        probes[name] = row
    result = dict(device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  cuda_runtime=torch.version.cuda, capability=torch.cuda.get_device_capability(),
                  probes=probes, purpose='Encoder compatibility and exposed dtype check only',
                  limitation='Does not establish internal accumulator precision, upstream binary identity, SLAM quality or speed.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__=='__main__':
    main()
