# /// script
# requires-python = ">=3.7,<3.8"
# dependencies = ["tensorflow==1.15.5", "numpy==1.18.5"]
# ///
import collections
import csv
import json
import time
import numpy as np
from google.protobuf import text_format
from runtime import ROOT, LABEL, Network, inputs, packet, write_json


def main() -> None:
    start = time.monotonic()
    out = ROOT / 'run-output/network'
    out.mkdir(parents=True, exist_ok=True)
    with Network() as net:
        signatures = {}
        for key, sig in net.meta.signature_def.items():
            signatures[key] = {'method': sig.method_name}
            for side in ('inputs', 'outputs'):
                signatures[key][side] = {k: {'name': v.name, 'dtype': int(v.dtype),
                    'shape': [d.size for d in v.tensor_shape.dim]}
                    for k, v in getattr(sig, side).items()}
        graph_def = net.graph.as_graph_def(add_shapes=True)
        (out / 'graph.pbtxt').write_text(text_format.MessageToString(graph_def))
        variables = []
        scopes = collections.defaultdict(int)
        for variable in net.graph.get_collection('variables'):
            shape = variable.shape.as_list()
            count = int(np.prod(shape))
            scope = variable.name.rsplit('/', 1)[0]
            scopes[scope] += count
            variables.append({'name': variable.name, 'shape': shape, 'params': count, 'scope': scope})
        image = next(inputs())[1]
        feed = {net.ins['image']: [image], net.ins['secret']: [packet()]}
        kinds = {'Conv2D', 'MatMul', 'BiasAdd', 'Relu', 'Reshape', 'ResizeNearestNeighbor',
                 'ConcatV2', 'Sigmoid', 'Round', 'Add', 'AddV2', 'AddN', 'Sub', 'Minimum', 'Maximum'}
        reachable = set()
        pending = [tensor.op for tensor in net.outs.values()]
        while pending:
            op = pending.pop()
            if op.name in reachable:
                continue
            reachable.add(op.name)
            pending.extend(tensor.op for tensor in op.inputs)
        stages = []
        all_ops = []
        for op in net.graph.get_operations():
            all_ops.append({'name': op.name, 'type': op.type, 'inputs': [t.name for t in op.inputs],
                            'outputs': [{'name': t.name, 'shape': str(t.shape)} for t in op.outputs]})
            if op.name in reachable and op.type in kinds and op.outputs and op.outputs[0].dtype.is_floating:
                value = net.session.run(op.outputs[0], feed)
                attributes = {}
                if op.type == 'Conv2D':
                    attributes = {'strides': list(op.get_attr('strides')),
                                  'padding': op.get_attr('padding').decode('ascii'),
                                  'data_format': op.get_attr('data_format').decode('ascii')}
                if op.type == 'ResizeNearestNeighbor':
                    attributes = {'align_corners': bool(op.get_attr('align_corners'))}
                stages.append({'name': op.name, 'type': op.type, 'attributes': attributes,
                               'inputs': [{'name': t.name, 'shape': str(t.shape)} for t in op.inputs],
                               'tensor': op.outputs[0].name, 'static_shape': str(op.outputs[0].shape),
                               'runtime_shape': list(value.shape), 'finite': bool(np.isfinite(value).all()),
                               'min': float(value.min()), 'max': float(value.max())})
        modules = collections.defaultdict(int)
        for variable in variables:
            modules[variable['name'].split('/')[0]] += variable['params']
        result = {'label': LABEL, 'measurement': 'loaded SavedModel; actual intermediate fetches on Airplane',
                  'signatures': signatures, 'variables': variables, 'params_by_scope': dict(scopes),
                  'params_by_module': dict(modules),
                  'params_total': sum(v['params'] for v in variables), 'stages': stages}
        write_json(out / 'shapes_params.json', result)
        write_json(out / 'graph_ops.json', all_ops)
        write_json(out / 'signatures.json', {'label': LABEL, 'signatures': signatures})
        write_json(out / 'measurement_timing.json', {'seconds': time.monotonic() - start})
        with (out / 'network_layers.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['label', 'module', 'params', 'inputs', 'output', 'runtime_shape', 'attributes'])
            normalized = json.loads((out / 'shapes_params.json').read_text())
            for stage in normalized['stages']:
                if stage['type'] in {'Conv2D', 'MatMul'}:
                    scope = stage['name'].rsplit('/', 1)[0]
                    writer.writerow([LABEL, scope, normalized['params_by_scope'].get(scope, 0),
                                     json.dumps(stage['inputs']), stage['tensor'],
                                     json.dumps(stage['runtime_shape']), json.dumps(stage['attributes'])])
        print('PARAMS', result['params_total'], dict(scopes), flush=True)
        print('SIGNATURES', signatures, flush=True)
        for row in stages:
            if row['type'] in {'Conv2D', 'MatMul', 'ResizeNearestNeighbor', 'Reshape'}:
                print(row['name'], row['runtime_shape'], flush=True)


if __name__ == '__main__':
    main()
