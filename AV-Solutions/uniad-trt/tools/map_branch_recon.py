#!/usr/bin/env python3
"""Locate the HD-map planning branch nodes in the mapfuse dense ONNX.

Strategy: forward-taint from every node that consumes a map-branch
initializer (map_lane_encoder / planning_head.map_attn_module /
map_delta_proj / map_gate), then subtract the forward-taint of the shared
BEV attention (planning_head.attn_module) and reg_branch. What remains is the
map attention island that must stay fp32 under TensorRT --fp16.
"""
import argparse
import collections
import onnx


MAP_INIT_PREFIXES = (
    'model.map_lane_encoder.',
    'model.planning_head.map_attn_module.',
    'model.planning_head.map_delta_proj.',
    'model.planning_head.map_gate.',
)
BARRIER_INIT_PREFIXES = (
    'model.planning_head.attn_module.',
    'model.planning_head.reg_branch.',
)


def build_consumers(graph):
    consumers = collections.defaultdict(list)
    for i, node in enumerate(graph.node):
        for inp in node.input:
            consumers[inp].append(i)
    return consumers


def seed_nodes(graph, prefixes):
    init_names = {init.name for init in graph.initializer
                  if init.name.startswith(prefixes)}
    seeds = set()
    for i, node in enumerate(graph.node):
        if any(inp in init_names for inp in node.input):
            seeds.add(i)
    return seeds


def forward_taint(graph, consumers, seeds, stop_at=None):
    """BFS forward from seed nodes over tensor edges. stop_at node ids are
    tainted but their outputs are not propagated (acts as an absorbing wall)."""
    stop_at = stop_at or set()
    out_index = {}
    for i, node in enumerate(graph.node):
        for out in node.output:
            out_index[out] = i
    tainted = set(seeds)
    queue = collections.deque(seeds)
    while queue:
        nid = queue.popleft()
        if nid in stop_at:
            continue
        for out in graph.node[nid].output:
            for cid in consumers.get(out, ()):
                if cid not in tainted:
                    tainted.add(cid)
                    queue.append(cid)
    return tainted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('onnx')
    ap.add_argument('--out', default=None,
                    help='write matched layer-precision spec to this file')
    ap.add_argument('--families', default=None,
                    help='comma-separated op_types to keep (semantic subset, '
                         'e.g. MatMul,Gemm,Softmax,ReduceMean,Pow,Sqrt,Div)')
    args = ap.parse_args()

    model = onnx.load(args.onnx)
    g = model.graph
    consumers = build_consumers(g)

    map_seeds = seed_nodes(g, MAP_INIT_PREFIXES)
    barrier_seeds = seed_nodes(g, BARRIER_INIT_PREFIXES)

    # Barrier: everything downstream of shared BEV attention + reg_branch.
    barrier = forward_taint(g, consumers, barrier_seeds)
    # Map island: forward from map seeds, but stop when we hit the barrier so
    # the shared trunk (attn_module -> reg_branch) is excluded.
    island = forward_taint(g, consumers, map_seeds, stop_at=barrier)
    island -= barrier

    if args.families:
        keep = {s for s in args.families.split(',') if s}
        island = {i for i in island if g.node[i].op_type in keep}

    fam = collections.Counter(g.node[i].op_type for i in island)
    print('== map island ==')
    print('seeds(map):', len(map_seeds), ' barrier nodes:', len(barrier))
    print('island size:', len(island))
    print('by op:', dict(fam.most_common()))

    named = sorted(g.node[i].name for i in island if g.node[i].name)
    print('== sample island node names (first 40) ==')
    for n in named[:40]:
        print('  ', n)

    if args.out:
        # Emit explicit per-node layerPrecisions spec (exact names, fp32).
        spec = ','.join(f'{g.node[i].name}:fp32'
                        for i in sorted(island) if g.node[i].name)
        with open(args.out, 'w') as f:
            f.write(spec)
        print('== wrote spec ==')
        print('  file:', args.out, ' entries:',
              sum(1 for i in island if g.node[i].name))


if __name__ == '__main__':
    main()
