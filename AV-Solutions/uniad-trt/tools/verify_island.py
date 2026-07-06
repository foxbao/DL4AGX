#!/usr/bin/env python3
"""Sanity-check the map island: node-number span, boundary edges, and that
the map encoder front (transform/topk/sentinel) + residual merge are covered."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('onnx')
    ap.add_argument('spec')
    args = ap.parse_args()

    model = onnx.load(args.onnx)
    g = model.graph
    with open(args.spec) as f:
        island_names = {e.split(':')[0] for e in f.read().split(',') if e}

    name2node = {n.name: n for n in g.node if n.name}
    island = [name2node[n] for n in island_names if n in name2node]

    def num(n):
        try:
            return int(n.name.rsplit('_', 1)[1])
        except (IndexError, ValueError):
            return -1

    nums = sorted(num(n) for n in island)
    print('island count:', len(island))
    print('node-num span:', nums[0], '..', nums[-1])
    # gaps: is it one contiguous block or scattered?
    lows = [x for x in nums if x < 5000]
    print('nodes with num<5000 (map encoder front lives early?):', len(lows), lows[:20])

    # Which map initializers feed the island, and are transform/topk present?
    op_by = collections.Counter(n.op_type for n in island)
    print('island ops:', dict(op_by.most_common()))
    print('TopK nodes:', [n.name for n in island if n.op_type == 'TopK'])
    print('Where nodes:', [n.name for n in island if n.op_type == 'Where'])
    print('Softmax nodes:', [n.name for n in island if n.op_type == 'Softmax'])
    print('ScatterElements:', [n.name for n in island if n.op_type == 'ScatterElements'])

    # Boundary: island outputs consumed by NON-island nodes = fp32->fp16 casts.
    island_set = set(island_names)
    out2node = {}
    for n in g.node:
        for o in n.output:
            out2node[o] = n.name
    consumers = collections.defaultdict(list)
    for n in g.node:
        for i in n.input:
            consumers[i].append(n.name)
    boundary_out = []
    for n in island:
        for o in n.output:
            ext = [c for c in consumers.get(o, []) if c not in island_set]
            if ext:
                boundary_out.append((n.name, n.op_type, o, ext[:3]))
    print('== island->outside edges (fp32->fp16 handoff points) ==')
    for name, op, o, ext in boundary_out:
        print(f'  {name}({op}) -> {ext}')
    # Inputs to island from outside (fp16->fp32 handoff)
    graph_inputs = {i.name for i in g.input}
    boundary_in = set()
    for n in island:
        for i in n.input:
            src = out2node.get(i)
            if src and src not in island_set:
                boundary_in.add((src, i))
            elif i in graph_inputs:
                boundary_in.add(('<graph_input>', i))
    print('== outside->island edges (count):', len(boundary_in))
    for src, i in sorted(boundary_in)[:20]:
        print(f'  {src} -> {i}')


if __name__ == '__main__':
    main()
