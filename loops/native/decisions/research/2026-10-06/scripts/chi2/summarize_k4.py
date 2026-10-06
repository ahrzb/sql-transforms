import json, re, collections, sys
r = json.load(open(sys.argv[1]))
agg = collections.OrderedDict()
for k, v in r.items():
    if not isinstance(v, dict) or 'max ulps' not in v: continue
    m = re.match(r'(\S+) steps=(\d+) s=(\S+) (\S+)', k)
    tag, steps, s, lane = m.groups()
    fam = tag.split('(')[0]
    a = agg.setdefault((steps, s, fam), dict(n=0, mism=0, maxulps=0, maxK=0, K1=0, K09=0, lanes=0, sqrt_ulps=0, nologdiff=0, arg=None))
    if lane == 'sqrt':
        a['n'] += v['n']; a['mism'] += v['log mismatches']; a['sqrt_ulps'] = max(a['sqrt_ulps'], v['max ulps']); continue
    a['lanes'] += v['n']; a['nologdiff'] += v['lanes differ where logs agree']
    a['maxulps'] = max(a['maxulps'], v['max ulps'])
    if v['max K'] > a['maxK']: a['maxK'] = v['max K']; a['arg'] = (tag, lane, v['argmax'])
    a['K1'] += v['K>1']; a['K09'] += v['K>0.9']
print("| steps | s | set | draws | log mismatches | lane evals | max ulps | max K | lanes K>1 | lanes differ w/o log mismatch | sqrt lane ulps |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for (steps, s, fam), a in agg.items():
    print(f"| {steps} | {s} | {fam} | {a['n']:,} | {a['mism']:,} | {a['lanes']:,} | {a['maxulps']:,.0f} | {a['maxK']:.3f} | {a['K1']} | {a['nologdiff']} | {a['sqrt_ulps']:.0f} |")
for (steps, s, fam), a in agg.items():
    if a['maxK'] > 0.95: print(steps, s, fam, a['arg'])
