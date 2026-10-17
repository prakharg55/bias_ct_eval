#!/usr/bin/env python3
"""Rebuild the full training configurations from the overlays in configs/ and a local checkout of c-wei/AttCT.

Usage: python3 materialize_configs.py --attct /path/to/AttCT --configs ../configs --out ./full_configs
The checkout must be at the commit named in each overlay (b3c1896). Each output file is the base configuration of the
repository with the dotted keys in `set` replaced and the keys in `unset` removed."""
import argparse, os, subprocess, yaml


def put(d, dotted, v):
    ks = dotted.split("."); 
    for k in ks[:-1]: d = d.setdefault(k, {})
    d[ks[-1]] = v


def drop(d, dotted):
    ks = dotted.split(".")
    for k in ks[:-1]: d = d[k]
    del d[ks[-1]]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--attct", required=True); ap.add_argument("--configs", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    head = subprocess.run(["git", "-C", a.attct, "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True).stdout.strip()
    n = 0
    for dp, _, fs in os.walk(a.configs):
        for f in sorted(fs):
            if not f.endswith(".yaml"): continue
            ov = yaml.safe_load(open(os.path.join(dp, f)))
            if head and not head.startswith(ov["attct_commit"]): raise SystemExit(f"AttCT checkout is at {head}, overlay needs {ov['attct_commit']}")
            cfg = yaml.safe_load(open(os.path.join(a.attct, ov["base"])))
            for k, v in ov.get("set", {}).items(): put(cfg, k, v)
            for k in ov.get("unset", []): drop(cfg, k)
            o = os.path.join(a.out, os.path.relpath(os.path.join(dp, f), a.configs)); os.makedirs(os.path.dirname(o), exist_ok=True)
            yaml.safe_dump(cfg, open(o, "w"), sort_keys=False); n += 1
    print(f"wrote {n} configurations to {a.out}")


if __name__ == "__main__":
    main()
