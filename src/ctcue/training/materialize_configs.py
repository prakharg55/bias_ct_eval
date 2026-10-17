"""Write the full training configurations from the overrides in configs/.

Each file in configs/ names a configuration of the AttCT repository (at commit b3c1896) and the keys
we changed. This script writes the full files into the repository's configs/ folder. Paths in the
overrides that start with /data/ct_root or /data/hf_home are written under $CT_ROOT and $HF_HOME (the
same paths when these variables are not set).

Usage: python -m ctcue.training.materialize_configs --attct /path/to/AttCT --configs configs
           --out /path/to/AttCT/configs
"""

import argparse, os, subprocess, yaml
from ctcue.paths import CT_ROOT, HF_HOME

# The overlays give data and model paths under /data/ct_root and /data/hf_home.
# localize() moves them under CT_ROOT and HF_HOME (the same paths unless those variables are set).
LOCAL = {"/data/ct_root": CT_ROOT, "/data/hf_home": HF_HOME}


def localize(v):
    for old, new in LOCAL.items():
        if isinstance(v, str) and (v == old or v.startswith(old + "/")):
            return new + v[len(old) :]
    return v


def put(d, dotted, v):
    ks = dotted.split(".")
    for k in ks[:-1]:
        d = d.setdefault(k, {})
    d[ks[-1]] = v


def drop(d, dotted):
    ks = dotted.split(".")
    for k in ks[:-1]:
        d = d[k]
    del d[ks[-1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attct", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    head = subprocess.run(
        ["git", "-C", a.attct, "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    n = 0
    for dp, _, fs in os.walk(a.configs):
        for f in sorted(fs):
            if not f.endswith(".yaml"):
                continue
            ov = yaml.safe_load(open(os.path.join(dp, f)))
            if head and not head.startswith(ov["attct_commit"]):
                raise SystemExit(f"AttCT checkout is at {head}, overlay needs {ov['attct_commit']}")
            cfg = yaml.safe_load(open(os.path.join(a.attct, ov["base"])))
            for k, v in ov.get("set", {}).items():
                put(cfg, k, localize(v))
            for k in ov.get("unset", []):
                drop(cfg, k)
            o = os.path.join(a.out, os.path.relpath(os.path.join(dp, f), a.configs))
            os.makedirs(os.path.dirname(o), exist_ok=True)
            yaml.safe_dump(cfg, open(o, "w"), sort_keys=False)
            n += 1
    print(f"wrote {n} configurations to {a.out}")


if __name__ == "__main__":
    main()
