#!/usr/bin/env python3
"""Validate declarative Tensor Flow DAG specs (`models/<slug>/graph.json` + `model.yaml`).

Checks performed:
  1. Required fields, enum domains (`OpKind`, `EdgeKind`, `LegendStyle`, `shape_mode`, `modality`).
  2. Unique node IDs, valid edge endpoints, no self-loops, valid `lane` (0..=7) and `group_idx`.
  3. Geometric guardrails:
     - Same-row nodes must satisfy `|col_q1 - col_q2| >= 3` (177px > 158px node width) to prevent overlap.
     - Nodes inside a group's `[row_start, row_end]` must lie within `[min_col_q, max_col_q]`.
     - Two `macro_skip` edges sharing the same `lane` must not overlap vertically.
  4. Connectivity check (no isolated orphan nodes when `len(nodes) > 1`).

Usage:
  python3 scripts/check_graphs.py [models_dir]
  python3 scripts/check_graphs.py --self-test
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

VALID_OP_KINDS = {
  "input",
  "embedding",
  "conv_or_linear",
  "avg_pool",
  "max_pool",
  "attention",
  "moe_router",
  "merge",
  "softmax",
  "fc_head",
  "dropout",
}
VALID_EDGE_KINDS = {"main", "inner_skip", "macro_skip", "aux_bypass"}
VALID_LEGEND_STYLES = {"classic_cnn", "llm_or_multimodal"}
VALID_SHAPE_MODES = {"image_2d", "volume_3d", "vector", "seq"}
VALID_MODALITIES = {
  "llm",
  "vlm",
  "audio",
  "diffusion",
  "embedding",
  "detection",
  "segmentation",
  "classic",
}


def validate_graph_dict(data: dict[str, Any], label: str) -> list[str]:
  errors: list[str] = []

  title = str(data.get("title", "")).strip()
  if not title:
    errors.append(f"{label}: missing non-empty `title`")

  legend_style = data.get("legend_style", "classic_cnn")
  if legend_style not in VALID_LEGEND_STYLES:
    errors.append(f"{label}: invalid `legend_style`: {legend_style!r}")

  shape_mode = data.get("shape_mode", "image_2d")
  if shape_mode not in VALID_SHAPE_MODES:
    errors.append(f"{label}: invalid `shape_mode`: {shape_mode!r}")

  nodes = data.get("nodes")
  edges = data.get("edges")
  groups = data.get("groups", [])

  if not isinstance(nodes, list) or not nodes:
    errors.append(f"{label}: `nodes` must be a non-empty list")
    return errors
  if not isinstance(edges, list):
    errors.append(f"{label}: `edges` must be a list")
    return errors
  if not isinstance(groups, list):
    errors.append(f"{label}: `groups` must be a list")
    return errors

  node_by_id: dict[str, dict[str, Any]] = {}
  rows_map: dict[int, list[tuple[str, int]]] = {}

  for idx, n in enumerate(nodes):
    if not isinstance(n, dict):
      errors.append(f"{label}: nodes[{idx}] must be an object")
      continue
    nid = str(n.get("id", "")).strip()
    if not nid:
      errors.append(f"{label}: nodes[{idx}] missing `id`")
      continue
    if nid in node_by_id:
      errors.append(f"{label}: duplicate node id `{nid}`")
      continue
    kind = n.get("kind", "conv_or_linear")
    if kind not in VALID_OP_KINDS:
      errors.append(f"{label}: node `{nid}` invalid kind `{kind}`")
    if not str(n.get("op_title", "")).strip():
      errors.append(f"{label}: node `{nid}` missing `op_title`")
    col_q = n.get("col_q", 0)
    row = n.get("row")
    if not isinstance(col_q, int) or not isinstance(row, int) or row < 0:
      errors.append(f"{label}: node `{nid}` invalid col_q/row ({col_q!r}, {row!r})")
      continue
    node_by_id[nid] = n
    rows_map.setdefault(row, []).append((nid, col_q))

  # Geometric guardrail 1: prevent same-row horizontal box overlap (node width 158px, step 59px => >=3)
  for row, entries in rows_map.items():
    entries.sort(key=lambda item: item[1])
    for i in range(1, len(entries)):
      prev_id, prev_q = entries[i - 1]
      cur_id, cur_q = entries[i]
      if cur_q - prev_q < 3:
        errors.append(
          f"{label}: row {row} horizontal overlap risk between `{prev_id}` (col_q={prev_q}) "
          f"and `{cur_id}` (col_q={cur_q}); require delta >= 3"
        )

  # Geometric guardrail 2: validate groups & containment
  for gi, g in enumerate(groups):
    if not isinstance(g, dict):
      errors.append(f"{label}: groups[{gi}] must be an object")
      continue
    rs = g.get("row_start")
    re = g.get("row_end")
    min_q = g.get("min_col_q")
    max_q = g.get("max_col_q")
    if not all(isinstance(v, int) for v in (rs, re, min_q, max_q)):
      errors.append(f"{label}: groups[{gi}] row/col bounds must be integers")
      continue
    if rs > re:
      errors.append(f"{label}: groups[{gi}] row_start({rs}) > row_end({re})")
    if min_q > max_q:
      errors.append(f"{label}: groups[{gi}] min_col_q({min_q}) > max_col_q({max_q})")
    for nid, n in node_by_id.items():
      r = n["row"]
      q = n.get("col_q", 0)
      if rs <= r <= re and not (min_q <= q <= max_q):
        errors.append(
          f"{label}: node `{nid}` (row={r}, col_q={q}) lies outside groups[{gi}] col range [{min_q}, {max_q}]"
        )

  # Edge validation + connectivity + multi-lane interval conflict check
  degree: dict[str, int] = {nid: 0 for nid in node_by_id}
  macro_lane_spans: dict[int, list[tuple[int, int, str, str]]] = {}

  for ei, e in enumerate(edges):
    if not isinstance(e, dict):
      errors.append(f"{label}: edges[{ei}] must be an object")
      continue
    src = str(e.get("from", ""))
    dst = str(e.get("to", ""))
    if src not in node_by_id:
      errors.append(f"{label}: edges[{ei}] unknown `from` id `{src}`")
      continue
    if dst not in node_by_id:
      errors.append(f"{label}: edges[{ei}] unknown `to` id `{dst}`")
      continue
    if src == dst:
      errors.append(f"{label}: edges[{ei}] self-loop on `{src}`")
      continue
    ekind = e.get("kind", "main")
    if ekind not in VALID_EDGE_KINDS:
      errors.append(f"{label}: edges[{ei}] invalid kind `{ekind}`")
    lane = e.get("lane", 0)
    if not isinstance(lane, int) or not (0 <= lane <= 7):
      errors.append(f"{label}: edges[{ei}] lane must be int in 0..=7, got {lane!r}")
    gidx = e.get("group_idx")
    if gidx is not None and (not isinstance(gidx, int) or not (0 <= gidx < len(groups))):
      errors.append(f"{label}: edges[{ei}] group_idx {gidx!r} out of range")

    degree[src] += 1
    degree[dst] += 1

    if ekind == "macro_skip":
      r1 = node_by_id[src]["row"]
      r2 = node_by_id[dst]["row"]
      if r2 > r1 + 1 and isinstance(lane, int):
        macro_lane_spans.setdefault(lane, []).append((r1, r2, src, dst))

  # Geometric guardrail 3: prevent vertical overlap on the same macro_skip lane
  for lane, spans in macro_lane_spans.items():
    spans.sort()
    for i in range(1, len(spans)):
      r1_a, r2_a, s_a, d_a = spans[i - 1]
      r1_b, r2_b, s_b, d_b = spans[i]
      if r1_b < r2_a:
        errors.append(
          f"{label}: macro_skip lane {lane} overlap between {s_a}->{d_a} (rows {r1_a}..{r2_a}) "
          f"and {s_b}->{d_b} (rows {r1_b}..{r2_b}); assign distinct lanes"
        )

  if len(node_by_id) > 1:
    for nid, deg in degree.items():
      if deg == 0:
        errors.append(f"{label}: orphan isolated node `{nid}`")

  return errors


def validate_model_yaml_light(yaml_path: pathlib.Path) -> list[str]:
  errors: list[str] = []
  text = yaml_path.read_text(encoding="utf-8")
  fields: dict[str, str] = {}
  for line in text.splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("- "):
      continue
    if ":" in stripped and not line.startswith(" "):
      k, v = stripped.split(":", 1)
      fields[k.strip()] = v.strip().strip('"').strip("'")
  for req in ("name", "modality", "summary"):
    if not fields.get(req):
      errors.append(f"{yaml_path}: missing required field `{req}`")
  mod = fields.get("modality", "").lower()
  if mod and mod not in VALID_MODALITIES:
    errors.append(f"{yaml_path}: invalid modality `{mod}` (expected one of {sorted(VALID_MODALITIES)})")
  return errors


def run_self_test() -> int:
  good_graph = {
    "title": "U-Net Self-Test",
    "legend_style": "classic_cnn",
    "shape_mode": "image_2d",
    "groups": [{"badge": "Encoder", "row_start": 1, "row_end": 2, "min_col_q": -2, "max_col_q": 2}],
    "nodes": [
      {"id": "in", "kind": "input", "op_title": "Input", "col_q": 0, "row": 0},
      {"id": "e1", "kind": "conv_or_linear", "op_title": "Enc1", "col_q": 0, "row": 1},
      {"id": "e2", "kind": "conv_or_linear", "op_title": "Enc2", "col_q": 0, "row": 2},
      {"id": "bot", "kind": "attention", "op_title": "Bot", "col_q": 0, "row": 3},
      {"id": "d2", "kind": "merge", "op_title": "Dec2", "col_q": 0, "row": 4},
      {"id": "d1", "kind": "merge", "op_title": "Dec1", "col_q": 0, "row": 5},
    ],
    "edges": [
      {"from": "in", "to": "e1"},
      {"from": "e1", "to": "e2"},
      {"from": "e2", "to": "bot"},
      {"from": "bot", "to": "d2"},
      {"from": "d2", "to": "d1"},
      {"from": "e2", "to": "d2", "kind": "macro_skip", "lane": 0, "label": "Skip 1/2"},
      {"from": "e1", "to": "d1", "kind": "macro_skip", "lane": 1, "label": "Skip 1/1"},
    ],
  }
  good_errs = validate_graph_dict(good_graph, "self-test-green")
  if good_errs:
    print("SELF-TEST FAIL (expected green):", good_errs, file=sys.stderr)
    return 1

  bad_graph = {
    "title": "Bad Graph",
    "nodes": [
      {"id": "a", "kind": "input", "op_title": "A", "col_q": 0, "row": 0},
      {"id": "b", "kind": "conv_or_linear", "op_title": "B", "col_q": 1, "row": 0},
      {"id": "c", "kind": "merge", "op_title": "C", "col_q": 0, "row": 2},
      {"id": "d", "kind": "merge", "op_title": "D", "col_q": 0, "row": 3},
    ],
    "edges": [
      {"from": "a", "to": "c", "kind": "macro_skip", "lane": 0},
      {"from": "b", "to": "d", "kind": "macro_skip", "lane": 0},
      {"from": "a", "to": "ghost"},
    ],
  }
  bad_errs = validate_graph_dict(bad_graph, "self-test-red")
  if len(bad_errs) < 3:
    print("SELF-TEST FAIL (expected >=3 red errors):", bad_errs, file=sys.stderr)
    return 1

  print(f"check_graphs.py self-test OK (green=0 errors, red={len(bad_errs)} errors caught)")
  return 0


def main(argv: list[str]) -> int:
  if "--self-test" in argv:
    return run_self_test()

  root = pathlib.Path(argv[1]) if len(argv) > 1 else pathlib.Path("assets/models")
  if not root.exists():
    print(f"Path not found: {root}", file=sys.stderr)
    return 2

  graph_files = sorted(root.rglob("graph.json"))
  if not graph_files:
    print(f"OK: 0 declarative graph.json files found under {root}")
    return 0

  all_errors: list[str] = []
  for gf in graph_files:
    try:
      data = json.loads(gf.read_text(encoding="utf-8"))
    except Exception as exc:
      all_errors.append(f"{gf}: invalid JSON ({exc})")
      continue
    if not isinstance(data, dict):
      all_errors.append(f"{gf}: top-level JSON must be an object")
      continue
    all_errors.extend(validate_graph_dict(data, str(gf)))
    yaml_path = gf.parent / "model.yaml"
    if yaml_path.is_file():
      all_errors.extend(validate_model_yaml_light(yaml_path))
    else:
      all_errors.append(f"{gf.parent}: missing sibling model.yaml")

  if all_errors:
    for err in all_errors:
      print(f"ERR: {err}", file=sys.stderr)
    print(f"FAILED: {len(all_errors)} errors across {len(graph_files)} graph.json file(s)", file=sys.stderr)
    return 1

  print(f"OK: validated {len(graph_files)} declarative graph.json file(s) with 0 errors")
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv))
