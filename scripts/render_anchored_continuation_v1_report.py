#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from pathlib import Path
from safetensors import safe_open

ROOT=Path(__file__).resolve().parents[1]
A=ROOT/"artifacts/continuation/v11-schedule-b-only-anchor-v1"
ANCHOR=ROOT/"artifacts/targeted-preservation-sft-smoke-v11/adapter/adapter_model.safetensors"

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()

def main()->None:
    m=json.loads((A/"metrics.json").read_text()); r=json.loads((A/"reload_validation.json").read_text())
    with safe_open(ANCHOR,framework="pt",device="cpu") as x,safe_open(A/"adapter/adapter_model.safetensors",framework="pt",device="cpu") as y:
        a_changed=sum(not bool((x.get_tensor(k)==y.get_tensor(k)).all()) for k in x.keys() if ".lora_A." in k)
        b_changed=sum(not bool((x.get_tensor(k)==y.get_tensor(k)).all()) for k in x.keys() if ".lora_B." in k)
        nonfinite=sum(not bool(y.get_tensor(k).isfinite().all()) for k in y.keys())
    p=m["post_validation"]; base=m["baseline_validation"]
    passed=(base["exact"]==23 and base["semantic_exact"]==26 and p["exact"]>=22 and p["semantic_exact"]>=24 and p["by_family"]["capability_preservation_numeric"]["semantic_exact"]>=10 and p["by_family"]["capability_preservation_choice"]["semantic_exact"]>=5 and p["by_family"]["semantic_arithmetic_process"]["semantic_exact"]>=7 and r["validation"]==p and a_changed==0 and b_changed==112 and nonfinite==0 and not m["failure_receipt_exists"])
    if not passed:raise SystemExit("anchored continuation does not pass frozen gate")
    report={
      "schema_version":"nano_train_public_anchored_continuation_v1","experiment_id":m["experiment_id"],"pre_registration_revision":"71c769c","passed":True,
      "identity":{"config_sha256":"41e9c47229f52b9ff3c97985b6a637429f17756791226eaca6b06f703d863c59","anchor_adapter_tree_sha256":m["anchor_adapter_tree_sha256"],"adapter_tree_sha256":m["adapter_sha256"],"dataset_sha256":m["dataset"]["sha256"],"model_config_sha256":m["model_config_sha256"]},
      "method":{"max_steps":8,"examples_seen":32,"schedule_examples_seen":4,"trainable_lora_b_only":True,"frozen_lora_a":True,"trainable_parameters":m["trainable_parameters"],"anchor_penalty_coefficient":1.0,"anchor_norm_l2":m["anchor_norm_l2"],"drift_norm_l2":m["drift_norm_l2"],"relative_drift_l2":m["relative_drift_l2"],"a_tensors_changed":a_changed,"b_tensors_changed":b_changed},
      "baseline_validation":base,"post_validation":p,"loss_curve":m["loss_curve"],
      "validation":{"reload_matches":r["validation"]==p,"reload_peak_allocated_gib":r["peak_allocated_gib"],"nonfinite_tensors":nonfinite,"training_peak_allocated_gib":m["peak_allocated_gib"],"failure_receipt_exists":m["failure_receipt_exists"]},
      "artifacts":{"metrics_sha256":sha(A/"metrics.json"),"generations_sha256":sha(A/"generations.json"),"reload_validation_sha256":sha(A/"reload_validation.json"),"adapter_sha256":m["adapter_sha256"]},
      "decision":{"accepted_local_smoke":True,"sealed_canary_allowed":True,"prior_full_suite_allowed":False,"independent_holdout_allowed":False,"merge_allowed":False,"scale_allowed":False,"rl_allowed":False,"next_action":"Run only the old sealed 40-case regression canary on the exact anchored adapter. Passing permits the old 211-case development suite, not the independent holdout."}
    }
    md=f"""# V11 Schedule B-Only Anchored Continuation Result

The anchored continuation passes every frozen local gate:

- baseline reproduces v11 at 23/32 strict and 26/32 semantic;
- post result: {p['exact']}/32 strict, {p['semantic_exact']}/32 semantic;
- numeric / choice / process semantic: {p['by_family']['capability_preservation_numeric']['semantic_exact']}/16, {p['by_family']['capability_preservation_choice']['semantic_exact']}/8, {p['by_family']['semantic_arithmetic_process']['semantic_exact']}/8;
- 112/112 LoRA A tensors remain byte-identical; 112/112 B tensors change;
- relative B drift: {m['relative_drift_l2']:.6f};
- independent reload exactly matches.

This is a preservation-method pass, not independent quality evidence. It
authorizes only the old sealed regression canary. Full suite, independent
holdout, merge, scale, and RL remain blocked.
"""
    out=ROOT/"docs/results";out.mkdir(parents=True,exist_ok=True)
    (out/"v11_schedule_b_only_anchor_continuation_v1.public.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    (out/"v11_schedule_b_only_anchor_continuation_v1.md").write_text(md)
    print(json.dumps({"passed":True,"strict":p["exact"],"semantic":p["semantic_exact"],"sealed_canary_allowed":True},sort_keys=True))
if __name__=="__main__":main()
