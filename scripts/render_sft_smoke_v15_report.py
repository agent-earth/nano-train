#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, math, statistics
from pathlib import Path
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts/schedule-isolation-preservation-sft-smoke-v15"
V11 = ROOT / "docs/results/targeted_preservation_sft_smoke_v11.public.json"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    m=json.loads((A/"metrics.json").read_text()); r=json.loads((A/"reload_validation.json").read_text())
    g=json.loads((A/"generations.json").read_text()); old=json.loads(V11.read_text())["post_sft_validation"]
    p=m["post_sft_validation"]; losses=[float(x["loss"]) for x in m["loss_curve"]]
    with safe_open(A/"adapter/adapter_model.safetensors",framework="pt",device="cpu") as f:
        tensors=len(f.keys()); nonfinite=sum(not bool(f.get_tensor(k).isfinite().all()) for k in f.keys())
    deltas={}
    for family,row in p["by_family"].items():
        before=set(old["by_family"][family]["semantic_failure_sample_ids"]); after=set(row["semantic_failure_sample_ids"])
        deltas[family]={"semantic_delta":row["semantic_exact"]-old["by_family"][family]["semantic_exact"],"strict_delta":row["exact"]-old["by_family"][family]["exact"],"fixed_sample_ids":sorted(before-after),"regressed_sample_ids":sorted(after-before)}
    report={
      "schema_version":"nano_train_public_sft_smoke_v15","experiment_id":m["experiment_id"],"pre_registration_revision":"85d7435","data_revision":"890b576","passed":False,
      "identity":{"config_sha256":"413cff6c370c69a9ef6ac9d4ebef32bf3f695ecd14f02c9f105da2893f63230d","dataset_sha256":m["dataset"]["sha256"],"model_config_sha256":m["model"]["config_sha256"],"adapter_sha256":m["adapter_sha256"]},
      "configuration":{"max_steps":32,"examples_seen":128,"schedule_examples_seen":7,"dtype":m["config"]["dtype"],"effective_batch_size":4,"generation_max_new_tokens":128},
      "baseline_validation":m["baseline_validation"],"post_sft_validation":p,
      "versus_v11":{"aggregate_exact_delta":p["exact"]-old["exact"],"aggregate_semantic_delta":p["semantic_exact"]-old["semantic_exact"],"family_deltas":deltas},
      "mechanism":{"isolated_family":"weighted_recurring_schedule_total","semantic_gain":True,"strict_and_choice_preservation_failed":True,"result":"capability_gain_with_contract_interference","data_family_matrix_complete":True},
      "optimization":{"steps":len(losses),"all_losses_finite":all(math.isfinite(x) for x in losses),"early_five_step_mean":statistics.mean(losses[:5]),"late_five_step_mean":statistics.mean(losses[-5:]),"peak_allocated_gib":m["hardware"]["peak_allocated_gib"],"failure_receipt_exists":(A/"failure.json").exists()},
      "adapter_validation":{"tensor_count":tensors,"nonfinite_tensors":nonfinite,"reload_success":r["reload_success"],"reload_matches":r["validation"]==p,"reload_peak_allocated_gib":r["peak_allocated_gib"]},
      "artifacts":{"metrics_sha256":sha(A/"metrics.json"),"generations_sha256":sha(A/"generations.json"),"reload_validation_sha256":sha(A/"reload_validation.json"),"adapter_sha256":m["adapter_sha256"]},
      "evaluation_boundary":{"sealed_canary_run":False,"prior_full_suite_run":False,"independent_holdout_run":False,"independent_holdout_prompts_loaded":False,"independent_holdout_references_loaded":False},
      "decision":{"accepted_local_smoke":False,"aggregate_semantic_at_least_24":p["semantic_exact"]>=24,"strict_exact_at_least_22":p["exact"]>=22,"sealed_canary_allowed":False,"independent_holdout_allowed":False,"merge_allowed":False,"scale_allowed":False,"rl_allowed":False,"next_action":"Preserve v11 and the v15 semantic signal. Close data-family expansion; design a method-level preservation intervention such as loss weighting, adapter composition, or staged training before another smoke."}
    }
    md=f"""# Schedule Isolation Preservation SFT Smoke v15 Result

V15 is stable and reaches the strongest semantic result, but fails the frozen
strict/preservation gate:

- aggregate exact / semantic: {p['exact']}/32 / {p['semantic_exact']}/32;
- numeric exact / semantic: {p['by_family']['capability_preservation_numeric']['exact']}/16 / {p['by_family']['capability_preservation_numeric']['semantic_exact']}/16;
- choice: {p['by_family']['capability_preservation_choice']['semantic_exact']}/8; process: {p['by_family']['semantic_arithmetic_process']['semantic_exact']}/8;
- versus v11: semantic +{p['semantic_exact']-old['semantic_exact']}, numeric semantic +{p['by_family']['capability_preservation_numeric']['semantic_exact']-old['by_family']['capability_preservation_numeric']['semantic_exact']}, strict {p['exact']-old['exact']}, choice -1;
- finite losses/tensors: {len(losses)}/{tensors}; reload exact: {r['validation']==p}.

Seven schedule exposures produce real semantic gain but contract interference.
Reject v15 for canary/holdout/promotion. Close further family-data expansion
and move to a method-level preservation intervention. V11 remains current.

Holdout prompts/references remain unread.
"""
    out=ROOT/"docs/results"; out.mkdir(parents=True,exist_ok=True)
    (out/"schedule_isolation_preservation_sft_smoke_v15.public.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    (out/"schedule_isolation_preservation_sft_smoke_v15.md").write_text(md)
    print(json.dumps({"passed":False,"post_exact":p["exact"],"post_semantic":p["semantic_exact"],"numeric_semantic":p["by_family"]["capability_preservation_numeric"]["semantic_exact"],"sealed_canary_allowed":False,"family_matrix_complete":True},sort_keys=True))

if __name__=="__main__": main()
