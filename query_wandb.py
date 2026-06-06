import wandb
api = wandb.Api()
projects = ["3DV-Project/hand-head-training", "roman-zberg-uzh-organization-org/hand-head-training"]
for proj in projects:
    try:
        runs = api.runs(proj)
        print(f"=== Project: {proj} ===")
        for run in list(runs)[:15]:
            print(f"Run: {run.name} ({run.id}) | State: {run.state} | Duration: {run.duration}s | Created: {run.created_at} | Steps: {run.summary.get('_step')}")
            cfg = run.config
            training_cfg = cfg.get("training", {})
            model_cfg = cfg.get("model", {})
            debug_cfg = cfg.get("debug", {})
            data_cfg = cfg.get("data", {})
            print(f"  Batch Size: {training_cfg.get('batch_size')} | enable_gs: {model_cfg.get('enable_gs')} | debug.enabled: {debug_cfg.get('enabled')} | val_split: {data_cfg.get('val_split')}")
    except Exception as e:
        print(f"Error querying {proj}: {e}")
