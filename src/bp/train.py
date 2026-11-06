"""Ciclo canónico: AdamW + warmup/coseno por lote, clipping, early stopping y MLflow."""
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn

from .dataset import train_loader
from .metrics import per_patient_metrics
from .models import BUILDERS
from .paths import ARTIFACTS, DATA_CARD, MLFLOW_URI

CKPT_DIR = ARTIFACTS / "checkpoints"
PROGRESS_LOG = ARTIFACTS.parent / "cache" / "progress.log"
DEFAULT_CFG = {"lr": 1e-3, "weight_decay": 1e-2, "batch_size": 128, "epochs": 30, "patience": 5,
               "clip": 1.0, "seed": 0, "eval_batch_size": 1024, "amp": False, "micro_batch": None}


def warmup_cosine(opt, steps_per_epoch: int, epochs: int):
    """Warmup lineal de 1 época + coseno; scheduler.step() se llama por lote."""
    total, warm = steps_per_epoch * epochs, steps_per_epoch
    f = lambda t: (t + 1) / warm if t < warm else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, total - warm)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def build_model(builder: str, n_channels: int, seed: int = 0, **kw) -> nn.Module:
    torch.manual_seed(seed)  # mismo theta0 para todas las variantes
    return BUILDERS[builder](n_channels, **kw)


@torch.no_grad()
def predict_norm(model: nn.Module, X: torch.Tensor, batch_size: int = 1024, amp: bool = False) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and X.is_cuda):
            out.append(model(X[i:i + batch_size]).float())
    return torch.cat(out)


def evaluate(model, data: dict, y_mean, y_std, batch_size: int = 1024, amp: bool = False) -> dict:
    """val_loss (MSE estandarizado) y MAE en mmHg, agrupado."""
    pn = predict_norm(model, data["X"], batch_size, amp)
    loss = float(((pn - data["yn"]) ** 2).mean())
    pred = pn.cpu().numpy() * y_std + y_mean
    mae = np.abs(pred - data["y"]).mean(0)
    return {"loss": loss, "mae_pas": float(mae[0]), "mae_pad": float(mae[1]), "pred": pred}


def param_groups(model: nn.Module, cfg: dict):
    if hasattr(model, "param_groups"):
        return model.param_groups(cfg)
    return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": cfg["lr"]}]


def export_state(model: nn.Module) -> dict:
    return model.export_state() if hasattr(model, "export_state") else copy.deepcopy(model.state_dict())


def load_state(model: nn.Module, state: dict) -> None:
    if hasattr(model, "load_exported_state"):
        model.load_exported_state(state)
    else:
        model.load_state_dict(state)


def sanity_checks(builder: str, data: dict, n_channels: int, cfg: dict, device, model_kw=None, overfit_steps: int = 300,
                  n_overfit: int = 64, n_init: int = 1024, eval_bs: int = 1024, amp: bool = False) -> dict:
    """Chequeos previos: MSE inicial ~ 1, sobreajuste de 64 ejemplos, gradientes finitos, eval() determinista."""
    model_kw = model_kw or {}
    model = build_model(builder, n_channels, cfg["seed"], **model_kw).to(device)
    xb, yb = data["X"][:n_overfit], data["yn"][:n_overfit]
    res = {}
    # 1. pérdida inicial (sobre n_init ejemplos de train, en eval)
    sel = torch.linspace(0, len(data["X"]) - 1, min(n_init, len(data["X"])), device=data["X"].device).long()  # muestra espaciada, no solo los primeros pacientes
    res["init_mse"] = float(((predict_norm(model, data["X"][sel], eval_bs, amp) - data["yn"][sel]) ** 2).mean())
    # 2-3. sobreajustar un lote de 64 y mirar gradientes (Dropout apagado en este chequeo)
    model.train()
    opt = torch.optim.AdamW(param_groups(model, cfg), weight_decay=cfg["weight_decay"])
    loss_fn = nn.MSELoss()
    for step in range(overfit_steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            out = model(xb)
        loss = loss_fn(out.float(), yb)
        loss.backward()
        if step == 3:  # con la cabeza (y B de LoRA) inicializadas en 0, el gradiente llega a todas las capas a partir del tercer paso
            norms = {n_: p.grad.norm().item() for n_, p in model.named_parameters() if p.requires_grad and p.grad is not None}
            res["grad_all_finite"] = all(math.isfinite(v) for v in norms.values())
            res["grad_zero_layers"] = [k for k, v in norms.items() if v == 0.0]
            res["grad_n_layers"] = len(norms)
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["clip"])
        opt.step()
        if loss.item() < 0.05:
            break
    res["overfit_loss"], res["overfit_steps"] = float(loss.item()), step + 1
    # 4. eval() determinista
    a, b = predict_norm(model, xb, eval_bs, amp), predict_norm(model, xb, eval_bs, amp)
    res["eval_deterministic"] = bool(torch.equal(a, b))
    return res


def fit(builder: str, inputs: list[str], train: dict, val: dict, y_mean, y_std, cfg: dict | None = None,
        run_name: str | None = None, model_kw: dict | None = None, mlflow_log: bool = True, device=None,
        post_build=None) -> dict:
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    model_kw = model_kw or {}
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_channels = train["X"].shape[1]
    model = build_model(builder, n_channels, cfg["seed"], **model_kw).to(device)
    if post_build is not None:  # p. ej. inicializar la cabeza con la del probe lineal
        post_build(model)
    micro = cfg["micro_batch"] or cfg["batch_size"]
    accum = cfg["batch_size"] // micro  # batch efectivo = micro * accum
    loader = train_loader(train, micro, cfg["seed"])
    opt = torch.optim.AdamW(param_groups(model, cfg), weight_decay=cfg["weight_decay"])
    sched = warmup_cosine(opt, len(loader) // accum, cfg["epochs"])
    loss_fn = nn.MSELoss()
    use_amp = bool(cfg["amp"]) and device.type == "cuda"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    run_name = run_name or f"{builder}_{'+'.join(inputs)}"

    run = None
    if mlflow_log:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("mimic_bp")
        run = mlflow.start_run(run_name=run_name)
        mlflow.log_params({"arch": builder, "inputs": "+".join(inputs), "n_params_trainable":
                           sum(p.numel() for p in model.parameters() if p.requires_grad),
                           **{k: v for k, v in cfg.items()}, **{f"model_{k}": v for k, v in model_kw.items()}})
    history, best_loss, best_epoch, best_state, bad = [], float("inf"), -1, None, 0
    t0 = time.time()
    try:
        for epoch in range(cfg["epochs"]):
            model.train()
            total, count, norms = 0.0, 0, []
            opt.zero_grad(set_to_none=True)
            for i, (xb, yb) in enumerate(loader):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    out = model(xb)
                loss = loss_fn(out.float(), yb)
                (loss / accum).backward()
                total += loss.item() * len(xb)
                count += len(xb)
                if (i + 1) % accum == 0:
                    norms.append(nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg["clip"]).item())
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
            v = evaluate(model, val, y_mean, y_std, cfg["eval_batch_size"], use_amp)
            rec = {"epoch": epoch, "train_loss": total / count, "val_loss": v["loss"],
                   "val_mae_pas_mmHg": v["mae_pas"], "val_mae_pad_mmHg": v["mae_pad"],
                   "grad_norm": float(np.mean(norms)), "learning_rate": opt.param_groups[0]["lr"]}
            history.append(rec)
            if run is not None:
                mlflow.log_metrics({k: x for k, x in rec.items() if k != "epoch"}, step=epoch)
            line = (f"[{run_name}] ep {epoch:02d} train {rec['train_loss']:.4f} val {rec['val_loss']:.4f} "
                    f"MAE {rec['val_mae_pas_mmHg']:.2f}/{rec['val_mae_pad_mmHg']:.2f} gn {rec['grad_norm']:.2f} "
                    f"t {time.time() - t0:.0f}s")
            print(line, flush=True)
            with open(PROGRESS_LOG, "a", encoding="utf-8") as fh:  # seguimiento de corridas largas desde fuera del notebook
                fh.write(line + "\n")
            if v["loss"] < best_loss:
                best_loss, best_epoch, bad = v["loss"], epoch, 0
                best_state = export_state(model)
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break
        load_state(model, best_state)
        final = evaluate(model, val, y_mean, y_std, cfg["eval_batch_size"], use_amp)
        ckpt = CKPT_DIR / f"{run_name}.pt"
        torch.save({"state": best_state, "builder": builder, "inputs": inputs, "model_kw": model_kw,
                    "cfg": cfg, "best_epoch": best_epoch}, ckpt)
        if run is not None:
            mlflow.log_metrics({"best_val_loss": best_loss, "best_epoch": best_epoch})
            mlflow.log_artifact(str(ckpt))
            if DATA_CARD.exists():
                mlflow.log_artifact(str(DATA_CARD))
    finally:
        if run is not None:
            mlflow.end_run()
    return {"run_name": run_name, "builder": builder, "inputs": inputs, "model_kw": model_kw, "cfg": cfg,
            "best_epoch": best_epoch, "best_val_loss": best_loss, "val_mae_pas": final["mae_pas"],
            "val_mae_pad": final["mae_pad"], "epochs_run": len(history), "seconds": time.time() - t0,
            "checkpoint": str(ckpt), "history": history}
