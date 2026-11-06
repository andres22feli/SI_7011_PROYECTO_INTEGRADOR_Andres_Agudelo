"""Arquitecturas: MLP, GRU con patching y TimesFM (probe lineal / LoRA)."""
import torch
import torch.nn as nn

N_SAMPLES = 3750
PATCH = 25


def _zero_head(layer: nn.Linear) -> None:
    """Capa de salida en 0: la predicción inicial es la media de train (MSE inicial ~ 1)."""
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    """Linear(bias=False) -> BatchNorm1d -> ReLU -> [Dropout]; 512 -> 256 -> 128 -> 2."""

    def __init__(self, n_channels: int, hidden=(512, 256, 128), dropout: float = 0.0):
        super().__init__()
        layers, d = [nn.Flatten()], n_channels * N_SAMPLES
        for h in hidden:
            layers += [nn.Linear(d, h, bias=False), nn.BatchNorm1d(h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        _zero_head(self.net[-1])

    def forward(self, x):
        return self.net(x)


class GRUPatch(nn.Module):
    """Ventanas de 25 muestras (T=150 pasos) -> GRU(64, 2 capas) -> Linear sobre h_T."""

    def __init__(self, n_channels: int, hidden: int = 64, layers: int = 2, patch: int = PATCH):
        super().__init__()
        self.patch, self.c = patch, n_channels
        self.gru = nn.GRU(patch * n_channels, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 2)
        _zero_head(self.head)

    def forward(self, x):  # x: (B, C, 3750)
        b, c, t = x.shape
        x = x.reshape(b, c, t // self.patch, self.patch).permute(0, 2, 1, 3).reshape(b, t // self.patch, c * self.patch)
        _, h = self.gru(x)
        return self.head(h[-1])


def build_mlp(n_channels: int, **kw) -> nn.Module:
    return MLP(n_channels, **kw)


def build_gru(n_channels: int, **kw) -> nn.Module:
    return GRUPatch(n_channels, **kw)


TIMESFM_ID = "google/timesfm-2.5-200m-transformers"
TIMESFM_PATCH = 32
TIMESFM_DIM = 1280


class TimesFMReg(nn.Module):
    """Backbone TimesFM 2.5 (200M) + cabeza Linear(C*1280, 2).

    mode='probe': backbone congelado. mode='lora': adaptadores LoRA (r=8, alpha=16) en q/k/v/o de la atención.
    Cada canal pasa por el backbone; sus embeddings (media sobre parches) se concatenan antes de la cabeza.
    La señal de 3750 muestras se rellena por la izquierda hasta 3776 (118 parches de 32) con máscara de relleno.
    """

    def __init__(self, n_channels: int, mode: str = "probe", lora_r: int = 8, lora_alpha: int = 16,
                 lora_dropout: float = 0.0, revision: str | None = None):
        super().__init__()
        from transformers import TimesFm2_5Model
        assert mode in ("probe", "lora")
        self.mode, self.c = mode, n_channels
        backbone = TimesFm2_5Model.from_pretrained(TIMESFM_ID, revision=revision)
        for p in backbone.parameters():
            p.requires_grad = False
        if mode == "lora":
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
            backbone = get_peft_model(backbone, cfg)
        self.backbone = backbone
        self.head = nn.Linear(n_channels * TIMESFM_DIM, 2)
        _zero_head(self.head)

    def embed(self, x):  # x: (B, C, 3750) -> (B, C*1280)
        b, c, t = x.shape
        flat = x.reshape(b * c, t)
        pad = (-t) % TIMESFM_PATCH
        mask = torch.cat([torch.ones(b * c, pad, dtype=torch.long, device=x.device),
                          torch.zeros(b * c, t, dtype=torch.long, device=x.device)], 1)
        flat = torch.cat([torch.zeros(b * c, pad, dtype=flat.dtype, device=x.device), flat], 1)
        if self.mode == "probe":
            with torch.no_grad():
                h = self.backbone(flat, mask).last_hidden_state
        else:
            h = self.backbone(flat, mask).last_hidden_state
        return h.float().mean(1).reshape(b, c * TIMESFM_DIM)

    def forward(self, x):
        return self.head(self.embed(x))

    def param_groups(self, cfg):
        groups = [{"params": list(self.head.parameters()), "lr": cfg.get("head_lr", cfg["lr"])}]
        lora = [p for p in self.backbone.parameters() if p.requires_grad]
        if lora:
            groups.append({"params": lora, "lr": cfg.get("backbone_lr", cfg["lr"])})
        return groups

    def export_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if k.startswith("head.") or "lora_" in k}

    def load_exported_state(self, state):
        missing = self.load_state_dict(state, strict=False)
        bad = [k for k in missing.unexpected_keys] + [k for k in missing.missing_keys
                                                       if k.startswith("head.") or "lora_" in k]
        if bad:
            raise RuntimeError(f"estado TimesFM incompatible: {bad[:5]}")


class LinearProbe(nn.Module):
    """Cabeza lineal sobre embeddings TimesFM precalculados (n, C*1280)."""

    def __init__(self, dim: int):
        super().__init__()
        self.head = nn.Linear(dim, 2)
        _zero_head(self.head)

    def forward(self, x):
        return self.head(x)


def build_timesfm(n_channels: int, **kw) -> nn.Module:
    return TimesFMReg(n_channels, **kw)


def build_probe_head(dim: int, **kw) -> nn.Module:
    return LinearProbe(dim)


BUILDERS = {"build_mlp": build_mlp, "build_gru": build_gru, "build_timesfm": build_timesfm,
            "build_probe_head": build_probe_head}
