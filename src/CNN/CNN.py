# train_cnn3d_optuna.py
"""
Optuna(TPE)로 3D CNN 하이퍼파라미터 최적화
- merged .pt(dict: x,y,xc,yc,time_keys)로드
- FNO 코드와 순서/흐름 유사: device→정규화(fit on full)→split→dataset→objective 학습→val L2 반환
- 최적 파라미터로 재학습 후 비교 그림 저장(역정규화)
"""

import sys
sys.path.append("./")

import math
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import optuna

from neuraloperator.neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuraloperator.neuralop import LpLoss

# =========================
# Dataset / Helpers
# =========================
class CustomDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x, self.y = x, y  # x:(N,C,nx,ny,nt) y:(N,1,nx,ny,nt)
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, idx): return {"x": self.x[idx], "y": self.y[idx]}

def to_3d_order(x):
    # (B,C,nx,ny,nt) -> (B,C,nt,nx,ny)
    return x.permute(0,1,4,2,3).contiguous()

def from_3d_order(x):
    # (B,C,nt,nx,ny) -> (B,C,nx,ny,nt)
    return x.permute(0,1,3,4,2).contiguous()

@torch.no_grad()
def plot_compare(pred_phys, gt_phys, save_path, t_index=0):
    pi = pred_phys[0,0,:,:,t_index].cpu().numpy()
    gi = gt_phys[0,0,:,:,t_index].cpu().numpy()
    err = abs(pi - gi)
    fig = plt.figure(figsize=(12,3.8), constrained_layout=True)
    ax1 = fig.add_subplot(1,3,1); im1=ax1.imshow(gi); ax1.set_title("Ground Truth"); fig.colorbar(im1, ax=ax1)
    ax2 = fig.add_subplot(1,3,2); im2=ax2.imshow(pi); ax2.set_title("Prediction");  fig.colorbar(im2, ax=ax2)
    ax3 = fig.add_subplot(1,3,3); im3=ax3.imshow(err);ax3.set_title("Abs Error");   fig.colorbar(im3, ax=ax3)
    for ax in (ax1,ax2,ax3): ax.set_xticks([]); ax.set_yticks([])
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200); plt.close(fig)
    print(f"[OK] Saved figure: {save_path}")

# =========================
# 3D CNN
# =========================
class SimpleCNN3D(nn.Module):
    def __init__(self, in_ch=9, out_ch=1, base=64, num_blocks=3, kt=3, ks=3, convs_per_block=2):
        super().__init__()
        k = (kt, ks, ks)
        p = (kt//2, ks//2, ks//2)
        ch_in = in_ch
        layers = []
        for _ in range(num_blocks):
            for _ in range(convs_per_block):
                layers.append(nn.Conv3d(ch_in, base, kernel_size=k, padding=p))
                layers.append(nn.ReLU(inplace=True))
                ch_in = base  # 첫 conv 이후에는 in/out 채널 동일
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv3d(base, out_ch, kernel_size=1)

    def forward(self, x):  # x:(B,C,nt,nx,ny)
        h = self.body(x)
        y = self.head(h)  # (B,1,nt,nx,ny)
        return y

# =========================
# Scheduler (FNO와 유사)
# =========================
class CappedCosineAnnealingWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_max, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0; self.T_max = T_max; self.T_mult = T_mult; self.eta_min = eta_min
        self.T_i = T_0; self.last_restart = 0
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        t = self.last_epoch - self.last_restart
        if t >= self.T_i:
            self.last_restart = self.last_epoch
            self.T_i = min(self.T_i * self.T_mult, self.T_max)
            t = 0
        return [ self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * t / self.T_i)) / 2
                 for base_lr in self.base_lrs ]

# =========================
# Data utilities
# =========================
def load_merged(merged_pt_path: str):
    bundle = torch.load(merged_pt_path, map_location="cpu")
    X = bundle["x"].float()  # (N,9,nx,ny,nt)
    Y = bundle["y"].float()  # (N,1,nx,ny,nt)
    print("Data:", tuple(X.shape), tuple(Y.shape))
    return X, Y

# =========================
# Main (Optuna TPE)
# =========================
def main():
    merged_pt_path = "./src/preprocessing/merged.pt"
    out_dir = Path("./runs/cnn3d_optuna"); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 데이터 로드
    X, Y = load_merged(merged_pt_path)
    X = X.to(device); Y = Y.to(device)

    # 2) 정규화 (FNO 흐름과 맞춤: 전체 데이터로 fit)
    in_norm  = UnitGaussianNormalizer(mean=X, std=X, dim=[0,2,3,4], eps=1e-6);  in_norm.fit(X)
    out_norm = UnitGaussianNormalizer(mean=Y, std=Y, dim=[0,2,3,4], eps=1e-6); out_norm.fit(Y)

    # 3) split & dataset
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_ds = CustomDataset(train_x, train_y)
    test_ds  = CustomDataset(test_x,  test_y)

    # 고정 요소
    N_EPOCHS = 10000
    EVAL_INTERVAL = 1

    l2 = LpLoss(d=3, p=2)

    def objective(trial: "optuna.trial.Trial"):
        # ---- 하이퍼파라미터 공간 (요청된 대상) ----
        depth = trial.suggest_int("depth", 2, 5)
        base  = trial.suggest_categorical("base_channels", [32, 48, 64, 96, 128])
        kt    = trial.suggest_categorical("kernel_t", [1,3,5])
        ks    = trial.suggest_categorical("kernel_s", [3,5])
        train_batch = trial.suggest_categorical("train_batch_size", [16, 32, 64, 128])
        weight_decay = trial.suggest_float("l2_weight", 1e-8, 1e-3, log=True)
        init_lr = trial.suggest_float("initial_lr", 1e-4, 1e-3, log=True)

        # ---- DataLoader (test는 전체 한 배치) ----
        train_loader = DataLoader(train_ds, batch_size=train_batch, shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=len(test_ds), shuffle=False)

        # ---- 모델/최적화/스케줄러 ----
        model = SimpleCNN3D(in_ch=X.shape[1], out_ch=Y.shape[1], base=base, depth=depth, kt=kt, ks=ks).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
        sched = CappedCosineAnnealingWarmRestarts(optim, T_0=10, T_max=160, T_mult=2, eta_min=1e-6)

        # ---- 학습 루프 (val L2 최소화) ----
        best = float("inf"); patience=320; bad=0
        for ep in range(1, N_EPOCHS+1):
            model.train()
            for batch in train_loader:
                x = to_3d_order(batch["x"]).to(device)  # (B,C,nt,nx,ny)
                y = to_3d_order(batch["y"]).to(device)
                pred = model(x)
                loss = l2(pred, y)
                optim.zero_grad(); loss.backward(); optim.step()
            # step scheduler
            sched.step()

            if ep % EVAL_INTERVAL == 0:
                model.eval()
                with torch.no_grad():
                    tb = next(iter(test_loader))
                    x = to_3d_order(tb["x"]).to(device)
                    y = to_3d_order(tb["y"]).to(device)
                    pred = model(x)
                    val = l2(pred, y).item()

                if val < best:
                    best = val; bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break

        return best

    # TPE Sampler
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=10)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=40, show_progress_bar=True)

    print("\n=== Optuna Best (3D CNN) ===")
    print("Value:", study.best_value)
    print("Params:", study.best_params)

    # 4) Best로 재학습 후 그림 저장(역정규화)
    bp = study.best_params
    depth = bp["depth"]; base = bp["base_channels"]; kt = bp["kernel_t"]; ks = bp["kernel_s"]
    train_batch = bp["train_batch_size"]; weight_decay = bp["l2_weight"]; init_lr = bp["initial_lr"]

    train_loader = DataLoader(train_ds, batch_size=train_batch, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=len(test_ds), shuffle=False)

    model = SimpleCNN3D(in_ch=X.shape[1], out_ch=Y.shape[1], base=base, depth=depth, kt=kt, ks=ks).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    sched = CappedCosineAnnealingWarmRestarts(optim, T_0=10, T_max=160, T_mult=2, eta_min=1e-6)

    best_model_path = out_dir / "cnn3d_best.pt"
    best = float("inf"); patience=320; bad=0
    for ep in range(1, 10000+1):
        model.train()
        for batch in train_loader:
            x = to_3d_order(batch["x"]).to(device)
            y = to_3d_order(batch["y"]).to(device)
            pred = model(x)
            loss = l2(pred, y)
            optim.zero_grad(); loss.backward(); optim.step()
        sched.step()

        # val
        model.eval()
        with torch.no_grad():
            tb = next(iter(test_loader))
            x = to_3d_order(tb["x"]).to(device)
            y = to_3d_order(tb["y"]).to(device)
            pred = model(x)
            val = l2(pred, y).item()

        if val < best:
            best = val; bad = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            bad += 1
            if bad >= patience:
                break

    print(f"[DONE] Best val L2: {best:.6f} | saved: {best_model_path}")

    # 그림 저장 (역정규화)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        xb = next(iter(test_loader))
        x = xb["x"][0:1].to(device)  # (1,C,nx,ny,nt) normalized? (우리는 전체정규화로 학습함)
        y = xb["y"][0:1].to(device)
        p = from_3d_order(model(to_3d_order(x)))  # (1,1,nx,ny,nt) normalized
        # 역정규화
        p_phys = out_norm.inverse_transform(p.detach().cpu())
        g_phys = out_norm.inverse_transform(y.detach().cpu())
    plot_compare(p_phys, g_phys, save_path=str(out_dir / "cnn3d_optuna_compare.png"), t_index=0)

if __name__ == "__main__":
    main()