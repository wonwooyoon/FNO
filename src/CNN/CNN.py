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
import numpy as np 
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import optuna

from neuraloperator.neuralop.data.transforms.normalizers import UnitGaussianNormalizer, MinMaxNormalizer
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
def plot_compare(pred_phys, gt_phys, save_path, sample_num=0, t_indices=(0,1,2,3,4)):

    # 데이터 수집
    pis, gis, ers = [], [], []
    for t in t_indices:
        pi = pred_phys[sample_num,0,:,:,t].cpu().numpy()
        gi = gt_phys[sample_num,0,:,:,t].cpu().numpy()
        pis.append(pi); gis.append(gi); ers.append(np.abs(pi-gi))

    # GT/Pred 공용 범위
    vmin = min(np.min(pis), np.min(gis))
    vmax = max(np.max(pis), np.max(gis))

    ncols = len(t_indices)
    # 가로 폭은 시점 개수에 비례해서 늘림
    fig_h = 3.6 * 3
    fig_w = 1.8 * ncols + 1.6  # 오른쪽 컬러바 폭 고려
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)

    # GridSpec: 3행(GT/Pred/Error) x ncols(시간), 오른쪽에 컬러바 2칸
    gs = GridSpec(nrows=3, ncols=ncols+2, figure=fig,
                  width_ratios=[*([1]*ncols), 0.05, 0.05],
                  height_ratios=[1,1,1], wspace=0.08, hspace=0.12)

    axes_gt, axes_pred, axes_err = [], [], []
    for r in range(3):
        row_axes = []
        for c in range(ncols):
            ax = fig.add_subplot(gs[r, c])
            row_axes.append(ax)
        if r == 0: axes_gt = row_axes
        elif r == 1: axes_pred = row_axes
        else: axes_err = row_axes

    # 플롯
    ims_gt, ims_pred, ims_err = [], [], []
    for c, (pi, gi, er, t) in enumerate(zip(pis, gis, ers, t_indices)):
        im1 = axes_gt[c].imshow(gi, vmin=vmin, vmax=vmax)
        im2 = axes_pred[c].imshow(pi, vmin=vmin, vmax=vmax)
        im3 = axes_err[c].imshow(er)
        ims_gt.append(im1); ims_pred.append(im2); ims_err.append(im3)

        axes_gt[c].set_title(f"GT (t={t})")
        if c == 0:
            axes_pred[c].set_ylabel("Prediction", rotation=90, labelpad=20)
            axes_err[c].set_ylabel("Abs Error", rotation=90, labelpad=20)

    # 축 꾸미기
    for row in (axes_gt, axes_pred, axes_err):
        for ax in row:
            ax.set_xticks([]); ax.set_yticks([])

    # 컬러바(오른쪽 2칸 사용)
    cax_main = fig.add_subplot(gs[:, ncols])     # GT/Pred 공용
    cax_err  = fig.add_subplot(gs[:, ncols+1])   # Error 전용
    # 공용 컬러바는 GT/Pred 중 아무거나 핸들로 사용
    cb_main = fig.colorbar(ims_gt[0], cax=cax_main)
    cb_main.set_label("Value")
    cb_err  = fig.colorbar(ims_err[0], cax=cax_err)
    cb_err.set_label("Abs Error")

    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved multi-time figure: {save_path}")

# =========================
# 3D CNN
# =========================
class SimpleCNN3D(nn.Module):
    def __init__(self, in_ch=9, out_ch=1, base=64, depth=3, kt=3, ks=3, convs_per_block=1):
        super().__init__()
        k = (kt, ks, ks)
        p = (kt//2, ks//2, ks//2)
        ch_in = in_ch
        layers = []
        for _ in range(depth):
            for _ in range(convs_per_block):
                layers.append(nn.Conv3d(ch_in, base, kernel_size=k, padding=p))
                layers.append(nn.GELU())
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
    out_dir = Path("./src/CNN/output"); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 데이터 로드
    X, Y = load_merged(merged_pt_path)
    X = X.to(device); Y = Y.to(device)

    # Log10 기반으로 가져온 농도값 수정
    Y = 10 ** Y
    Y[:, :, 14:18, 14:18, :] = 0

    # 2) 정규화 (FNO 흐름과 맞춤: 전체 데이터로 fit)
    # in_norm  = UnitGaussianNormalizer(mean=X, std=X, dim=[0,2,3,4], eps=1e-6);  in_norm.fit(X)
    # out_norm = UnitGaussianNormalizer(mean=Y, std=Y, dim=[0,2,3,4], eps=1e-6); out_norm.fit(Y)
    in_norm  = MinMaxNormalizer(data_min=X, data_max=X, dim=[0,2,3,4], eps=1e-6);  in_norm.fit(X)
    out_norm = MinMaxNormalizer(data_min=Y, data_max=Y, dim=[0,2,3,4], eps=1e-6); out_norm.fit(Y)

    X = in_norm.transform(X); Y = out_norm.transform(Y)  # 정규화된 데이터

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
        # depth = trial.suggest_int("depth", 2, 5)
        # base  = trial.suggest_categorical("base_channels", [32, 48, 64, 96, 128])
        # kt    = trial.suggest_categorical("kernel_t", [1,3,5])
        # ks    = trial.suggest_categorical("kernel_s", [3,5])
        # train_batch = trial.suggest_categorical("train_batch_size", [16, 32, 64, 128])
        # weight_decay = trial.suggest_float("l2_weight", 1e-8, 1e-3, log=True)
        # init_lr = trial.suggest_float("initial_lr", 1e-4, 1e-3, log=True)

        depth = trial.suggest_int("depth", 2, 4)
        base  = trial.suggest_categorical("base_channels", [32, 64])
        kt    = trial.suggest_categorical("kernel_t", [1,3,5])
        ks    = trial.suggest_categorical("kernel_s", [3,5])
        train_batch = trial.suggest_categorical("train_batch_size", [32, 64])
        weight_decay = trial.suggest_float("l2_weight", 1e-8, 1e-3, log=True)
        init_lr = trial.suggest_float("initial_lr", 1e-4, 1e-3, log=True)

        # ---- DataLoader (test는 전체 한 배치) ----
        train_loader = DataLoader(train_ds, batch_size=train_batch, shuffle=True)
        test_loader  = DataLoader(test_ds,  batch_size=len(test_ds), shuffle=False)

        # ---- 모델/최적화/스케줄러 ----
        model = SimpleCNN3D(in_ch=X.shape[1], out_ch=Y.shape[1], base=base, depth=depth, kt=kt, ks=ks).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
        # sched = CappedCosineAnnealingWarmRestarts(optim, T_0=10, T_max=160, T_mult=2, eta_min=1e-6)
        sched = torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.1)

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
                    val = l2(pred, y).item() / y.size(0)
                print(f"Epoch {ep:04d} | Val L2: {val:.6f}")

                if val < best:
                    best = val; bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        break

        return best

    # # TPE Sampler
    # sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=10)
    # study = optuna.create_study(direction="minimize", sampler=sampler)
    # study.optimize(objective, n_trials=10, show_progress_bar=True)

    # print("\n=== Optuna Best (3D CNN) ===")
    # print("Value:", study.best_value)
    # print("Params:", study.best_params)

    # # 4) Best로 재학습 후 그림 저장(역정규화)
    # bp = study.best_params
    
    bp = {'depth': 3, 'base_channels': 64, 'kernel_t': 5, 'kernel_s': 5, 'train_batch_size': 64, 'l2_weight': 7.26480307482672e-05, 'initial_lr': 0.00015802131864103882}
    
    depth = bp["depth"]; base = bp["base_channels"]; kt = bp["kernel_t"]; ks = bp["kernel_s"]
    train_batch = bp["train_batch_size"]; weight_decay = bp["l2_weight"]; init_lr = bp["initial_lr"]

    train_loader = DataLoader(train_ds, batch_size=train_batch, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=len(test_ds), shuffle=False)

    model = SimpleCNN3D(in_ch=X.shape[1], out_ch=Y.shape[1], base=base, depth=depth, kt=kt, ks=ks).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    sched = CappedCosineAnnealingWarmRestarts(optim, T_0=10, T_max=80, T_mult=2, eta_min=1e-6)

    print(f'number of parameters: {sum(p.numel() for p in model.parameters())}')

    best_model_path = out_dir / "cnn3d_best.pt"
    best = float("inf"); patience=80; bad=0
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
            val = l2(pred, y).item() / y.size(0)
            print(f"Epoch {ep:04d} | Val L2: {val:.6f}")

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
        x = xb["x"].to(device) 
        y = xb["y"].to(device)
        p = from_3d_order(model(to_3d_order(x)))  # (1,1,nx,ny,nt) normalized

        # 역정규화
        p_phys = out_norm.inverse_transform(p).detach().cpu()
        g_phys = out_norm.inverse_transform(y).detach().cpu()

    plot_compare(p_phys, g_phys, save_path=str(out_dir / "cnn3d_optuna_compare.png"), sample_num=8, t_indices=(0, 4, 8, 12, 16))

if __name__ == "__main__":
    main()