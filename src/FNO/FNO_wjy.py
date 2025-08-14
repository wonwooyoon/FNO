# train_tfno.py
"""
- merged 결과물(.pt dict: x,y,xc,yc,time_keys)을 직접 로드
- 기존 기능/로직 유지하되, 하이퍼파라미터 탐색을 Grid → Optuna TPE로 변경
- 각 Trial에서: 모델/옵티마이저/스케줄러/Trainer 생성 → 학습 → val 손실 반환
- 최종적으로 best params로 1회 재학습하여 비교 그림 저장
"""

import sys
sys.path.append('./')

import math
import numpy as np
import torch
import optuna
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from neuraloperator.neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuraloperator.neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuraloperator.neuralop import LpLoss
from neuraloperator.neuralop.models import TFNO
from neuraloperator.neuralop import Trainer
from neuraloperator.neuralop.training import AdamW

# =========================
# Dataset
# =========================
class CustomDataset(Dataset):
    def __init__(self, input_tensor, output_tensor, meta_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.meta_tensor = meta_tensor
    def __len__(self):
        return self.input_tensor.shape[0]
    def __getitem__(self, idx):
        return {'x': self.input_tensor[idx], 'y': self.output_tensor[idx], 'meta': self.meta_tensor[idx]}

# =========================
# Learning Rate Scheduler (원본 유지)
# =========================
class CappedCosineAnnealingWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_max, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_max = T_max
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_i = T_0
        self.last_restart = 0
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        t = self.last_epoch - self.last_restart
        if t >= self.T_i:
            self.last_restart = self.last_epoch
            self.T_i = min(self.T_i * self.T_mult, self.T_max)
            t = 0
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * t / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]

# =========================
# Utilities
# =========================
def load_merged_tensors(merged_pt_path: str):
    bundle = torch.load(merged_pt_path, map_location="cpu")
    in_summation = bundle["x"].float()   # (N, 9, nx, ny, nt)
    out_summation = bundle["y"].float()  # (N, 1, nx, ny, nt)
    meta_summation = bundle["meta"].float()  # (N, 2)
    print("Loaded merged tensors:", tuple(in_summation.shape), tuple(out_summation.shape), tuple(meta_summation.shape))
    return in_summation, out_summation, meta_summation

def load_wjy_tensors(wjy_path: str):
    in_summation = np.load(wjy_path + "log_normalized_permeability_maps.npy").astype(np.float32)  # (N, nx, ny)
    out_summation = np.load(wjy_path + "co2_saturation_maps.npy").astype(np.float32)  # (N, nx, ny)
    meta_summation = np.load(wjy_path + "input.npy").astype(np.float32)  # (N, 10)

    in_summation = torch.from_numpy(in_summation)
    out_summation = torch.from_numpy(out_summation)
    meta_summation = torch.from_numpy(meta_summation)

    in_summation = in_summation.unsqueeze(1)  # (N, 1, nx, ny)
    out_summation = out_summation.unsqueeze(1)  # (N, 1, nx, ny)
    
    print("Loaded WJY tensors:", tuple(in_summation.shape), tuple(out_summation.shape), tuple(meta_summation.shape))

    return in_summation, out_summation, meta_summation


def build_model(n_modes, hidden_channels, n_layers, domain_padding, domain_padding_mode, device):
    model = TFNO(
        n_modes=n_modes,
        in_channels=1,
        out_channels=1,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        lifting_channel_ratio=2,
        projection_channel_ratio=2,
        positional_embedding='grid',
        domain_padding=domain_padding,
        domain_padding_mode=domain_padding_mode,
        film_layer=True,
        meta_dim=10
    ).to(device)
    return model

@torch.no_grad()
def plot_compare(pred_phys, gt_phys, save_path, sample_nums=(0,)):
    # sample_nums: tuple of sample indices to plot
    # pred_phys, gt_phys: (N, 1, nx, ny)

    n_samples = len(sample_nums)

    # 데이터 수집
    pis, gis, ers = [], [], []
    for s in sample_nums:
        pi = pred_phys[s, 0].cpu().numpy()  # (nx, ny)
        gi = gt_phys[s, 0].cpu().numpy()    # (nx, ny)
        pis.append(pi)
        gis.append(gi)
        ers.append(np.abs(pi - gi))

    # GT/Pred 공용 범위
    vmin = min(np.min(pis), np.min(gis))
    vmax = max(np.max(pis), np.max(gis))

    ncols = 1
    nrows = 3 * n_samples  # 3 rows per sample (GT/Pred/Error)
    fig_h = 3.6 * n_samples * 3
    fig_w = 4.5  # 오른쪽 컬러바 폭 고려
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)

    # GridSpec: (3 * n_samples)행 x 1열, 오른쪽에 컬러바 2칸
    gs = GridSpec(nrows=nrows, ncols=ncols + 2, figure=fig,
                  width_ratios=[1, 0.05, 0.05],
                  height_ratios=[1] * nrows, wspace=0.08, hspace=0.12)

    axes_gt, axes_pred, axes_err = [], [], []
    for s in range(n_samples):
        ax_gt = fig.add_subplot(gs[3 * s + 0, 0])
        ax_pred = fig.add_subplot(gs[3 * s + 1, 0])
        ax_err = fig.add_subplot(gs[3 * s + 2, 0])
        axes_gt.append(ax_gt)
        axes_pred.append(ax_pred)
        axes_err.append(ax_err)

    # 플롯
    ims_gt, ims_pred, ims_err = [], [], []
    for s in range(n_samples):
        im1 = axes_gt[s].imshow(gis[s], vmin=vmin, vmax=vmax)
        im2 = axes_pred[s].imshow(pis[s], vmin=vmin, vmax=vmax)
        im3 = axes_err[s].imshow(ers[s])
        ims_gt.append(im1)
        ims_pred.append(im2)
        ims_err.append(im3)

        axes_gt[s].set_title(f"GT (sample={sample_nums[s]})")
        axes_pred[s].set_ylabel("Prediction", rotation=90, labelpad=20)
        axes_err[s].set_ylabel("Abs Error", rotation=90, labelpad=20)

    # 축 꾸미기
    for ax in axes_gt + axes_pred + axes_err:
        ax.set_xticks([])
        ax.set_yticks([])

    # 컬러바(오른쪽 2칸 사용)
    cax_main = fig.add_subplot(gs[:, 1])     # GT/Pred 공용
    cax_err = fig.add_subplot(gs[:, 2])      # Error 전용
    cb_main = fig.colorbar(ims_gt[0], cax=cax_main)
    cb_main.set_label("Value")
    cb_err = fig.colorbar(ims_err[0], cax=cax_err)
    cb_err.set_label("Abs Error")

    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved figure: {save_path}")

# =========================
# Main (Optuna TPE)
# =========================
def main():
    # merged_pt_path = "./src/preprocessing/merged.pt"
    wjy_path = "./src/preprocessing/"

    # 1) 데이터 로드
    # in_summation, out_summation, meta_summation = load_merged_tensors(merged_pt_path)
    in_summation, out_summation, meta_summation = load_wjy_tensors(wjy_path)

    # 2) 디바이스
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    in_summation = in_summation.to(device)
    out_summation = out_summation.to(device)
    meta_summation = meta_summation.to(device)

    out_summation = 10 ** out_summation

    # 3) 정규화 (원본 방식 유지: 전체 데이터로 fit)
    in_normalizer  = UnitGaussianNormalizer(mean=in_summation,  std=in_summation,  dim=[0,2,3], eps=1e-6)
    out_normalizer = UnitGaussianNormalizer(mean=out_summation, std=out_summation, dim=[0,2,3], eps=1e-6)
    meta_normalizer = UnitGaussianNormalizer(mean=meta_summation, std=meta_summation, dim=[0], eps=1e-6)
    in_normalizer.fit(in_summation)
    out_normalizer.fit(out_summation)
    meta_normalizer.fit(meta_summation)
    processor = DefaultDataProcessor(in_normalizer, out_normalizer, meta_normalizer).to(device)

    # 4) train/test split
    train_in, test_in, train_out, test_out, train_meta, test_meta = train_test_split(
        in_summation, out_summation, meta_summation, test_size=0.2, random_state=42
    )
    train_dataset = CustomDataset(train_in, train_out, train_meta)
    test_dataset  = CustomDataset(test_in,  test_out, test_meta)

    # 고정 요소들
    domain_padding_mode_fixed = 'symmetric'
    N_EPOCHS = 10000
    EVAL_INTERVAL = 1

    # 5) Optuna 객체
    def objective(trial: "optuna.trial.Trial"):
        # ---- 하이퍼파라미터 공간 (원본 리스트를 카테고리컬로 유지) ----
        # n_modes = trial.suggest_categorical("n_modes",
        #     [(32,16,10), (16,16,10), (16,8,10), (32,16,5), (16,16,5), (16,8,5)]
        # )
        # hidden_channels = trial.suggest_categorical("hidden_channels", [16, 32, 64, 128])
        # n_layers = trial.suggest_int("n_layers", 3, 5)
        # domain_padding = trial.suggest_categorical("domain_padding", [[0.1,0.1,0.1], [0.125,0.25,0.4]])
        # train_batch_size = trial.suggest_categorical("train_batch_size", [16, 32, 64, 128])
        # l2_weight = trial.suggest_float("l2_weight", 1e-8, 1e-3, log=True)
        # initial_lr = trial.suggest_float("initial_lr", 1e-4, 1e-3, log=True)

        n_modes = trial.suggest_categorical("n_modes",
            [(16,16,5), (16,8,5)]
        )
        hidden_channels = trial.suggest_categorical("hidden_channels", [8])
        n_layers = trial.suggest_categorical("n_layers", [2])
        domain_padding = trial.suggest_categorical("domain_padding", [[0.125,0.25,0.4]])
        train_batch_size = trial.suggest_categorical("train_batch_size", [64])
        l2_weight = trial.suggest_float("l2_weight", 1e-8, 1e-3, log=True)
        initial_lr = trial.suggest_float("initial_lr", 1e-4, 1e-3, log=True)

        # ---- DataLoader (train만 hp 반영, test는 전체 한 배치) ----
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        test_loader  = {'test_dataloader': DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)}

        # ---- 모델/최적화/스케줄러/트레이너 ----
        model = build_model(n_modes, hidden_channels, n_layers, domain_padding, domain_padding_mode_fixed, device)
        optimizer = AdamW(model.parameters(), lr=initial_lr, weight_decay=l2_weight)
        scheduler = CappedCosineAnnealingWarmRestarts(optimizer, T_0=10, T_max=80, T_mult=2, eta_min=1e-6)

        l2loss = LpLoss(d=3, p=2)
        trainer = Trainer(
            model=model, n_epochs=N_EPOCHS, device=device,
            data_processor=processor, wandb_log=False,
            eval_interval=EVAL_INTERVAL, use_distributed=False, verbose=True
        )

        # ---- 학습 ----

        best_model_path = './src/FNO/output/optuna'

        trainer.train(
            train_loader=train_loader,
            test_loaders=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            regularizer=False,
            early_stopping=True,
            training_loss=l2loss,
            eval_losses={'l2': l2loss},
            save_best='test_dataloader_l2',
            save_dir=best_model_path
        )

        model.load_state_dict(torch.load(f'{best_model_path}/best_model_state_dict.pt', map_location=device,weights_only=False))
        model.eval()

        with torch.no_grad():
            xy = next(iter(test_loader))
            x = xy["x"].to(device)
            y = xy["y"].to(device)
            pred = model(x)
            val_l2 = l2loss(pred, y).item()
        
        return val_l2

    # # TPE Sampler (Bayesian)
    # sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=2)
    # study = optuna.create_study(direction="minimize", sampler=sampler)
    # study.optimize(objective, n_trials=2, show_progress_bar=True)

    # print("\n=== Optuna Best ===")
    # print("Value:", study.best_value)
    # print("Params:", study.best_params)

    # 6) Best로 재학습 후 비교 그림 저장
    #bp = study.best_params

    bp = {"n_modes": (10, 5), "hidden_channels": 12, "n_layers": 3, "domain_padding": [0.1, 0.1], "train_batch_size": 32, "l2_weight": 0, "initial_lr": 1e-4}
    best_model = build_model(
        bp["n_modes"], bp["hidden_channels"], bp["n_layers"],
        bp["domain_padding"], domain_padding_mode_fixed, device
    )

    print(f'number of parameters: {sum(p.numel() for p in best_model.parameters())}')

    optimizer = AdamW(best_model.parameters(), lr=bp["initial_lr"], weight_decay=bp["l2_weight"])
    # scheduler = CappedCosineAnnealingWarmRestarts(optimizer, T_0=10, T_max=80, T_mult=2, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.85)

    for param in best_model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)

    l2loss = LpLoss(d=3, p=2)
    trainer = Trainer(
        model=best_model, n_epochs=10000, device=device,
        data_processor=processor, wandb_log=False,
        eval_interval=1, use_distributed=False, verbose=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=bp["train_batch_size"], shuffle=True)
    test_loader  = {'test_dataloader': DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)}

    best_model_path = './src/FNO/output/final'

    trainer.train(
        train_loader=train_loader,
        test_loaders=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        early_stopping=True,
        training_loss=l2loss,
        eval_losses={'l2': l2loss},
        save_best='test_dataloader_l2',
        save_dir=best_model_path
    )
    
    # 그림 저장 (역정규화)
    best_model.load_state_dict(torch.load(f'{best_model_path}/best_model_state_dict.pt', map_location=device,weights_only=False))
    best_model.eval()

    with torch.no_grad():
        xb = next(iter(test_loader["test_dataloader"]))
        x = xb["x"].to(device) 
        y = xb["y"].to(device)
        meta = xb["meta"].to(device)
        p = best_model(in_normalizer.transform(x), meta_normalizer.transform(meta)) 

        # 역정규화
        p_phys = out_normalizer.inverse_transform(p).detach().cpu()
        g_phys = y.detach().cpu()

        sse = torch.sum((p_phys - g_phys) ** 2).item()
        g_mean = torch.mean(g_phys).item()
        sst = torch.sum((g_phys - g_mean) ** 2).item()

        r2_global = 1.0 - sse / sst
        print(f"R2 Global: {r2_global:.6f}")


    # 최종 그림
    plot_compare(p_phys, g_phys, save_path=str('./src/FNO/output/FNO_compare.png'), sample_nums=(0, 5, 10, 15, 20))

if __name__ == "__main__":
    main()