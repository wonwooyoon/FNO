# train_tfno.py
"""
- merged 결과물(예: ./src/preprocessing/combined.pt)을 로드해 TFNO 학습/평가/비교그림 생성
- 기능은 원본과 동일하게 유지:
  * UnitGaussianNormalizer(mean=tensor, std=tensor, ...) 방식 + fit 호출
  * Trainer 설정/하이퍼파라미터/스케줄러/평가/플로팅 로직 동일
  * save_best='test_dataloader_l2' 후 'TFNO_best_model_{idx}.pt' 로드 시도
- 변경점: merge 함수 호출 대신, 병합 산출물(.pt dict: x,y,xc,yc,time_keys)을 직접 로드
"""

import sys
sys.path.append('./')

import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from itertools import product
import matplotlib.pyplot as plt

from neuraloperator.neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuraloperator.neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuraloperator.neuralop import LpLoss, H1Loss
from neuraloperator.neuralop.models import TFNO
from neuraloperator.neuralop import Trainer
from neuraloperator.neuralop.training import AdamW


# =========================
# Dataset
# =========================
class CustomDataset(Dataset):
    def __init__(self, input_tensor, output_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
    def __len__(self):
        return self.input_tensor.shape[0]
    def __getitem__(self, idx):
        return {'x': self.input_tensor[idx], 'y': self.output_tensor[idx]}


# =========================
# Utilities
# =========================
def load_merged_tensors(merged_pt_path: str):
    """
    병합 산출물(.pt)에 저장된 dict에서 x, y를 로드해 반환.
    merged payload keys: {"x", "y", "xc", "yc", "time_keys"}
    """
    bundle = torch.load(merged_pt_path, map_location="cpu")
    in_summation = bundle["x"].float()   # (N, 9, nx, ny, nt)
    out_summation = bundle["y"].float()  # (N, 1, nx, ny, nt)
    print("Loaded merged tensors:", tuple(in_summation.shape), tuple(out_summation.shape))
    return in_summation, out_summation


def build_model(n_modes, hidden_channels, n_layers, domain_padding, domain_padding_mode, device):
    model = TFNO(
        n_modes=n_modes,
        in_channels=9,
        out_channels=1,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        lifting_channel_ratio=2,
        projection_channel_ratio=2,
        positional_embedding='grid',
        domain_padding=domain_padding,
        domain_padding_mode=domain_padding_mode
    ).to(device)
    return model


def plot_sample_comparison(processor, model, test_dataset, device, save_path="./src/FNO/output_vs_prediction_sample.png"):
    model.eval()
    sample_idx = 0
    item = test_dataset[sample_idx]
    input_sample, output_gt = item['x'], item['y']
    input_sample = input_sample.unsqueeze(0).to(device)  # add batch dim

    with torch.no_grad():
        pred = model(input_sample)
        # 원본 코드 사용법 유지
        pred = processor.out_normalizer.inverse_transform(pred)
        output_gt = output_gt.unsqueeze(0).to(device)

    pred_img = pred[0, 0, :, :, 0].cpu().numpy()
    gt_img = output_gt[0, 0, :, :, 0].cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.imshow(gt_img, cmap='viridis')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title("Prediction")
    plt.imshow(pred_img, cmap='viridis')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved comparison figure → {save_path}")


# =========================
# Main training (refactor only)
# =========================
def main():
    # --- 경로만 하드코딩 (merge 산출물) ---
    merged_pt_path = "./src/preprocessing/merged.pt"

    # 1) 데이터 로드 (merge 결과물에서 직접)
    in_summation, out_summation = load_merged_tensors(merged_pt_path)

    # 2) 디바이스 및 텐서 이동
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    in_summation = in_summation.to(device)
    out_summation = out_summation.to(device)

    # 3) 정규화 (원본 방식 유지)
    in_normalizer = UnitGaussianNormalizer(mean=in_summation, std=in_summation, dim=[0, 2, 3, 4], eps=1e-8)
    out_normalizer = UnitGaussianNormalizer(mean=out_summation, std=out_summation, dim=[0, 2, 3, 4], eps=1e-8)
    in_normalizer.fit(in_summation)
    out_normalizer.fit(out_summation)
    processor = DefaultDataProcessor(in_normalizer, out_normalizer).to(device)

    # 4) 학습/검증 분할 (원본 sklearn 사용)
    train_in, test_in, train_out, test_out = train_test_split(
        in_summation, out_summation, test_size=0.2, random_state=42
    )

    # 5) DataLoader 구성
    train_dataset = CustomDataset(train_in, train_out)
    test_dataset = CustomDataset(test_in, test_out)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = {'test_dataloader': DataLoader(test_dataset, batch_size=32, shuffle=False)}

    # 6) 손실 정의 (원본 동일)
    l2loss = LpLoss(d=3, p=2)
    train_loss = l2loss
    eval_loss = {'l2': l2loss}

    # 7) 하이퍼파라미터 조합 (원본 동일; 1개 조합)
    n_modes_list = [(32, 16, 5)]
    hidden_channels_list = [16]
    n_layers_list = [3]
    domain_padding_list = [[0.1, 0.1, 0.1]]
    domain_padding_mode_list = ['symmetric']
    hyperparameter_combination = list(product(
        n_modes_list, hidden_channels_list, n_layers_list, domain_padding_list, domain_padding_mode_list
    ))

    # 8) 학습 루프 (원본 동작 유지)
    for idx, (n_modes, hidden_channels, n_layers, domain_padding, domain_padding_mode) in enumerate(hyperparameter_combination):
        print(f"Training with hyperparameters {idx + 1}/{len(hyperparameter_combination)}")
        print(f"n_modes: {n_modes}, hidden_channels: {hidden_channels}, n_layers: {n_layers}, "
              f"domain_padding: {domain_padding}, domain_padding_mode: {domain_padding_mode}")

        model = build_model(n_modes, hidden_channels, n_layers, domain_padding, domain_padding_mode, device)
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

        trainer = Trainer(
            model=model, n_epochs=100, device=device,
            data_processor=processor, wandb_log=False,
            eval_interval=3, use_distributed=False, verbose=True
        )

        ########## 학습 or 불러오기
        # trainer.train(
        #     train_loader=train_loader,
        #     test_loaders=test_loader,
        #     optimizer=optimizer,
        #     scheduler=scheduler,
        #     regularizer=False,
        #     training_loss=train_loss,
        #     eval_losses=eval_loss,
        #     save_best='test_dataloader_l2'
        # )

        model.load_state_dict(torch.load(f"./ckpt/best_model_state_dict.pt", weights_only=False))
        ##########

        model.to(device)
        model.eval()

        test_loss = trainer.evaluate(eval_loss, test_loader['test_dataloader'], 'test_dataloader')
        print(f'Test Loss: {test_loss}')

        # 비교 그림 (원본 스타일)
        plot_sample_comparison(processor, model, test_dataset, device,
                               save_path="output_vs_prediction_sample.png")

if __name__ == "__main__":
    main()