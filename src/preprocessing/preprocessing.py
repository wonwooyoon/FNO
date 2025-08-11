import h5py
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

sys.path.append('./')

from neuraloperator.neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuraloperator.neuralop.data.transforms.data_processors import DefaultDataProcessor
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from neuraloperator.neuralop import LpLoss, H1Loss

from neuraloperator.neuralop.models import TFNO
from neuraloperator.neuralop import Trainer
from neuraloperator.neuralop.training import AdamW

from itertools import product

class CustomDataset(Dataset):
    def __init__(self, input_tensor, output_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def __len__(self):
        return self.input_tensor.shape[0]

    def __getitem__(self, idx):
        return {'x': self.input_tensor[idx], 'y': self.output_tensor[idx]}

def read_pflotran_data(min_count: int, max_count: int):
    base_dir = Path("./src/pflotran_run/output")
    others = pd.read_csv("./src/initial_others/output/others.csv")  # 한 번만 읽기

    sims_inputs = []
    sims_outputs = []

    for i in range(min_count, max_count+1):
        file_path = base_dir / f"pflotran_{i}" / f"pflotran_{i}.h5"
        with h5py.File(file_path, "r") as f:
            keys_list = list(f.keys())

            # --- 공통 좌표/마스크 (원 코드와 동일 기준) ---
            XC = np.array(f["Domain"]["XC"][:])
            YC = np.array(f["Domain"]["YC"][:])
            ZC = np.array(f["Domain"]["ZC"][:])
            zc_mask = (ZC >= -1e-5) & (ZC <= 1e-5)

            XC_m = XC[zc_mask]
            YC_m = YC[zc_mask]

            # 좌표 반올림은 동일하게 decimals=5 유지
            x_round = np.round(XC_m, 5)
            y_round = np.round(YC_m, 5)

            # unique + inverse index로 (xi, yi) 매핑 벡터화
            xc_unique, inv_x = np.unique(x_round, return_inverse=True)
            yc_unique, inv_y = np.unique(y_round, return_inverse=True)
            nx, ny = len(xc_unique), len(yc_unique)

            def to_grid(vec):
                """벡터를 (nx, ny) 그리드로 산포 (원 코드 동일 로직)."""
                grid = np.zeros((nx, ny), dtype=np.float32)
                grid[inv_x, inv_y] = vec.astype(np.float32, copy=False)
                return grid

            grp0 = f["   0 Time  0.00000E+00 y"]
            perm      = np.array(grp0["Permeability [m^2]"][:])[zc_mask]
            calcite   = np.array(grp0["Calcite VF [m^3 mnrl_m^3 bulk]"][:])[zc_mask]
            clino     = np.array(grp0["Clinochlore VF [m^3 mnrl_m^3 bulk]"][:])[zc_mask]
            pyrite    = np.array(grp0["Pyrite VF [m^3 mnrl_m^3 bulk]"][:])[zc_mask]
            smectite  = np.array(grp0["Smectite_MX80 VF [m^3 mnrl_m^3 bulk]"][:])[zc_mask]
            material  = np.array(grp0["Material ID"][:])[zc_mask]

            grp1 = f["   1 Time  5.00000E+01 y"]
            x_velo    = np.array(grp1["Liquid X-Velocity [m_per_yr]"][:])[zc_mask]
            y_velo    = np.array(grp1["Liquid Y-Velocity [m_per_yr]"][:])[zc_mask]

            # others.csv에서 i번째 행만 사용 (원 코드와 동일한 열 인덱스)
            others_pressure = float(others.iat[i, 0])
            others_ratio    = float(others.iat[i, 2])

            # 기본 입력 채널 그리드 구성 (시간 제외)
            perm_grid      = to_grid(perm)
            calcite_grid   = to_grid(calcite)
            clino_grid     = to_grid(clino)
            pyrite_grid    = to_grid(pyrite)
            smectite_grid  = to_grid(smectite)
            material_grid  = to_grid(material)
            x_velo_grid    = to_grid(x_velo)
            y_velo_grid    = to_grid(y_velo)
            ratio_grid     = np.full((nx, ny), others_ratio, dtype=np.float32)

            # (채널, nx, ny, 1) 형태의 기본 입력
            input_base = np.stack(
                [
                    perm_grid,
                    calcite_grid,
                    clino_grid,
                    pyrite_grid,
                    smectite_grid,
                    material_grid,
                    x_velo_grid,
                    y_velo_grid,
                    ratio_grid,
                ],
                axis=0,
            )[:, :, :, np.newaxis]  # (9, nx, ny, 1)

            # --- 시간 키 수집 (원 기준 유지: "N Time" 형태가 포함된 그룹) ---
            # 100~2000까지 100 step: key_name = f"{int(X/50)} Time"
            # 키 검색은 포함 여부로 동일하게 유지
            available = {}
            for X in range(100, 2001, 100):
                token = f"{int(X/50)} Time"
                # 최초 매칭 키만 사용 (원 코드 동일)
                match = next((k for k in keys_list if token in k), None)
                if match is not None:
                    available[int(X/50)] = match

            # 시간 정렬(원 코드와 동일: key.split()[0] 정수 기준)
            times_sorted = sorted(available.keys())

            # 시간축별 입력/출력 누적 (리스트에 모아 한 번에 결합)
            input_time_slices = []
            output_time_slices = []

            for t_num in times_sorted:
                matched_key = available[t_num]
                total_uo2 = np.log10(np.array(f[matched_key]["Total UO2++ [M]"][:])[zc_mask])
                output_grid = to_grid(total_uo2)  # (nx, ny)

                # (nx, ny, 1)을 마지막 축으로 유지
                output_time_slices.append(output_grid[np.newaxis, :, :, np.newaxis])
                input_time_slices.append(input_base)

            # 시간축 결합: 마지막 축으로 이어붙임 (원 코드 axis=3)
            # input: (9, nx, ny, nt), output: (nx, ny, nt)
            input_tensor = np.concatenate(input_time_slices, axis=3).astype(np.float32, copy=False)
            output_tensor = np.concatenate(output_time_slices, axis=3).astype(np.float32, copy=False)

            # 시뮬레이션 축 추가 후 누적 (list에 모아서 마지막에 stack)
            sims_inputs.append(input_tensor[np.newaxis, ...])   # (1, 9, nx, ny, nt)
            sims_outputs.append(output_tensor[np.newaxis, ...]) # (1, 1, nx, ny, nt)

    # 모든 시뮬레이션 결합
    input_tensor_full = np.concatenate(sims_inputs, axis=0)    # (N, 9, nx, ny, nt)
    output_tensor_full = np.concatenate(sims_outputs, axis=0)  # (N, 1, nx, ny, nt)

    print("input_tensor shape:", input_tensor_full.shape)
    print("output_tensor shape:", output_tensor_full.shape)

    # torch 텐서로 저장 (float32)
    torch.save(torch.from_numpy(input_tensor_full), "./src/preprocessing/input_tensor_com1.pt")
    torch.save(torch.from_numpy(output_tensor_full), "./src/preprocessing/output_tensor_com1.pt")


def load_and_concat_inoutput_tensors():
    in_paths = [
        "./src/preprocessing/input_tensor_com1.pt",
        "./src/preprocessing/input_tensor_com2.pt",
        "./src/preprocessing/input_tensor_com3.pt",
    ]
    out_paths = [
        "./src/preprocessing/output_tensor_com1.pt",
        "./src/preprocessing/output_tensor_com2.pt",
        "./src/preprocessing/output_tensor_com3.pt",
    ]

    in_tensors = [torch.load(p) for p in in_paths]
    out_tensors = [torch.load(p) for p in out_paths]

    in_summation = torch.cat(in_tensors, dim=0)
    out_summation = torch.cat(out_tensors, dim=0)

    print("Input Summation shape:", in_summation.shape)
    print("Output Summation shape:", out_summation.shape)

    return in_summation, out_summation

def FNO_settings(in_summation, out_summation):

    device = 'cuda'
    in_summation = in_summation.to(device)
    out_summation = out_summation.to(device)

    # Normalize the input and output tensors
    in_normalizer = UnitGaussianNormalizer(mean=in_summation, std=in_summation, dim=[0, 2, 3, 4], eps=1e-8)
    out_normalizer = UnitGaussianNormalizer(mean=out_summation, std=out_summation, dim=[0, 2, 3, 4], eps=1e-8)

    in_normalizer.fit(in_summation)
    out_normalizer.fit(out_summation)

    processor = DefaultDataProcessor(in_normalizer, out_normalizer).to(device)

    train_in, test_in, train_out, test_out = train_test_split(
        in_summation, out_summation, test_size=0.2, random_state=42
    )

    # Create DataLoader for training and validation sets
    train_dataset = CustomDataset(train_in, train_out)
    test_dataset = CustomDataset(test_in, test_out)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = {'test_dataloader': DataLoader(test_dataset, batch_size=32, shuffle=False)}

    l2loss = LpLoss(d=3, p=2)
    h1loss = H1Loss(d=3)

    train_loss = l2loss
    eval_loss = {'l2' : l2loss}

    n_modes_list = [(32, 16, 5)]
    hidden_channels_list = [128]
    n_layers_list = [4]
    domain_padding_list = [[0.1, 0.1, 0.1]]
    domain_padding_mode_list = ['symmetric']

    hyperparameter_combination = list(product(n_modes_list, hidden_channels_list, n_layers_list, domain_padding_list, domain_padding_mode_list))

    best_loss = float('inf')
    best_hp = None

    for idx, (n_modes, hidden_channels, n_layers, domain_padding, domain_padding_mode) in enumerate(hyperparameter_combination):
        print(f"Training with hyperparameters {idx + 1}/{len(hyperparameter_combination)}")
        print(f"n_modes: {n_modes}, hidden_channels: {hidden_channels}, n_layers: {n_layers}, domain_padding: {domain_padding}, domain_padding_mode: {domain_padding_mode}")

        model = TFNO(
            n_modes = n_modes,
            in_channels = 9,
            out_channels = 1,
            hidden_channels = hidden_channels,
            n_layers = n_layers,
            lifting_channel_ratio = 2,
            projection_channel_ratio = 2,
            positional_embedding = 'grid',
            domain_padding = domain_padding,
            domain_padding_mode = domain_padding_mode
        ).to(device)

        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

        trainer = Trainer(model=model, n_epochs=100,
                            device=device,
                            data_processor=processor,
                            wandb_log=False,
                            eval_interval=3,
                            use_distributed=False,
                            verbose=True
                            )

        trainer.train(train_loader=train_loader,
                      test_loaders=test_loader,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      regularizer=False,
                      training_loss=train_loss,
                      eval_losses=eval_loss,
                      save_best=f'test_dataloader_l2'
                      )

        # To load the model later:
        model.load_state_dict(torch.load(f"TFNO_best_model_{idx}.pt"))
        model.to(device)

        model.eval()

        test_loss = trainer.evaluate(eval_loss, test_loader['test_dataloader'], 'test_dataloader')
        print(f'Test Loss: {test_loss}')

        import matplotlib.pyplot as plt

        # Pick a sample from the test set
        sample_idx = 0
        input_sample, output_gt = test_dataset[sample_idx]
        input_sample = input_sample.unsqueeze(0).to(device)  # add batch dim

        # Model prediction
        with torch.no_grad():
            pred = model(input_sample)
            pred = processor.output_denormalizer(pred)
            output_gt = processor.output_denormalizer(output_gt.unsqueeze(0).to(device))

        # Select a time slice (e.g., first time step)
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
        plt.savefig("output_vs_prediction_sample.png")
        plt.close()

if __name__ == "__main__":
    i = 0
    j = 46
    read_pflotran_data(i, j)
    # in_summation, out_summation = load_and_concat_inoutput_tensors()
    # FNO_settings(in_summation, out_summation)
