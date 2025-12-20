import numpy as np
import random
import h5py
from multiprocessing import Pool


def generate_scp_field_v2(N, b, size_x, size_y, level_max, density_map_ratio, iteration):
    """
    Generate permeability field with consistent 4×4 averaging but variable stride.

    New approach:
    1. Always generate on 256×128 base resolution
    2. Apply 4×4 box averaging
    3. Stride determined by density_map_ratio:
       - ratio=0.25 (LR): stride=4 → 64×32
       - ratio=0.5 (HR): stride=2 → 128×64

    This ensures:
    - Same box filter (4×4) for LR and HR
    - HR[::2, ::2] = LR (exact signal inclusion)
    - Consistent DC gain
    """

    # np.random.seed(42)
    # random.seed(42)

    # Base resolution (always 256×128)
    base_size_x = 256
    base_size_y = 128

    # Create base density map at full resolution with base value
    # Base value ensures non-zero baseline density
    base_value = 1.0 / 16.0
    base_density_map = np.ones((base_size_x, base_size_y)) * base_value

    def draw_squares(N, b, size_x, size_y, level=1, parent_coords=[(0, 0)]):
        if level > level_max:
            return

        # Calculate side_length as average of dimensions divided by b^level
        side_length = (size_x + size_y) / 2 / (b ** level)
        new_coords = []
        for px, py in parent_coords:
            for _ in range(N):
                if level == 1:  # First level: distribute across entire domain
                    x = random.uniform(0, size_x)
                    y = random.uniform(0, size_y)
                else:  # Subsequent levels: fractal subdivision
                    x = px + random.uniform(0, side_length * b)
                    y = py + random.uniform(0, side_length * b)

                # Handle wrapping for rectangular domain
                if x >= size_x:
                    x = x - size_x
                if y >= size_y:
                    y = y - size_y

                new_coords.append((x, y))

                if level == level_max:
                    update_density_map(base_density_map, x, y, side_length, size_x, size_y)

        draw_squares(N, b, size_x, size_y, level + 1, new_coords)

    def update_density_map(density_map, x, y, side_length, size_x, size_y):
        """Update base density map (256×128) with fractal counts"""
        for i in range(int(side_length)):
            for j in range(int(side_length)):
                # Calculate grid indices for base resolution
                xi = int(x + i)
                yj = int(y + j)

                # Handle wrapping
                if xi >= base_size_x:
                    xi = xi - base_size_x
                if yj >= base_size_y:
                    yj = yj - base_size_y

                density_map[xi, yj] += 1

    def apply_box_averaging_with_stride(base_map, kernel_size=4, target_output_size=None):
        """
        Apply 4×4 box averaging with calculated stride to achieve target output size

        Args:
            base_map: (256, 128) base density map
            kernel_size: always 4 for consistency
            target_output_size: (nx, ny) desired output size
                              - (64, 32) for LR (ratio=0.25)
                              - (128, 64) for HR (ratio=0.5)

        Returns:
            averaged_map: downsampled map

        Logic (사용자 분석):
            For ratio=0.5 (target 128×64):
            - stride = 4 * ratio = 4 * 0.5 = 2
            - Output size: (256 - 4 + 2*pad_h) / 2 + 1 = 128
            - Solving: (128 - 1) * 2 + 4 - 256 = 2 → pad_h = 2
            - Use edge padding (replicate edge values) to preserve physical info
        """
        if target_output_size is None:
            # Default: no downsampling
            return base_map

        nx_target, ny_target = target_output_size
        h, w = base_map.shape  # (256, 128)

        # Calculate stride: stride = base_size / target_size
        # For LR (64×32): stride = 256/64 = 4
        # For HR (128×64): stride = 256/128 = 2
        stride_x = int(h / nx_target)
        stride_y = int(w / ny_target)

        # Verify consistent stride (should be same for rectangular domain)
        assert stride_x == stride_y, f"Inconsistent strides: {stride_x} vs {stride_y}"
        stride = stride_x

        # Calculate required input size: (output - 1) * stride + kernel_size
        # Example for HR (128×64):
        # required_h = (128 - 1) * 2 + 4 = 127 * 2 + 4 = 258
        # required_w = (64 - 1) * 2 + 4 = 63 * 2 + 4 = 130
        required_h = (nx_target - 1) * stride + kernel_size
        required_w = (ny_target - 1) * stride + kernel_size

        # Calculate padding needed
        # For HR: pad_h = 258 - 256 = 2, pad_w = 130 - 128 = 2
        pad_h = required_h - h
        pad_w = required_w - w

        # Apply edge padding (replicate edge values to preserve physical information)
        if pad_h > 0 or pad_w > 0:
            base_map = np.pad(base_map,
                            ((0, max(0, pad_h)), (0, max(0, pad_w))),
                            mode='edge')  # Edge: replicate border values
            h, w = base_map.shape

        # Verify padded size matches requirements
        assert h >= required_h and w >= required_w, \
            f"Padding failed: got ({h}, {w}), need ({required_h}, {required_w})"

        # Perform 4×4 box averaging with stride
        output = np.zeros((nx_target, ny_target))

        for i in range(nx_target):
            for j in range(ny_target):
                start_i = i * stride
                start_j = j * stride

                # Extract 4×4 window and average
                window = base_map[start_i:start_i+kernel_size,
                                 start_j:start_j+kernel_size]
                output[i, j] = window.mean()

        return output

    def generate_perm_map(density_map, iteration):
        """Generate permeability and porosity maps from density map"""
        average_aperature = np.random.uniform(0.00001, 0.0001)

        # Determine output directory and discretization based on ratio
        if density_map_ratio == 0.25:
            # LR
            output_dir = "output_lr"
            discretization = [0.25, 0.25]
        elif density_map_ratio == 0.5:
            # HR
            output_dir = "output_hr"
            discretization = [0.125, 0.125]

        perm_dir = f"/home/geofluids/research/FNO/src/initial_perm/{output_dir}/perm_map_{iteration}.h5"
        poro_dir = f"/home/geofluids/research/FNO/src/initial_perm/{output_dir}/poro_map_{iteration}.h5"

        domain_size = density_map.shape[0] * density_map.shape[1]
        total_density = np.sum(density_map)
        aperature_ratio = average_aperature * domain_size / total_density
        aperature_map = np.minimum(0.005, aperature_ratio * density_map)

        perm_map = aperature_map ** 3 / 12.0 / 100.0 / 0.005
        poro_map = aperature_map / 0.005 + 0.01 * (1 - aperature_map / 0.005)

        data_shape = perm_map.shape
        data_dtype = perm_map.dtype

        with h5py.File(perm_dir, 'w') as hdf5_file:
            permsX_group = hdf5_file.create_group('permsX')
            data_dataset = permsX_group.create_dataset('Data', shape=data_shape, dtype=data_dtype)
            data_dataset[:, :] = perm_map
            permsX_group.attrs.create('Dimension', ['XY'], dtype=h5py.string_dtype(encoding='ascii', length=10))
            permsX_group.attrs['Discretization'] = discretization
            permsX_group.attrs['Origin'] = [-8, -4]
            permsX_group.attrs['Cell Centered'] = [True]

        with h5py.File(poro_dir, 'w') as hdf5_file:
            porosX_group = hdf5_file.create_group('porosX')
            data_dataset = porosX_group.create_dataset('Data', shape=data_shape, dtype=data_dtype)
            data_dataset[:, :] = poro_map
            porosX_group.attrs.create('Dimension', ['XY'], dtype=h5py.string_dtype(encoding='ascii', length=10))
            porosX_group.attrs['Discretization'] = discretization
            porosX_group.attrs['Origin'] = [-8, -4]
            porosX_group.attrs['Cell Centered'] = [True]

    # Main execution
    # 1. Generate base fractal density map (256×128)
    draw_squares(N, b, size_x, size_y)

    # 2. Determine target output size based on ratio
    kernel_size = 4

    if density_map_ratio == 0.25:
        # LR: 64×32
        target_size = (64, 32)
    elif density_map_ratio == 0.5:
        # HR: 128×64
        target_size = (128, 64)
    else:
        # Generic: calculate from ratio
        target_size = (int(base_size_x * density_map_ratio),
                      int(base_size_y * density_map_ratio))

    # 3. Apply 4×4 box averaging with auto-calculated stride and padding
    density_map = apply_box_averaging_with_stride(base_density_map, kernel_size, target_size)

    # 4. Generate permeability maps
    generate_perm_map(density_map, iteration)

    # # Print verification info
    # stride = int(base_size_x / target_size[0])
    # print(f"[{iteration}] Generated: {density_map.shape}, stride={stride}, ratio={density_map_ratio}")


if __name__ == '__main__':
 
    # Fix random seed for reproducibility
    # Example usage - DO NOT RUN (to preserve existing data)

    # LR generation (ratio=0.25, stride=4 → 64×32)
    # map_num = 100
    # N = 9
    # b = 2.64
    # size_x = 256
    # size_y = 128
    # level_max = 5
    # density_map_ratio = 0.25
    #
    # with Pool(30) as p:
    #     p.starmap(generate_scp_field_v2,
    #               [(N, b, size_x, size_y, level_max, density_map_ratio, i)
    #                for i in range(map_num)])

    # HR generation (ratio=0.5, stride=2 → 128×64)
    
    map_num = 100
    N = 9
    b = 2.64
    size_x = 256
    size_y = 128
    level_max = 5
    density_map_ratio = 0.5
    
    with Pool(30) as p:
        p.starmap(generate_scp_field_v2,
                  [(N, b, size_x, size_y, level_max, density_map_ratio, i)
                   for i in range(map_num)])

