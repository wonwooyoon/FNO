import numpy as np
import h5py

def generate_gaussian_field_direct_cov(X, Y, nX, nY, x_corr, y_corr, mean, std):
    
    x = np.linspace(0, X, nX)
    y = np.linspace(0, Y, nY)
    xx, yy = np.meshgrid(x, y)
    points = np.vstack([xx.ravel(), yy.ravel()]).T

    dx = points[:, 0:1] - points[:, 0:1].T
    dy = points[:, 1:2] - points[:, 1:2].T
    dist2 = (dx / x_corr)**2 + (dy / y_corr)**2
    cov_matrix = std**2 * np.exp(-dist2)

    Z = np.random.multivariate_normal(mean=np.full(points.shape[0], mean), cov=cov_matrix)
    Z_map = Z.reshape(nY, nX)
    return xx, yy, Z_map

def generate_gaussian_field_fft_anisotropic(X, Y, nX, nY, x_corr, y_corr, mean, std):
    dx = X / (nX-1)
    dy = Y / (nY-1)
    x = np.linspace(0, X, nX)
    y = np.linspace(0, Y, nY)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    kx = np.fft.fftfreq(nX, d=dx)
    ky = np.fft.fftfreq(nY, d=dy)
    kxx, kyy = np.meshgrid(kx, ky, indexing='ij')
    k2 = (kxx * x_corr)**2 + (kyy * y_corr)**2

    spectrum = np.exp(-2 * np.pi ** 2 * k2)
    
    noise = np.random.normal(size=(nX, nY)) + 1j * np.random.normal(size=(nX, nY))
    field_fft = noise * np.sqrt(spectrum)

    field = np.real(np.fft.ifft2(field_fft))
    field -= np.mean(field)
    field /= np.std(field)
    field = mean + std * field

    return xx, yy, field

def generate_calcite_map(Z, mu, sigma, output_dir, i):

    Z = 10 ** (mu + sigma * Z)
    #Z = Z.T
    
    Z = np.clip(Z, a_max=2.00, a_min=0.0)
    Z[Z < 1e-6] = 0 # Under detection limit is treated as 0
    Z = Z / 5.0 # Divided by 5 mm thickness to calculate the volume fraction

    with h5py.File(output_dir, 'w') as hdf5_file:
        calcite_group = hdf5_file.create_group('calcite_mapX')
        data_shape = Z.shape
        data_dtype = Z.dtype
        data_dataset = calcite_group.create_dataset('Data', shape=data_shape, dtype=data_dtype)
        data_dataset[:, :] = Z
        calcite_group.attrs.create('Dimension', ['XY'], dtype=h5py.string_dtype(encoding='ascii', length=10))
        calcite_group.attrs.create('Space Interpolation Method', ['STEP'], dtype=h5py.string_dtype(encoding='ascii', length=10))
        calcite_group.attrs['Discretization'] = [0.125, 0.125]
        calcite_group.attrs['Origin'] = [-8.0, -4.0]
        calcite_group.attrs['Cell Centered'] = [True]

def generate_clinochlore_map(Z, mu, sigma, output_dir, n):

    Z = 10 ** (mu + sigma * Z)
    #Z = Z.T

    Z = np.clip(Z, a_max=2.00, a_min=0.0)
    Z[Z < 1e-6] = 0 # Under detection limit is treated as 0
    Z = Z / 5.0 # 5 mm thickness

    with h5py.File(output_dir, 'w') as hdf5_file:
        clinochlore_group = hdf5_file.create_group('clinochlore_mapX')
        data_shape = Z.shape
        data_dtype = Z.dtype
        data_dataset = clinochlore_group.create_dataset('Data', shape=data_shape, dtype=data_dtype)
        data_dataset[:, :] = Z
        clinochlore_group.attrs.create('Dimension', ['XY'], dtype=h5py.string_dtype(encoding='ascii', length=10))
        clinochlore_group.attrs.create('Space Interpolation Method', ['STEP'], dtype=h5py.string_dtype(encoding='ascii', length=10))
        clinochlore_group.attrs['Discretization'] = [0.125, 0.125]
        clinochlore_group.attrs['Origin'] = [-8.0, -4.0]
        clinochlore_group.attrs['Cell Centered'] = [True]

def generate_pyrite_map(Z, mu, sigma, output_dir, n):

    Z = 10 ** (mu + sigma * Z)
    #Z = Z.T

    Z = np.clip(Z, a_max=0.05, a_min=0.0)
    Z[Z < 1e-9] = 0 # Under detection limit is treated as 0
    Z = Z / 5.0 # 5 mm thickness

    with h5py.File(output_dir, 'w') as hdf5_file:
        pyrite_group = hdf5_file.create_group('pyrite_mapX')
        data_shape = Z.shape
        data_dtype = Z.dtype
        data_dataset = pyrite_group.create_dataset('Data', shape=data_shape, dtype=data_dtype)
        data_dataset[:, :] = Z
        pyrite_group.attrs.create('Dimension', ['XY'], dtype=h5py.string_dtype(encoding='ascii', length=10))
        pyrite_group.attrs.create('Space Interpolation Method', ['STEP'], dtype=h5py.string_dtype(encoding='ascii', length=10))
        pyrite_group.attrs['Discretization'] = [0.125, 0.125]
        pyrite_group.attrs['Origin'] = [-8.0, -4.0]
        pyrite_group.attrs['Cell Centered'] = [True]

if __name__ == "__main__":
    
    n = 100
    X = 128
    Y = 64
    nX = 128
    nY = 64
    mean = 0.0
    std = 1.0

    for i in range(n):
        
        x_corr1 = np.random.uniform(4.0, 20.0)
        y_corr1 = np.random.uniform(4.0, 20.0)
        x_corr2 = np.random.uniform(4.0, 20.0)
        y_corr2 = np.random.uniform(4.0, 20.0)
        x_corr3 = np.random.uniform(4.0, 20.0)
        y_corr3 = np.random.uniform(4.0, 20.0)

        print(f"Iteration {i}: x_corr1={x_corr1}, y_corr1={y_corr1}, x_corr2={x_corr2}, y_corr2={y_corr2}, x_corr3={x_corr3}, y_corr3={y_corr3}")

        xx, yy, Z1 = generate_gaussian_field_fft_anisotropic(
            X, Y, nX, nY, x_corr1, y_corr1, mean, std
        )
        
        xx, yy, Z2 = generate_gaussian_field_fft_anisotropic(
            X, Y, nX, nY, x_corr2, y_corr2, mean, std
        )

        xx, yy, Z3 = generate_gaussian_field_fft_anisotropic(
            X, Y, nX, nY, x_corr3, y_corr3, mean, std
        )

        output_dir1 = f"/home/geofluids/research/FNO/src/initial_mineral/output_hr/calcite_{i}.h5"
        output_dir2 = f"/home/geofluids/research/FNO/src/initial_mineral/output_hr/clinochlore_{i}.h5"
        output_dir3 = f"/home/geofluids/research/FNO/src/initial_mineral/output_hr/pyrite_{i}.h5"

        mu1 = np.random.uniform(-1.77, -1.17) # mean -1.47
        s1 = np.random.uniform(0.6, 0.8) # mean 0.7
        mu2 = np.random.uniform(-1.23, -0.63) # mean -0.93
        s2 = np.random.uniform(0.36, 0.56) # std 0.46
        mu3 = np.random.uniform(-5.01, -3.71) # std -4.01
        s3 = np.random.uniform(1.16, 1.36) # std 1.26

        generate_calcite_map(Z1, mu1, s1, output_dir1, i)
        generate_clinochlore_map(Z2, mu2, s2, output_dir2, i)
        generate_pyrite_map(Z3, mu3, s3, output_dir3, i)


    
