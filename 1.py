"""
Part 1
Complete Codebase covering s 1.2.2, 1.2.3, 1.3.1, and 1.3.2.
"""

import numpy as np
import numpy.fft as fft
from PIL import Image
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg, LinearOperator
import bm3d


# 1.1 / 1.2.1 Data Loading, Preprocessing & System Setup

psfname = "psf_sample.tif"
imgname = "rawdata_hand_sample.tif"
f = 0.25 # Downsampling factor

def loadData():
    psf = np.array(Image.open(psfname), dtype='float32')
    data = np.array(Image.open(imgname), dtype='float32')

    # Background subtraction
    bg = np.mean(psf[5:15, 5:15])
    psf -= bg
    data -= bg

    # Resize function
    def resize(img, factor):
        num = int(-np.log2(factor))
        for i in range(num):
            img = 0.25 * (img[::2, ::2, ...] + img[1::2, ::2, ...] + img[::2, 1::2, ...] + img[1::2, 1::2, ...])
        return img

    psf = resize(psf, f)
    data = resize(data, f)

    # Normalize
    psf /= np.linalg.norm(psf.ravel())
    data /= np.linalg.norm(data.ravel())
    return psf, data


psf, b = loadData()
sensor_h, sensor_w = b.shape

# Padding dimensions
pad_h, pad_w = sensor_h // 2, sensor_w // 2
shape_x = (sensor_h + 2 * pad_h, sensor_w + 2 * pad_w)

# Pre-compute FFT of PSF
psf_padded = np.zeros(shape_x, dtype=np.float32)
psf_padded[pad_h:pad_h+sensor_h, pad_w:pad_w+sensor_w] = psf
psf_shifted = fft.ifftshift(psf_padded)
H = fft.fft2(psf_shifted)
H_conj = np.conj(H)
H_mag_sq = np.abs(H)**2

# Operators
def crop(x):
    return x[pad_h:pad_h+sensor_h, pad_w:pad_w+sensor_w]

def pad(y):
    padded = np.zeros(shape_x, dtype=np.float32)
    padded[pad_h:pad_h+sensor_h, pad_w:pad_w+sensor_w] = y
    return padded

def A(x):
    """Forward operator"""
    X = fft.fft2(x)
    return crop(np.real(fft.ifft2(H * X)))

def AT(y):
    """Adjoint operator"""
    Y = fft.fft2(pad(y))
    return np.real(fft.ifft2(H_conj * Y))

def soft_threshold(v, kappa):
    return np.maximum(0, np.abs(v) - kappa) * np.sign(v)

ATb = AT(b) # Precompute A^T b since it is used in all methods



#  1.2.2: ADMM with L1 Regularization

def admm_l1(max_iters=15, rho=1e-3, tau=1e-4):
    x = np.zeros(shape_x, dtype=np.float32)
    v = np.zeros(shape_x, dtype=np.float32) # Auxiliary var for L1
    u = np.zeros(shape_x, dtype=np.float32) # Dual var

    def lhs_operator(x_vec):
        x_2d = x_vec.reshape(shape_x)
        return (AT(A(x_2d)) + rho * x_2d).flatten()

    LHS = LinearOperator((np.prod(shape_x), np.prod(shape_x)), matvec=lhs_operator)

    for i in range(max_iters):
        print(f"  Iteration {i+1}/{max_iters}")
        # 1. x-update (CG)
        rhs = ATb + rho * (v - u)
        x_flat, _ = cg(LHS, rhs.flatten(), x0=x.flatten(), rtol=1e-4, maxiter=20)
        x = x_flat.reshape(shape_x)

        # 2. v-update (L1 Soft Thresholding directly on pixels)
        v = soft_threshold(x + u, tau / rho)

        # 3. non-negativity constraint (Applied directly to v)
        v = np.maximum(v, 0)

        # 4. u-update
        u = u + x - v

    return crop(x)



#  1.2.3: ADMM with L2 Regularization

def admm_l2(max_iters=15, rho=1e-3, tau=1e-4):
    x = np.zeros(shape_x, dtype=np.float32)
    v = np.zeros(shape_x, dtype=np.float32)
    u = np.zeros(shape_x, dtype=np.float32)

    def lhs_operator(x_vec):
        x_2d = x_vec.reshape(shape_x)
        return (AT(A(x_2d)) + rho * x_2d).flatten()

    LHS = LinearOperator((np.prod(shape_x), np.prod(shape_x)), matvec=lhs_operator)

    for i in range(max_iters):
        print(f"  Iteration {i+1}/{max_iters}")
        # 1. x-update (CG)
        rhs = ATb + rho * (v - u)
        x_flat, _ = cg(LHS, rhs.flatten(), x0=x.flatten(), rtol=1e-4, maxiter=20)
        x = x_flat.reshape(shape_x)

        # 2. v-update (L2 Shrinkage)
        # argmin_v (tau * ||v||_2^2 + (rho/2) * ||v - (x+u)||_2^2)
        v = (rho / (2 * tau + rho)) * (x + u)

        # 3. non-negativity constraint
        v = np.maximum(v, 0)

        # 4. u-update
        u = u + x - v

    return crop(x)



#  1.3.1: Plug and Play ADMM - Closed Formula Update

def pnp_admm_closed_form(max_iters=5, rho=1e-3, sigma=0.01):
    print("\n--- Running  1.3.1: PnP ADMM (Closed Formula) ---")
    x = np.zeros(shape_x, dtype=np.float32)
    v = np.zeros(shape_x, dtype=np.float32)
    u = np.zeros(shape_x, dtype=np.float32)

    for i in range(max_iters):
        print(f"  Iteration {i+1}/{max_iters}")

        # 1. x-update (Closed form inversion in Fourier Domain)
        # Using Equation 22 approach: x = F^-1 [ F(A^Tb + rho*(v-u)) / (|H|^2 + rho) ]
        rhs = ATb + rho * (v - u)
        RHS_fft = fft.fft2(rhs)
        X = RHS_fft / (H_mag_sq + rho)
        x = np.real(fft.ifft2(X))

        # 2. v-update (BM3D Denoiser step)
        # Scale input to [0,1] range for BM3D stability
        noisy_img = x + u
        norm_factor = np.max(np.abs(noisy_img)) if np.max(np.abs(noisy_img)) > 0 else 1.0
        v = bm3d.bm3d(noisy_img / norm_factor, sigma_psd=sigma) * norm_factor

        # 3. non-negativity
        v = np.maximum(v, 0)

        # 4. u-update
        u = u + x - v

    return crop(x)



#  1.3.2: (Optional) PnP ADMM - Conjugate Gradient Update

def pnp_admm_cg(max_iters=5, rho=1e-3, sigma=0.01):
    print("\n--- Running  1.3.2: PnP ADMM (Conjugate Gradient) ---")
    x = np.zeros(shape_x, dtype=np.float32)
    v = np.zeros(shape_x, dtype=np.float32)
    u = np.zeros(shape_x, dtype=np.float32)

    def lhs_operator(x_vec):
        x_2d = x_vec.reshape(shape_x)
        return (AT(A(x_2d)) + rho * x_2d).flatten()

    LHS = LinearOperator((np.prod(shape_x), np.prod(shape_x)), matvec=lhs_operator)

    for i in range(max_iters):
        print(f"  Iteration {i+1}/{max_iters}")

        # 1. x-update (CG)
        rhs = ATb + rho * (v - u)
        x_flat, _ = cg(LHS, rhs.flatten(), x0=x.flatten(), rtol=1e-4, maxiter=20)
        x = x_flat.reshape(shape_x)

        # 2. v-update (BM3D Denoiser step)
        noisy_img = x + u
        norm_factor = np.max(np.abs(noisy_img)) if np.max(np.abs(noisy_img)) > 0 else 1.0
        v = bm3d.bm3d(noisy_img / norm_factor, sigma_psd=sigma) * norm_factor

        # 3. non-negativity
        v = np.maximum(v, 0)

        # 4. u-update
        u = u + x - v

    return crop(x)



# Execution and Display

if __name__ == "__main__":
    # Note: Hyperparameters (rho, tau, sigma) may need tuning depending on your scene.

    img_l1 = admm_l1(max_iters=15, rho=0.001, tau=0.005)
    img_l2 = admm_l2(max_iters=15, rho=0.001, tau=0.005)
    img_pnp_cf = pnp_admm_closed_form(max_iters=5, rho=0.001, sigma=0.05)
    img_pnp_cg = pnp_admm_cg(max_iters=5, rho=0.001, sigma=0.05)

    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    titles = [
        "1.2.2: ADMM (L1 Reg)",
        "1.2.3: ADMM (L2 Reg)",
        "1.3.1: PnP ADMM (Closed Form)",
        "1.3.2: PnP ADMM (CG)"
    ]
    images = [img_l1, img_l2, img_pnp_cf, img_pnp_cg]

    for ax, title, img in zip(axes, titles, images):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=14)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("all_reconstructions.png", dpi=150)
    plt.show()
