import os
import csv
import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_params(exp_number, main_path, params, save=True):
    if save:
        main_path = main_path + 'result_' + str(exp_number) + '/'
        make_dirs(main_path)
        make_dirs(main_path + '/plots/')
        make_dirs(main_path + '/models/')
        
        csvpath = './hparams.csv'
        if os.path.exists(csvpath):
            with open(csvpath, 'a') as f:
                w = csv.DictWriter(f, params.keys())
                w.writeheader()
                w.writerow(params)
        else:
            with open(csvpath, 'w') as f:
                w = csv.DictWriter(f, params.keys())
                w.writeheader()
                w.writerow(params)
        print('\n', params, flush=True)
        print('\n', flush=True)
        return main_path
    else:
        return exp_number

def plot_grid(imgs, n_row, n_col, plot_size, save_results_dir, epoch, name, plot=False):
    _, axs = plt.subplots(n_row, n_col, figsize=(plot_size, plot_size))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        im = ax.imshow(np.squeeze(img))
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
        ax.axis('off')
        plt.tight_layout()
    if plot:
        plt.show()
    else:
        plt.savefig(save_results_dir + 'ld_' + name + '_' + str(epoch) + '.png', dpi=150)
        plt.close()
        
def progress_plot(epoch, all_fid, interval):
    fids = np.array([x for x in all_fid if x.all() != 0])
    plt.figure()
    xax = np.arange(0, epoch+1, interval)
    plt.plot(xax, fids[:, 0], label='train')
    plt.plot(xax, fids[:, 1], label='test')
    plt.xlabel('Epoch')
    plt.ylabel('FID')
    plt.legend()
    plt.tight_layout()
    plt.show()

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """
    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"
    
    diff = mu1 - mu2
    
    # product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def calculate_fid(model, images1, images2):
    # calculate activations
    act1, _ = model.predict(images1)
    act2, _ = model.predict(images2)
    
    # calculate mean and covariance statistics
    mu1 = np.mean(act1, axis=0)
    sigma1 = np.cov(act1, rowvar=False)
    
    mu2 = np.mean(act2, axis=0)
    sigma2 = np.cov(act2, rowvar=False)
    
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid