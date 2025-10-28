import torch
import torch.nn.functional as F

# ---------- PSNR ----------

@torch.no_grad()
def psnr(x, y, data_range=1.0, crop=0, y_only=False):
    """
    Compute PSNR between two batches.
    Args:
        x,y: [N,C,H,W], float, range [0,data_range].
        data_range: Max intensity value.
        crop: Border to remove before scoring.
        y_only: Use Y channel if True.
    Returns:
        Mean PSNR value in dB.
    """
    if y_only:
        w = torch.tensor([0.2126, 0.7152, 0.0722], device=x.device).view(1,3,1,1)
        x, y = (x*w).sum(1,True), (y*w).sum(1,True)
    if crop > 0:
        x, y = x[..., crop:-crop, crop:-crop], y[..., crop:-crop, crop:-crop]
    mse = F.mse_loss(x, y, reduction='none').flatten(1).mean(1)
    return 10 * torch.log10((data_range ** 2) / (mse + 1e-12)).mean()


# ---------- SSIM ----------

@torch.no_grad()
def ssim(x, y, data_range=1.0, ksize=11, sigma=1.5, crop=0, y_only=False):
    """
    Compute SSIM between two batches.
    Args:
        x,y: [N,C,H,W], float, range [0,data_range].
        data_range: Max intensity value.
        ksize: Gaussian window size.
        sigma: Gaussian std.
        crop: Border crop.
        y_only: Use Y channel if True.
    Returns:
        Mean SSIM value.
    """
    if y_only:
        w = torch.tensor([0.2126, 0.7152, 0.0722], device=x.device).view(1,3,1,1)
        x, y = (x*w).sum(1,True), (y*w).sum(1,True)
    if crop > 0:
        x, y = x[..., crop:-crop, crop:-crop], y[..., crop:-crop, crop:-crop]

    C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2
    ch = x.shape[1]
    g = torch.arange(ksize, device=x.device) - (ksize-1)/2
    w = torch.exp(-(g**2)/(2*sigma**2)); w /= w.sum()
    w2d = (w[:,None]*w[None,:]).expand(ch,1,ksize,ksize)
    mu_x = F.conv2d(x, w2d, padding=ksize//2, groups=ch)
    mu_y = F.conv2d(y, w2d, padding=ksize//2, groups=ch)
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x*mu_y
    sigma_x2 = F.conv2d(x*x, w2d, padding=ksize//2, groups=ch) - mu_x2
    sigma_y2 = F.conv2d(y*y, w2d, padding=ksize//2, groups=ch) - mu_y2
    sigma_xy = F.conv2d(x*y, w2d, padding=ksize//2, groups=ch) - mu_xy
    ssim_map = ((2*mu_xy+C1)*(2*sigma_xy+C2)) / ((mu_x2+mu_y2+C1)*(sigma_x2+sigma_y2+C2))
    return ssim_map.mean()

# ---------- SSIM ----------

@torch.no_grad()
def ssim(x, y, data_range=1.0, ksize=11, sigma=1.5, crop=0, y_only=False):
    """
    Compute SSIM between two batches.
    Args:
        x,y: [N,C,H,W], float, range [0,data_range].
        data_range: Max intensity value.
        ksize: Gaussian window size.
        sigma: Gaussian std.
        crop: Border crop.
        y_only: Use Y channel if True.
    Returns:
        Mean SSIM value.
    """
    if y_only:
        w = torch.tensor([0.2126, 0.7152, 0.0722], device=x.device).view(1,3,1,1)
        x, y = (x*w).sum(1,True), (y*w).sum(1,True)
    if crop > 0:
        x, y = x[..., crop:-crop, crop:-crop], y[..., crop:-crop, crop:-crop]

    C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2
    ch = x.shape[1]
    g = torch.arange(ksize, device=x.device) - (ksize-1)/2
    w = torch.exp(-(g**2)/(2*sigma**2)); w /= w.sum()
    w2d = (w[:,None]*w[None,:]).expand(ch,1,ksize,ksize)
    mu_x = F.conv2d(x, w2d, padding=ksize//2, groups=ch)
    mu_y = F.conv2d(y, w2d, padding=ksize//2, groups=ch)
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x*mu_y
    sigma_x2 = F.conv2d(x*x, w2d, padding=ksize//2, groups=ch) - mu_x2
    sigma_y2 = F.conv2d(y*y, w2d, padding=ksize//2, groups=ch) - mu_y2
    sigma_xy = F.conv2d(x*y, w2d, padding=ksize//2, groups=ch) - mu_xy
    ssim_map = ((2*mu_xy+C1)*(2*sigma_xy+C2)) / ((mu_x2+mu_y2+C1)*(sigma_x2+sigma_y2+C2))
    return ssim_map.mean()

# ---------- LPIPS ----------

try:
    import lpips
    _lpips_model = lpips.LPIPS(net="alex").eval()
except Exception:
    _lpips_model = None

@torch.no_grad()
def lpips_score(x, y, model=None, crop=0, y_only=False):
    """
    Compute LPIPS distance.
    Args:
        x,y: [N,3,H,W], float, range [0,1].
        model: Preloaded LPIPS model or None.
        crop: Border crop size.
        y_only: Convert to Y then replicate.
    Returns:
        Mean LPIPS value (lower is better).
    """
    if _lpips_model is None and model is None:
        raise RuntimeError("LPIPS not installed or model missing.")
    m = model or _lpips_model
    if y_only:
        w = torch.tensor([0.2126, 0.7152, 0.0722], device=x.device).view(1,3,1,1)
        x, y = (x*w).sum(1,True).repeat(1,3,1,1), (y*w).sum(1,True).repeat(1,3,1,1)
    if crop > 0:
        x, y = x[..., crop:-crop, crop:-crop], y[..., crop:-crop, crop:-crop]
    x, y = x*2-1, y*2-1
    return m(x, y).mean()

# ---------- NIQE ----------
@torch.no_grad()
def niqe(x: torch.Tensor, pop_mu: torch.Tensor, pop_cov: torch.Tensor, patch: int = 96) -> torch.Tensor:
    """
    Compute Natural Image Quality Evaluator (NIQE).

    Args:
        x: Tensor [N,3,H,W], RGB in [0,1].
        pop_mu: Tensor [D], population mean (float64).
        pop_cov: Tensor [D,D], population covariance (float64).
        patch: Patch size.

    Returns:
        Scalar tensor — mean NIQE score (lower is better).
    """
    w = torch.tensor([0.2126,0.7152,0.0722], device=x.device, dtype=x.dtype).view(1,3,1,1)
    Y = (x*w).sum(1)
    def _gauss_1d(half=3,sigma=7/6): 
        a=torch.arange(-half,half+1,device=Y.device,dtype=Y.dtype)
        w=torch.exp(-(a*a)/(2*sigma*sigma))
        return w/w.sum()
    def _mscn(im):
        k=_gauss_1d(); k2=(torch.outer(k,k)/k.sum()).to(im.dtype)[None,None]
        im2=(im-im.mean())/im.std()
        mu=F.conv2d(im[None,None],k2,padding=3)
        mu2=F.conv2d(im[None,None]*im[None,None],k2,padding=3)
        var=(mu2-mu*mu).clamp_min(0).sqrt()
        return ((im[None,None]-mu)/(var+1)).squeeze()
    g, p = torch.arange(0.2,10,0.001,device=Y.device), torch.special.gamma(2/torch.arange(0.2,10,0.001,device=Y.device))**2
    scores=[]
    for i in range(Y.size(0)):
        m=_mscn(Y[i])
        Z=m.flatten().double()
        X=Z-Z.mean()
        cov=torch.cov(X.unsqueeze(0))
        diff=(X.mean()-pop_mu.to(Y.device))
        val=torch.sqrt(diff @ torch.linalg.pinv(0.5*(pop_cov.to(Y.device)+cov)) @ diff)
        scores.append(val)
    return torch.stack(scores).mean()