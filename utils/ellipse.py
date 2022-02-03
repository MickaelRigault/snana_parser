import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def covariance_to_ellipse(covariance, mean=(0,0), n_std=3):
    """ """
    vals, vecs = np.linalg.eigh(covariance)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:,order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * np.sqrt(vals)* n_std
    return {"xy":mean, "width":width, "height":height, "angle":theta}

def confidence_ellipse(x, y, ax=None, n_std=3.0, facecolor='0.5', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    
    from matplotlib.patches import Ellipse
    ellipse = Ellipse(**covariance_to_ellipse(np.cov(x, y), mean=np.mean([x,y], axis=1), n_std=n_std),
                          facecolor=facecolor, **kwargs)
    if ax is not None:
        ax.add_patch(ellipse)
        
    return ellipse
