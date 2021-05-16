import numpy as np


def interp2(x, y, img, xi, yi):
    """
    按照matlab interp2写的加速2d插值
    当矩阵规模很大的时候,numba就快,矩阵规模小,则启动numba有开销
    原图是规整矩阵才能这么做
    """
    # @nb.jit
    def _interpolation(x, y, m, n, mm, nn, zxi, zyi, alpha, beta, img, return_img):
        qsx = int(m/2)
        qsy = int(n/2)
        for i in range(mm):     # 行号
            for j in range(nn):
                zsx, zsy = int(zxi[i, j]+qsx), int(zyi[i, j]+qsy)  # 左上的列坐标和行坐标
                # 左下的列坐标和行坐标
                zxx, zxy = int(zxi[i, j]+qsx), int(zyi[i, j]+qsy+1)
                # 右上的列坐标和行坐标
                ysx, ysy = int(zxi[i, j]+qsx+1), int(zyi[i, j]+qsy)
                # 右下的列坐标和行坐标
                yxx, yxy = int(zxi[i, j]+qsx+1), int(zyi[i, j]+qsy+1)
                fu0v = img[zsy, zsx]+alpha[i, j] * \
                    (int(img[ysy, ysx])-int(img[zsy, zsx]))
                fu0v1 = img[zxy, zxx]+alpha[i, j] * \
                    (int(img[yxy, yxx])-int(img[zxy, zxx]))
                fu0v0 = fu0v+beta[i, j]*(fu0v1-fu0v)
                return_img[i, j] = fu0v0
        return return_img
    m, n = img.shape  # 原始大矩阵大小
    mm, nn = xi.shape  # 小矩阵大小,mm为行,nn为列
    zxi = np.floor(xi)  # 用[u0]表示不超过S的最大整数
    zyi = np.floor(yi)
    alpha = xi-zxi   # u0-[u0]
    beta = yi-zyi
    return_img = np.zeros((mm, nn))
    return_img = _interpolation(
        x, y, m, n, mm, nn, zxi, zyi, alpha, beta, img, return_img)
    return return_img
