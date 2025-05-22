import numpy as np
from ..utils.candy import _debug
class LinearSchedBlend:
    def __init__(self, mask: np.ndarray, n_steps):
        self.h, self.w, _ = mask.shape
        self.n_steps = n_steps
        self.base = (1-np.arange(n_steps)/(n_steps-1)).reshape((1, 1, n_steps))

        l = -np.ones((self.h, self.w, 1), np.float32)
        r =  np.ones((self.h, self.w, 1), np.float32)
        for i in range(128):
            
            m = (l+r)/2
            a3d  = self._get_a3d_by_m(m)
            if (np.abs(l-r).mean() < 1e-8):
                break
            a_fin = self._get_final_alpha_by_3d(a3d)
            l[a_fin<mask]  = m[a_fin<mask]
            r[a_fin>=mask] = m[a_fin>=mask]
        self.a3d = a3d
    def _get_a3d_by_m(self, m):
        # out shape (h, w, n)
        ret = np.zeros((self.h, self.w, self.n_steps), np.float32)
        # ret[m<0] = (self.base*(1+m))[m<0]
        ret += self.base*(1+m) * (m<0)
        ret += (self.base*(1-m) + m) * (m>=0)
        return ret
    def _get_final_alpha_by_3d(self, a3d):
        ret = np.zeros((self.h, self.w, 1), np.float32)
        for i in range(self.n_steps):
            ret = ret*i/(i+1)
            astep = a3d[:,:,i:i+1]
            ret = ret*(1-astep) + astep
        return ret
    
if (__name__=="__main__"):
    from ..model.inference.image2image import schedule_image_blend
    alpha_mask = np.random.uniform(0, 1, (800//8, 1280//8, 1))
    n_steps = 40
    tmp = LinearSchedBlend(alpha_mask, n_steps)

    a00 = alpha_mask[0, 0, 0]
    _debug(a00, schedule_image_blend(n_steps, alpha_mask[0, 0, 0]))
    _debug(tmp.a3d[0, 0])

