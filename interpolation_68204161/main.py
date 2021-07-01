import torch
import torch.nn.functional as F

b = 2
c = 4
h = w = 8

a = torch.randn((b, c, h, w))
a_upsample = F.interpolate(a, [h*2, w*2], mode='bilinear', align_corners=True)

a_mod = a.clone()
a_mod[:, 0] *= 1000
a_mod_upsample = F.interpolate(a_mod, [h*2, w*2], mode='bilinear', align_corners=True)

print(torch.isclose(a_upsample[:,0], a_mod_upsample[:,0]).all())
print(torch.isclose(a_upsample[:,1], a_mod_upsample[:,1]).all())
print(torch.isclose(a_upsample[:,2], a_mod_upsample[:,2]).all())
print(torch.isclose(a_upsample[:,3], a_mod_upsample[:,3]).all())