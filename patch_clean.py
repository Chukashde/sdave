p = 'scripts/g1_diag.py'
s = open(p).read()
s = s.replace("choices=['random', 'fixed', 'spectral']",
              "choices=['random', 'fixed', 'spectral', 'clean']")
old = "    else:\n        x = torch.randn(n, dim) * 0.1\n"
new = ("    elif mode == 'clean':\n"
       "        k = U.shape[1]\n"
       "        idx = np.abs(U).argmax(0)\n"
       "        sgn = np.sign(U[idx, np.arange(k)])\n"
       "        x = torch.zeros(n, dim)\n"
       "        x[:, 1:1 + k] = torch.tensor(U * sgn, dtype=torch.float32) * np.sqrt(n)\n"
       + old)
assert old in s
s = s.replace(old, new, 1)
open(p, 'w').write(s)
print('patched')
