# cnlplib

[![python](https://github.com/madsuite-org/cnlplib/actions/workflows/python.yml/badge.svg)](https://github.com/madsuite-org/cnlplib/actions/workflows/python.yml)

[cnlp libraries](https://cnlplib.madsuite.org) (compiled nonlinear programs) in Python:
evaluate them with NumPy or CuPy, and solve them with a cnlp solver in this process.

```sh
pip install "cnlplib @ git+https://github.com/madsuite-org/cnlplib#subdirectory=interfaces/cnlplib-py"
```

```python
import cnlplib

m = cnlplib.CModel("@emp", "acp", "pglib_opf_case118_ieee")
r = cnlplib.solve(m, solver="madnlp")
```

Documentation: [cnlplib.madsuite.org/python](https://cnlplib.madsuite.org/python/).
