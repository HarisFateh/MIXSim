# Installation

Clone the repository and install it in editable mode with the development dependencies:

```bash
git clone https://github.com/HarisFateh/MIXSim.git
cd MIXSim
python -m pip install -e ".[dev]"
python -m pip check
pytest -q
```

MIXSim requires Python 3.10 or newer. Runtime dependencies are NumPy, pandas, and Matplotlib.
