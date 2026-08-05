"""
Import torch before anything that pulls in pandas.

On this Windows setup, importing pandas (directly, or transitively via
scraping/yfinance) before torch causes torch's DLL loader to access-violate on
c10.dll — reproduced outside pytest too: `import pandas; import torch` crashes,
`import torch; import pandas` does not. conftest.py is guaranteed to load
before any test module regardless of collection order, which is the only place
that can enforce this across the whole suite — a guard inside one test file
only protects that file if it happens to collect first.
"""
try:
    import torch  # noqa: F401
except ImportError:
    pass
