from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    rust_extensions=[RustExtension("atom_rust", "rust_bindings/Cargo.toml", binding=Binding.PyO3)],
)
