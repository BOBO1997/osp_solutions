# Available at setup time due to pyproject.toml
# from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

__version__ = "0.0.1"

setup(
    name="osp_solutions",
    packages=find_packages(), # (where=["qamp_gse"]), # ["qamp_gse"],
    # package_dir={
    #     "qamp_gse": "qamp_gse",
    # },
    version=__version__,
    author="Bo YANG",
    author_email="Bo.Yang@lip6.fr",
    url="https://github.com/BOBO1997/osp_solutions",
    description="",
    long_description="",
    # ext_modules=[
    #     Pybind11Extension(
    #         "qamp_gse_cpp",
    #         sources=["cpp/sd.cpp"],
    #         ### Example: passing in the version to the compiled code
    #         define_macros=[("VERSION_INFO", __version__)],
    #     ),
    # ],
    extras_require={"test": "pytest"},
    # cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.9",
)