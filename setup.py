from setuptools import find_packages, setup

setup(
    name="bnn_pref",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
