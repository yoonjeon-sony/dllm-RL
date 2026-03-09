from setuptools import setup, find_packages

setup(
    name="llava",
    version="0.1.0",
    package_dir={"llava": "."},
    packages=["llava"] + ["llava." + p for p in find_packages(".")],
)
