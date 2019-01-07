import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch_datasets",
    version="0.0.1",
    author="Mike Peven",
    author_email="mpeven@gmail.com",
    description="Pytorch datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mpeven/pytorch_datasets",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
