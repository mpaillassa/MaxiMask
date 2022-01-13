#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="MaxiMask",
    version="1.3.1",
    author="Maxime Paillassa",
    author_email="maxime.paillassa@nagoya-u.jp",
    description="Convolutional neural networks to detect contaminants in astronomical images.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mpaillassa/MaxiMask",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "maximask = maximask_and_maxitrack.scripts.maximask:main",
            "maxitrack = maximask_and_maxitrack.scripts.maxitrack:main",
        ]
    },
    install_requires=["astropy", "scipy", "numpy"],
    python_requires=">=3.6",
    license="MIT",
    include_package_data=True,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
