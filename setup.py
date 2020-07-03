#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

with open("requirements.txt") as requirements_file:
    requirements = [
        req[:-1] if req[-1] == "\n" else req for req in requirements_file.readlines()
    ]

print(requirements)
setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Mykyta Makarov",
    author_email="evil.unicorn1@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Minimalistic framework for reinforcement learning with OpenAI gym environments.",
    entry_points={"console_scripts": ["gym-loop=gym_loop.cli:main",],},
    install_requires=requirements,
    extras_require={"pytorch": ["torch", "torchvision"]},
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="gym-loop",
    name="gym-loop",
    packages=find_packages(include=["gym_loop", "gym_loop.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/eublefar/gym-loop",
    version="0.1.0",
    zip_safe=False,
)
