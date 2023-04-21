# flake8: noqa

from setuptools import setup

with open("README.md") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

exec(open("evalem/__version__.py").read())

setup(
    name="evalem",
    version=__version__,
    description="An evaluation framework for your NLP pipelines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NASA-IMPACT/evalem",
    author_email="np0069@uah.edu",
    python_requires=">=3.8",
    packages=[
        "evalem",
        "evalem.evaluators",
        "evalem.metrics",
        "evalem.misc",
        "evalem.models",
        "evalem.pipelines",
    ],
    install_requires=required,
    classifiers=[
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Machine Learning",
        "Topic :: Neural Network",
        "Topic :: Transformers",
        "Topic :: Large Language Models",
        "Topic :: NLP",
        "Topic :: Natural Language Processing",
    ],
    zip_safe=False,
)
