# flake8: noqa

from setuptools import setup

with open("README.md") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

version = exec(open("evalem/__init__.py").read())

setup(
    name="evalem",
    version=version,
    description="An evaluation framework for your NLP pipelines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NASA-IMPACT/evalem",
    author_email="np0069@uah.edu",
    python_requires=">=3.8",
    packages=[
        "evalem",
        "evalem.nlp",
        "evalem.nlp.evaluators",
        "evalem.nlp.metrics",
        "evalem.nlp.misc",
        "evalem.nlp.models",
        "evalem.cv",
        "evalem.cv.evaluators",
        "evalem.cv.metrics",
        "evalem.cv.models",
        "evalem.cv.misc",
        "evalem.cv.pipelines",
        "evalem.misc",
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
