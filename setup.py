import os
from setuptools import setup, find_packages


with open(os.path.join("src", "cappr", "__init__.py")) as f:
    for line in f:
        if line.startswith("__version__ = "):
            version = str(line.split()[-1].strip('"'))
            break


requirements_base = [
    "numpy>=1.21.0",
    "tqdm>=4.27.0",
]

requirements_openai = [
    "openai>=0.26.0",
    "tiktoken>=0.2.0",
]

requirements_huggingface = [
    "torch>=1.12.1",
    "transformers>=4.26.1",
]

requirements_demos = [
    "datasets>=2.10.0",
    "jupyter>=1.0.0",
    "pandas>=1.5.3",
    "scikit-learn>=1.2.2",
]

requirements_dev = [
    "black>=23.1.0",
    "docutils<0.19",
    "pydata-sphinx-theme>=0.13.1",
    "pytest>=7.2.1",
    "pytest-cov>=4.0.0",
    "sphinx>=6.1.3",
    "sphinx-togglebutton>=0.3.2",
    "sphinxcontrib-napoleon>=0.7",
    "twine>=4.0.2",
]


with open("README.md", mode="r", encoding="utf-8") as f:
    readme = f.read()


setup(
    name="cappr",
    version=version,
    description="Zero-shot text classification using autoregressive language models.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/kddubey/cappr/",
    license="Apache License 2.0",
    python_requires=">=3.8.0",
    install_requires=requirements_base + requirements_openai,
    extras_require={
        "hf": requirements_huggingface,
        "demos": requirements_huggingface + requirements_demos,
        "dev": requirements_huggingface + requirements_demos + requirements_dev,
    },
    author_email="kushdubey63@gmail.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
