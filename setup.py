import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    # long_description=long_description,
    name="DSC-Believer",
    packages=setuptools.find_packages(),
    install_requires = [
        "datasets",
        "gdown",
        "sentence-transformers",
        "tqdm",
        "numpy",
        "torch",
        "ipywidgets",
        "pyparsing"
    ],
    # dependency_links=[
    #     'https://download.pytorch.org/whl/cu121'
    # ]
)