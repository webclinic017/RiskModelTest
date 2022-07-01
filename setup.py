import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="risk_barra_model", # Replace with your own username
    version="0.0.1",
    author="Lam Nguyen",
    author_email="lam.nguyen@pysparks.com",
    description="Risk Barra Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VietQuant/RiskBarraModel",
    packages=["rework_backtrader"] + setuptools.find_packages(include=["rework_backtrader.*", "rework_backtrader"]),
    package_dir = {"rework_backtrader" : "rework_backtrader"},
    package_data={"rework_backtrader" : ["metadata/FDMT/mapping_normal_bank.csv"]},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)