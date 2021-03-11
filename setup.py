import setuptools

setuptools.setup(
    name="pyCARPool",
    version="0.1.7",
    author="Nicolas Chartier",
    author_email="nicolas.chartier412@gmail.com",
    description="Provides custom class and functions to implement CARPool estimates, stemming from the control variates principle (arXiv 2009.08970)",
    long_description=open('README.md').read(),
    packages=setuptools.find_packages(),
    install_requires=["numpy", "scipy", "tqdm"],
    classifiers=[
        'Programming Language :: Python',
        "License :: OSI Approved :: MIT License",
    ],
)