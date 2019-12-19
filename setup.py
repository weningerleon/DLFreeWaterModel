import setuptools

with open('README.md') as f:
    README = f.read()

setuptools.setup(
    author="Leon Weninger",
    author_email="leon.weninger@lfb.rwth-aachen.de",
    name='difreewater',
    license="MIT",
    description='Small package for free water elimination in mri diffusion images',
    version='v0.0.1',
    long_description=README,
    url='https://github.com/weningerleon/difreewater',
    packages=setuptools.find_packages(),
    python_requires=">=3.5",
)