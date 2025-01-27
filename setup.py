from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.readlines()

setup(
    author='Xiong Bio Lab',
    url='https://github.com/xiongbiolab/scProca',
    license="MIT license",
    name='scproca',
    version='0.1',
    description='Integrate and generate single-cell proteomics from transcriptomics with cross-attention.',
    packages=find_packages(),
    install_requires=requirements
)
