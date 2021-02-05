from setuptools import setup
import setuptools

setup(
    name='mlbalance',
    packages=setuptools.find_packages(),
    version='0.1',
    description='',
    long_description='...',
    author='Kilbas Igor, Mukhin Artem, Gribanov Danil',
    author_email='whitemarsstudios@gmail.com',
    url='https://github.com/MakiResearchTeam/MakiFlow.git',
    include_package_data=True,  # This will include all files in MANIFEST.in in the package when installing.
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ], install_requires=[]
)
