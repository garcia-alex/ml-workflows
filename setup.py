import os
import sys
import warnings

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from distutils.command.build_py import build_py

path, script = os.path.split(sys.argv[0])
os.chdir(os.path.abspath(path))

install_requires = [
    'numpy==1.16.1',
    'scikit-learn==0.20.2',
    'scipy==1.2.0'
]

if sys.version_info < (3, 6):
    warnings.warn(
        'Python versions less than 3.6 are not supported '
        'If you have any questions, please file an issue on Github ',
        DeprecationWarning)
    install_requires.append('pyjwt==1.4.0')
else:
    install_requires.append('pyjwt==1.4.0')

setup(
    name='ml-workflow',
    cmdclass={'build_py': build_py},
    version="0.5.5",
    description='Machine learning model evalutation workflow',
    long_description="Machine learning model evalutation workflow "
                     "at https://github.com/kintisheff/ml-workflow",
    author='Tsvetan Kintisheff',
    author_email='kintisheff@gmail.com',
    url='https://github.com/kintisheff/ml-workflow',
    packages=['ml-workflow', ],
    install_requires=install_requires,
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Software Development :: Libraries :: Python Modules",
])