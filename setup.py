from setuptools import setup

setup(
    name='hdc',
    version='0.1.0',
    description='tool to visualise the highest density region (HDR) for MCMC samples',
    url='https://github.com/yqiuu/hd-region',
    author='Yisheng Qiu',
    author_email='yishengq@student.unimelb.edu.au',
    license='MIT',
    packages=['hdc'],
    install_requires=['numpy', 'scipy', 'matplotlib', 'pandas', 'seaborn']
)
