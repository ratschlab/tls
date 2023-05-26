#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

setup_requirements = ['pytest-runner']

setup(
    author="anonymous",
    author_email='anonymous@anonymous.org',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="Temporal label smoothing for ICU data.",
    entry_points={
        "console_scripts": ['tls = tls.run:main']
    },
    install_requires=[],  # dependencies managed via conda for the moment
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='tls',
    name='tls',
    packages=find_packages(include=['tls']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=[],
    url='',
    version='1.0.0',
    zip_safe=False,
)
