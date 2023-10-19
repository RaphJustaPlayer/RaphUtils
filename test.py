from setuptools import setup

setup(
    name='raphutils',
    version='0.1.0',
    description='test',
    url='https://github.com/shuds13/pyexample',
    author='Raphaël Ribes',
    author_email='raphael.ribes@gmail.com',
    license='BSD+Patent',
    packages=['utils'],
    install_requires=['scipy',
                      'numpy',
                      'matplotlib',
                      'pandas',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
    ],
)
