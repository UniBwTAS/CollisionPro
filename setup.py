from setuptools import setup, find_packages

setup(
    name='CollisionPro',
    version="1.0.0",
    packages=["collisionpro"],
    url='https://github.com/UniBwTAS/CollisionPro',
    author='Thomas Steinecker',
    author_email='thomas.steinecker@unibw.de',
    include_package_data=True,
    install_requires=[
        'numpy',
        'matplotlib',
        'tensorflow',
    ],
)
