from setuptools import setup


setup(
    name='crowdnav',
    version='0.0.1',
    packages=[
        'crowd_nav',
        'crowd_nav.configs',
        'crowd_nav.policy',
        'crowd_nav.utils',
        'crowd_sim',
        'crowd_sim.envs',
        'crowd_sim.envs.policy',
        'crowd_sim.envs.utils',
        'Magni',
        'Magni.src'
    ],
    install_requires=[
        'gitpython',
        # 'gym==0.18.0',
        'pyparsing==2.4.7',
        'matplotlib==3.7.5',
        'numpy',
        'scipy',
        'torch',
        'torchvision',
        'pandas',
        'pytest'
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
    },
)
