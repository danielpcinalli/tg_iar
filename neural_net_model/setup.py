import os
from glob import glob
from setuptools import setup

package_name = 'neural_net_model'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('model/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='osboxes',
    maintainer_email='osboxes@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'nn_client = neural_net_model.nn_client:main',
            'nn_service = neural_net_model.nn_service:main'
        ],
    },
)
