import os
from glob import glob
from setuptools import setup

package_name = 'tg_rviz'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.py')),
        (os.path.join('share', package_name), glob('urdf/*')),
        (os.path.join('share', package_name), glob('meshes/*')),
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
            'state_publisher = tg_rviz.state_publisher:main',
            'state_listener = tg_rviz.state_listener:main',
            'service_joint = tg_rviz.service_joint:main',
            'client_joint = tg_rviz.joint_client:main',
            'training_examples_creator = tg_rviz.training_examples_creator:main',
            'nn_client = tg_rviz.nn_client:main'
        ],
    },
)
