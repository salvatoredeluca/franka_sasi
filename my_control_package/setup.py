from setuptools import find_packages, setup

package_name = 'my_control_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
             'example = my_control_package.example:main',
             'pinocchio_ex = my_control_package.pinocchio_ex:main',
             'task1 = my_control_package.task1:main',
             'opti_ex = my_control_package.opti_ex:main',
             'OcclusionTask = my_control_package.OcclusionTask:main',
             'CollisionsTask1 = my_control_package.CollisionsTask1:main',
             'EdgesDetection = my_control_package.EdgesDetection:main'

        ],
    },
)
