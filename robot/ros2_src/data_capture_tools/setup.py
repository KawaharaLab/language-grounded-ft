from glob import glob
from setuptools import find_packages, setup

package_name = 'data_capture_tools'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        (
            'share/' + package_name + '/config',
            glob('config/*.yaml'),
        ),
        (
            'share/' + package_name + '/launch',
            glob('launch/*.launch.py'),
        ),
    ],
    install_requires=['setuptools', 'PyYAML', 'numpy', 'openai'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='kazuki.takahashi@akg.t.u-tokyo.ac.jp',
    description='Record rosbag2 sessions and export datasets for teleoperation experiments.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'data_capture_manager = data_capture_tools.data_capture_node:main',
            (
                'autonomous_motion_capture = '
                'data_capture_tools.autonomous_motion_capture_node:main'
            ),
            (
                'height_limit_check = '
                'data_capture_tools.height_limit_check_node:main'
            ),
            (
                'key_motion_capture = '
                'data_capture_tools.key_motion_capture_node:main'
            ),
            (
                'model_place_capture = '
                'data_capture_tools.model_place_capture_node:main'
            ),
            (
                'two_surface_selection_capture = '
                'data_capture_tools.two_surface_selection_capture_node:main'
            ),
            (
                'fz_baseline_capture = '
                'data_capture_tools.fz_baseline_capture_node:main'
            ),
            'bag_to_dataset = data_capture_tools.bag_converter:cli_main',
        ],
    },
)
