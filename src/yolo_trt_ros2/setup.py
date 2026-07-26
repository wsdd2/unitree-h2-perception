from setuptools import setup

package_name = 'yolo_trt_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name, package_name + '.backends'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/yolo_detector.launch.py',
            'launch/inspection_perception.launch.py',
        ]),
        ('share/' + package_name + '/config', [
            'config/detector.yaml',
            'config/inspection_perception.yaml',
            'config/cabinet_controls_classes.txt',
            'config/control_labels_zh.txt',
            'config/yoloe_classes.example.txt',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='MscapeTech',
    maintainer_email='TODO@example.com',
    description='Lightweight ROS2 Foxy YOLO detector node for Jetson Orin NX.',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'cabinet_automation_node = yolo_trt_ros2.cabinet_automation_node:main',
            'direct_realsense_node = yolo_trt_ros2.direct_realsense_node:main',
            'integrated_perception_node = yolo_trt_ros2.integrated_perception_node:main',
            'yolo_detector_node = yolo_trt_ros2.yolo_detector_node:main',
            'coordinate_projector_node = yolo_trt_ros2.coordinate_projector_node:main',
            'benchmark_topics_node = yolo_trt_ros2.benchmark_topics_node:main',
            'web_dashboard_node = yolo_trt_ros2.web_dashboard_node:main',
        ],
    },
)
