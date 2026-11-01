from setuptools import setup, find_packages

package_name = "viz_node"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name] if False else [],
        ),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Subrat",
    maintainer_email="subrat@example.com",
    description="Visualization ROS2 node",
    license="MIT",
    entry_points={
        "console_scripts": [
            "viz_node = viz_node.viz_node:main",
        ],
    },
)
