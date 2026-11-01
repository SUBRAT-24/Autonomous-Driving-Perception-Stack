import os
from glob import glob
from setuptools import setup, find_packages

package_name = "perception_bringup"

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
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Subrat",
    maintainer_email="subrat@example.com",
    description="Bringup launch files for perception stack",
    license="MIT",
)
