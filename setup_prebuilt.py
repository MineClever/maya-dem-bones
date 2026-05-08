# Python 2/3 compatible setup script for packaging a pre-built C extension.
# Used by the Python 2.7 build path (cmake-direct) where scikit-build is not available.
# The .pyd must already be present in src/dem_bones/ before running this script.
#
# Notes:
#   - Explicitly sets name/version because older build tools do not auto-read setup.cfg.
#   - BinaryDistribution forces bdist_wheel to produce a platform-specific wheel
#     (e.g. cp27-cp27-win_amd64) rather than a pure-Python wheel (py2-none-any).
#   - Maya Python 2.7 installs often lack setuptools, so install mode falls back
#     to the standard-library distutils package.
try:
    from setuptools import setup
    from setuptools.dist import Distribution
except ImportError:
    from distutils.core import setup
    from distutils.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(
    name="dem_bones",
    version="1.0.0",
    packages=["dem_bones"],
    package_dir={"": "src"},
    package_data={"dem_bones": ["*.pyd", "*.pyi"]},
    distclass=BinaryDistribution,
)
