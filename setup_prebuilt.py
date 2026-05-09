# Python 2/3 compatible setup script for packaging a pre-built C extension.
# Used by the Python 2.7 build path (cmake-direct) where scikit-build is not available.
# The .pyd must already be present in src/dem_bones/ before running this script.
#
# Notes:
#   - Explicitly sets name/version because older build tools do not auto-read setup.cfg.
#   - Maya Python 2.7 installs often lack pip/wheel/setuptools, so this file
#     provides a small bdist_wheel command that packages the pre-built files
#     directly using only the standard library.
import base64
import hashlib
import os
import sys
import zipfile

try:
    from setuptools import setup
    from setuptools.dist import Distribution
except ImportError:
    from distutils.core import setup
    from distutils.dist import Distribution
from distutils.cmd import Command
from distutils.util import get_platform


NAME = "dem_bones"
VERSION = "1.0.0"


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class BDistWheel(Command):
    description = "build a wheel from the pre-built Maya extension"
    user_options = []

    def initialize_options(self):
        self.dist_dir = None

    def finalize_options(self):
        self.dist_dir = os.path.join(os.path.dirname(__file__), "dist")

    def run(self):
        package_dir = os.path.join(os.path.dirname(__file__), "src", NAME)
        if not os.path.isdir(package_dir):
            raise RuntimeError("package directory not found: %s" % package_dir)

        pyd_files = [f for f in os.listdir(package_dir) if f.lower().endswith(".pyd")]
        if not pyd_files:
            raise RuntimeError("pre-built _core.pyd not found in %s" % package_dir)

        if not os.path.isdir(self.dist_dir):
            os.makedirs(self.dist_dir)

        py_tag = "cp%s%s" % (sys.version_info[0], sys.version_info[1])
        platform_tag = get_platform().replace("-", "_").replace(".", "_")
        wheel_tag = "%s-none-%s" % (py_tag, platform_tag)
        dist_info = "%s-%s.dist-info" % (NAME, VERSION)
        wheel_name = "%s-%s-%s.whl" % (NAME, VERSION, wheel_tag)
        wheel_path = os.path.join(self.dist_dir, wheel_name)

        records = []
        with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename in sorted(os.listdir(package_dir)):
                source_path = os.path.join(package_dir, filename)
                if os.path.isdir(source_path):
                    continue
                if filename.endswith((".pyc", ".pyo")):
                    continue
                arcname = "%s/%s" % (NAME, filename)
                self._write_file(zf, source_path, arcname, records)

            self._write_text(zf, "%s/METADATA" % dist_info, self._metadata(), records)
            self._write_text(zf, "%s/WHEEL" % dist_info, self._wheel_metadata(wheel_tag), records)

            record_name = "%s/RECORD" % dist_info
            record_body = "".join(records)
            record_body += "%s,,\n" % record_name
            zf.writestr(record_name, record_body)

        print("created %s" % wheel_path)

    def _metadata(self):
        return (
            "Metadata-Version: 2.1\n"
            "Name: %s\n"
            "Version: %s\n"
            "Summary: DemBones python bindings for use in Maya.\n"
            "\n"
        ) % (NAME, VERSION)

    def _wheel_metadata(self, wheel_tag):
        return (
            "Wheel-Version: 1.0\n"
            "Generator: setup_prebuilt.py\n"
            "Root-Is-Purelib: false\n"
            "Tag: %s\n"
            "\n"
        ) % wheel_tag

    def _write_file(self, zf, source_path, arcname, records):
        with open(source_path, "rb") as handle:
            data = handle.read()
        zf.writestr(arcname, data)
        records.append("%s,sha256=%s,%d\n" % (arcname, self._digest(data), len(data)))

    def _write_text(self, zf, arcname, text, records):
        data = text.encode("utf-8")
        zf.writestr(arcname, data)
        records.append("%s,sha256=%s,%d\n" % (arcname, self._digest(data), len(data)))

    def _digest(self, data):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
        if not isinstance(digest, str):
            digest = digest.decode("ascii")
        return digest.rstrip("=")


setup(
    name=NAME,
    version=VERSION,
    packages=["dem_bones"],
    package_dir={"": "src"},
    package_data={"dem_bones": ["*.pyd", "*.pyi"]},
    distclass=BinaryDistribution,
    cmdclass={"bdist_wheel": BDistWheel},
)
