import hashlib
from pathlib import Path
import tempfile
import tarfile
import unittest
from tools.release.package import archive, verify


class ReleaseTests(unittest.TestCase):
    def test_reproducible_archive_and_download_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "source.txt").write_text("fixed source")
            archive(bundle, root / "one.tar.xz", 100)
            archive(bundle, root / "two.tar.xz", 100)
            self.assertTrue((root / "one.tar.xz").read_bytes().startswith(b"\xfd7zXZ\0"))
            with tarfile.open(root / "one.tar.xz") as stream:
                entry = stream.getmember("bundle/source.txt")
                self.assertEqual((entry.uid, entry.gid, entry.mtime), (0, 0, 100))
            self.assertEqual((root / "one.tar.xz").read_bytes(), (root / "two.tar.xz").read_bytes())
            digest = hashlib.sha256((root / "one.tar.xz").read_bytes()).hexdigest()
            (root / "SHA256SUMS").write_text(f"{digest}  one.tar.xz\n")
            verify(root)
            (root / "one.tar.xz").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify(root)
