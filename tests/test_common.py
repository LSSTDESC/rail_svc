"""Tests for rail_svc.common utilities."""

from pathlib import Path
from unittest.mock import patch

import pytest

from rail_svc.common import (
    LoadType,
    copy_file_to_archive,
    handle_file,
    link_file_to_archive,
    slice_to_str,
    str_to_slice,
)


class TestStrToSlice:
    """Tests for str_to_slice."""

    def test_none_returns_none(self):
        assert str_to_slice(None) is None

    def test_single_number(self):
        result = str_to_slice("5")
        assert result == slice(5, None, None)

    def test_start_stop(self):
        result = str_to_slice("1:5")
        assert result == slice(1, 5, None)

    def test_start_stop_step(self):
        result = str_to_slice("1:10:2")
        assert result == slice(1, 10, 2)

    def test_no_start(self):
        result = str_to_slice(":5")
        assert result == slice(None, 5, None)

    def test_no_stop(self):
        result = str_to_slice("5:")
        assert result == slice(5, None, None)

    def test_only_step(self):
        result = str_to_slice("::2")
        assert result == slice(None, None, 2)

    def test_empty_parts(self):
        result = str_to_slice("::")
        assert result == slice(None, None, None)

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid slice format"):
            str_to_slice("1:2:3:4")


class TestSliceToStr:
    """Tests for slice_to_str."""

    def test_none_returns_none(self):
        assert slice_to_str(None) is None

    def test_start_and_stop(self):
        result = slice_to_str(slice(1, 5, None))
        assert result == "1:5"

    def test_start_stop_step(self):
        result = slice_to_str(slice(1, 10, 2))
        assert result == "1:10:2"

    def test_no_start(self):
        result = slice_to_str(slice(None, 5, None))
        assert result == ":5"

    def test_no_stop(self):
        result = slice_to_str(slice(5, None, None))
        assert result == "5:"

    def test_only_step(self):
        result = slice_to_str(slice(None, None, 2))
        assert result == "::2"


class TestCopyFileToArchive:
    """Tests for copy_file_to_archive."""

    def test_copies_file(self, tmp_path):
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()

        source = tmp_path / "source.txt"
        source.write_text("hello")

        with patch("rail_svc.common.global_config.storage.archive", str(archive_dir)):
            result = copy_file_to_archive(source, "dest.txt")

        assert result == Path("dest.txt")
        assert (archive_dir / "dest.txt").read_text() == "hello"

    def test_preserves_content(self, tmp_path):
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()

        source = tmp_path / "data.bin"
        source.write_bytes(b"\x00\x01\x02\x03")

        with patch("rail_svc.common.global_config.storage.archive", str(archive_dir)):
            copy_file_to_archive(source, "data.bin")

        assert (archive_dir / "data.bin").read_bytes() == b"\x00\x01\x02\x03"


class TestLinkFileToArchive:
    """Tests for link_file_to_archive."""

    def test_creates_symlink(self, tmp_path):
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()

        source = tmp_path / "source.txt"
        source.write_text("hello")

        with patch("rail_svc.common.global_config.storage.archive", str(archive_dir)):
            result = link_file_to_archive(source, "linked.txt")

        assert result == Path("linked.txt")
        link_path = archive_dir / "linked.txt"
        assert link_path.is_symlink()


class TestHandleFile:
    """Tests for handle_file."""

    def test_in_place_returns_original_path(self, tmp_path):
        source = tmp_path / "file.txt"
        source.write_text("data")

        result = handle_file(source, "dest.txt", LoadType.in_place)
        assert result == source

    def test_copy_mode(self, tmp_path):
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()

        source = tmp_path / "file.txt"
        source.write_text("data")

        with patch("rail_svc.common.global_config.storage.archive", str(archive_dir)):
            result = handle_file(source, "dest.txt", LoadType.copy)

        assert result == Path("dest.txt")
        assert (archive_dir / "dest.txt").read_text() == "data"

    def test_link_mode(self, tmp_path):
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()

        source = tmp_path / "file.txt"
        source.write_text("data")

        with patch("rail_svc.common.global_config.storage.archive", str(archive_dir)):
            result = handle_file(source, "linked.txt", LoadType.link)

        assert result == Path("linked.txt")
        assert (archive_dir / "linked.txt").is_symlink()
