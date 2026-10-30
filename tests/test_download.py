import io
import json
import stat
import zipfile

import pytest

from name_that_llm import download


def dataset_zip():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        csv = "text,model,prompt_id,split\nSynthetic text.,gpt-5.5,n001,new\n"
        archive.writestr("passages.csv", csv)
        archive.writestr("passages_clean.csv", csv)
        archive.writestr("README.md", "Synthetic dataset fixture, not competition data.")
    return stream.getvalue()


def test_public_download_and_cached_verification(tmp_path, monkeypatch):
    requests = []

    def urlopen(request, *, timeout):
        requests.append(request)
        assert timeout > 0
        assert "Authorization" not in request.headers
        assert download.DATASET in request.full_url
        assert "datasetVersionNumber=7" in request.full_url
        return io.BytesIO(dataset_zip())

    monkeypatch.setattr(download, "urlopen", urlopen)
    destination = tmp_path / "download"
    manifest = download.download_dataset(destination)
    assert manifest["dataset"] == download.DATASET
    assert manifest["version"] == 7
    assert manifest["clean_rows"] == 1
    assert download.REQUIRED_FILES.issubset(manifest["files_sha256"])
    assert download.download_dataset(destination) == manifest
    assert len(requests) == 1  # Reuse must verify hashes, not download again.
    (destination / "passages_clean.csv").write_text("changed")
    with pytest.raises(ValueError, match="modified downloaded file"):
        download.download_dataset(destination)


def test_preserves_existing_directory(tmp_path):
    with pytest.raises(ValueError, match="already exists"):
        download.download_dataset(tmp_path)


def test_rejects_different_version(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "urlopen", lambda *args, **kwargs: io.BytesIO(dataset_zip()))
    destination = tmp_path / "download"
    download.download_dataset(destination)
    with pytest.raises(ValueError, match="different source or version"):
        download.download_dataset(destination, version=8)


def test_rejects_incomplete_manifest(tmp_path):
    manifest = {"dataset": download.DATASET, "version": 7, "files_sha256": {}}
    (tmp_path / "download_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing required files"):
        download.verify_download(tmp_path, 7)


@pytest.mark.parametrize("path", ["../outside.txt", "/outside.txt", "..\\outside.txt"])
def test_rejects_unsafe_archives_before_extracting(tmp_path, path):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("safe.txt", "safe")
        archive.writestr(path, "unsafe")
    destination = tmp_path / "extracted"
    with pytest.raises(ValueError, match="Unsafe archive path"):
        download.extract_archive(archive_path, destination)
    assert not destination.exists()
    assert not (tmp_path / "outside.txt").exists()


def test_rejects_symlinks(tmp_path):
    archive_path = tmp_path / "symlink.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        link = zipfile.ZipInfo("link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, "../outside.txt")
    with pytest.raises(ValueError, match="Unsafe archive path"):
        download.extract_archive(archive_path, tmp_path / "extracted")


def test_html_response_does_not_leave_partial_download(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "urlopen", lambda *args, **kwargs: io.BytesIO(b"<html>"))
    destination = tmp_path / "download"
    with pytest.raises(ValueError, match="did not return a ZIP"):
        download.download_dataset(destination)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []
