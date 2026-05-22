from hogfm.utils_optional import dependency_status


def test_optional_dependency_status_does_not_import_package() -> None:
    status = dependency_status("triton")
    assert status.name == "triton"
    assert isinstance(status.available, bool)
