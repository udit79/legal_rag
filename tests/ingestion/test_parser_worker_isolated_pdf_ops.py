import base64
from pathlib import Path

import pymupdf
import pytest

from src.ingestion.parser_contracts import SourceCoordinates
from src.ingestion.parser_worker import worker
from src.ingestion.parser_worker.result import (
    ImageCropRequest,
    ImageCropResponse,
    ImageCropSpec,
    PdfInspectionRequest,
)
from src.security.ingestion_screening.enums import AdmissionDecision
from src.security.ingestion_screening.file_gate import (
    FileGateError,
    apply_pdf_structure_inspection,
    screen_pdf_file,
)
from src.security.ingestion_screening.policy import ScreeningPolicy
from src.ingestion.parser_worker.limits import pdf_execution_timeout_seconds


class _CleanScanner:
    name = "test"
    version = "1"
    database_version = "1"

    def scan(self, path: Path) -> bool:
        return path.is_file()


def _pdf(path):
    document = pymupdf.open()
    page = document.new_page(width=200, height=200)
    page.draw_rect(pymupdf.Rect(10, 10, 100, 80), color=(1, 0, 0), fill=(1, 0, 0))
    document.save(path)
    document.close()


def _patch_pdf_path(monkeypatch, path):
    def resolve_pdf(filename: str) -> Path:
        assert filename.endswith(".pdf")
        return path

    monkeypatch.setattr(worker, "_pdf_path", resolve_pdf)


def _inspection(path, monkeypatch):
    _patch_pdf_path(monkeypatch, path)
    request = PdfInspectionRequest(
        request_id="c" * 32,
        filename=f"{'c' * 32}.pdf",
        max_pdf_objects=200_000,
    )
    return worker._handle_pdf_inspection(request.model_dump_json().encode())


def _save_pdf_with_open_action(path, action):
    document = pymupdf.open()
    document.new_page()
    if isinstance(action, str):
        document.xref_set_key(document.pdf_catalog(), "OpenAction", action)
    else:
        action_xref = document.get_new_xref()
        document.update_object(action_xref, action)
        document.xref_set_key(
            document.pdf_catalog(), "OpenAction", f"{action_xref} 0 R"
        )
    document.save(path)
    document.close()


def test_pdf_execution_timeout_scales_with_document_size_and_is_capped():
    assert pdf_execution_timeout_seconds(1) == 60
    assert pdf_execution_timeout_seconds(10) == 600
    assert pdf_execution_timeout_seconds(500) == 1800


def test_worker_pdf_inspection_returns_page_and_object_counts(tmp_path, monkeypatch):
    path = tmp_path / "source.pdf"
    _pdf(path)
    _patch_pdf_path(monkeypatch, path)
    request = PdfInspectionRequest(
        request_id="a" * 32,
        filename=f"{'a' * 32}.pdf",
        max_pdf_objects=200_000,
    )

    result = worker._handle_pdf_inspection(request.model_dump_json().encode())

    assert result.inspection_status == "OK"
    assert result.page_count == 1
    assert result.object_count is not None
    assert result.object_count > 0


def test_worker_admits_open_action_with_internal_goto(tmp_path, monkeypatch):
    path = tmp_path / "goto.pdf"
    document = pymupdf.open()
    document.new_page()
    page_xref = document.page_xref(0)
    action_xref = document.get_new_xref()
    document.update_object(
        action_xref, f"<< /S /GoTo /D [{page_xref} 0 R /XYZ null null 1] >>"
    )
    document.xref_set_key(
        document.pdf_catalog(), "OpenAction", f"{action_xref} 0 R"
    )
    document.save(path)
    document.close()

    result = _inspection(path, monkeypatch)

    assert result.inspection_status == "OK"
    assert result.active_content_xrefs == ()


def test_worker_admits_open_action_with_named_local_goto(tmp_path, monkeypatch):
    path = tmp_path / "named-goto.pdf"
    document = pymupdf.open()
    document.new_page()
    page_xref = document.page_xref(0)
    document.xref_set_key(
        document.pdf_catalog(), "Dests", f"<< /start [{page_xref} 0 R /Fit] >>"
    )
    action_xref = document.get_new_xref()
    document.update_object(action_xref, "<< /S /GoTo /D /start >>")
    document.xref_set_key(
        document.pdf_catalog(), "OpenAction", f"{action_xref} 0 R"
    )
    document.save(path)
    document.close()

    result = _inspection(path, monkeypatch)

    assert result.inspection_status == "OK"
    assert result.active_content_xrefs == ()


def test_worker_admits_direct_internal_destination(tmp_path, monkeypatch):
    path = tmp_path / "destination.pdf"
    document = pymupdf.open()
    document.new_page()
    page_xref = document.page_xref(0)
    document.xref_set_key(
        document.pdf_catalog(), "OpenAction", f"[{page_xref} 0 R /Fit]"
    )
    document.save(path)
    document.close()

    result = _inspection(path, monkeypatch)

    assert result.inspection_status == "OK"
    assert result.active_content_xrefs == ()


def test_worker_admits_internal_destination_with_matching_generation(
    tmp_path, monkeypatch
):
    path = tmp_path / "generation-one.pdf"
    document = pymupdf.open()
    document.new_page()
    page_xref = document.page_xref(0)
    document.xref_set_key(
        document.pdf_catalog(), "OpenAction", f"[{page_xref} 1 R /Fit]"
    )
    document.save(path, garbage=0)
    document.close()

    data = path.read_bytes()
    data = data.replace(
        f"{page_xref} 0 R".encode(), f"{page_xref} 1 R".encode()
    ).replace(f"{page_xref} 0 obj".encode(), f"{page_xref} 1 obj".encode())
    lines = data.splitlines(keepends=True)
    xref_line = next(index for index, line in enumerate(lines) if line == b"xref\n")
    entry_line = xref_line + 2 + page_xref
    lines[entry_line] = (
        lines[entry_line][:11] + b"00001" + lines[entry_line][16:]
    )
    path.write_bytes(b"".join(lines))

    result = _inspection(path, monkeypatch)

    assert result.inspection_status == "OK"
    assert result.active_content_xrefs == ()


def test_worker_quarantines_open_action_javascript(tmp_path, monkeypatch):
    path = tmp_path / "javascript.pdf"
    _save_pdf_with_open_action(
        path, "<< /S /JavaScript /JS (app.alert('blocked')) >>"
    )

    result = _inspection(path, monkeypatch)

    assert result.inspection_status == "OK"
    assert result.active_content_xrefs
    scanner = _CleanScanner()
    raw_record = screen_pdf_file(
        path, policy=ScreeningPolicy(), scanner=scanner, filename=path.name
    )
    with pytest.raises(FileGateError) as error:
        apply_pdf_structure_inspection(
            raw_record,
            result,
            policy=ScreeningPolicy(),
            scanner=scanner,
        )
    assert error.value.record.decision == AdmissionDecision.QUARANTINED
    assert error.value.record.justification_codes == (
        "PDF_ACTIVE_CONTENT_OR_EMBEDDED_FILES",
    )


def test_worker_quarantines_javascript_additional_action(tmp_path, monkeypatch):
    path = tmp_path / "additional-action.pdf"
    document = pymupdf.open()
    document.new_page()
    action_xref = document.get_new_xref()
    document.update_object(
        action_xref, "<< /S /JavaScript /JS (app.alert('blocked')) >>"
    )
    document.xref_set_key(
        document.pdf_catalog(), "AA", f"<< /WC {action_xref} 0 R >>"
    )
    document.save(path)
    document.close()

    result = _inspection(path, monkeypatch)

    assert result.inspection_status == "OK"
    assert result.active_content_xrefs


def test_worker_admits_internal_goto_additional_action(tmp_path, monkeypatch):
    path = tmp_path / "additional-goto.pdf"
    document = pymupdf.open()
    document.new_page()
    page_xref = document.page_xref(0)
    action_xref = document.get_new_xref()
    document.update_object(
        action_xref, f"<< /S /GoTo /D [{page_xref} 0 R /Fit] >>"
    )
    document.xref_set_key(
        document.pdf_catalog(), "AA", f"<< /WC {action_xref} 0 R >>"
    )
    document.save(path)
    document.close()

    result = _inspection(path, monkeypatch)

    assert result.inspection_status == "OK"
    assert result.active_content_xrefs == ()


def test_worker_quarantines_javascript_in_chained_action(tmp_path, monkeypatch):
    path = tmp_path / "chained-javascript.pdf"
    document = pymupdf.open()
    document.new_page()
    javascript_xref = document.get_new_xref()
    document.update_object(
        javascript_xref, "<< /S /JavaScript /JS (app.alert('blocked')) >>"
    )
    navigation_xref = document.get_new_xref()
    document.update_object(
        navigation_xref,
        f"<< /S /GoTo /D [{document.page_xref(0)} 0 R /Fit] "
        f"/Next {javascript_xref} 0 R >>",
    )
    document.xref_set_key(
        document.pdf_catalog(), "OpenAction", f"{navigation_xref} 0 R"
    )
    document.save(path)
    document.close()

    result = _inspection(path, monkeypatch)

    assert result.inspection_status == "OK"
    assert result.active_content_xrefs


def test_worker_quarantines_excessively_chained_actions(tmp_path, monkeypatch):
    path = tmp_path / "deep-actions.pdf"
    document = pymupdf.open()
    document.new_page()
    page_xref = document.page_xref(0)
    chain = [document.get_new_xref() for _ in range(34)]
    for index, action_xref in enumerate(chain):
        next_action = (
            f" /Next {chain[index + 1]} 0 R" if index + 1 < len(chain) else ""
        )
        document.update_object(
            action_xref,
            f"<< /S /GoTo /D [{page_xref} 0 R /Fit]{next_action} >>",
        )
    document.xref_set_key(
        document.pdf_catalog(), "OpenAction", f"{chain[0]} 0 R"
    )
    document.save(path)
    document.close()

    result = _inspection(path, monkeypatch)

    assert result.inspection_status == "OK"
    assert result.active_content_xrefs


def test_worker_quarantines_embedded_file(tmp_path, monkeypatch):
    path = tmp_path / "embedded.pdf"
    document = pymupdf.open()
    document.new_page()
    document.embfile_add("payload.txt", b"sample attachment")
    document.save(path)
    document.close()

    result = _inspection(path, monkeypatch)

    assert result.inspection_status == "OK"
    assert result.active_content_xrefs


@pytest.mark.parametrize(
    "action",
    (
        "<< /S /URI /URI (https://example.test) >>",
        "<< /S /Launch /F (program.exe) >>",
        "<< /S /GoTo >>",
        "null",
    ),
)
def test_worker_quarantines_unsupported_or_malformed_open_action(
    tmp_path, monkeypatch, action
):
    path = tmp_path / "unsupported-action.pdf"
    _save_pdf_with_open_action(path, action)

    result = _inspection(path, monkeypatch)

    assert result.inspection_status == "OK"
    assert result.active_content_xrefs


def test_worker_quarantines_unresolved_open_action_reference(
    tmp_path, monkeypatch
):
    path = tmp_path / "unresolved-action.pdf"
    document = pymupdf.open()
    document.new_page()
    document.xref_set_key(
        document.pdf_catalog(), "OpenAction", "9999 0 R"
    )
    document.save(path)
    document.close()

    result = _inspection(path, monkeypatch)

    assert result.inspection_status == "OK"
    assert result.active_content_xrefs


def test_worker_quarantines_open_action_with_mismatched_generation(
    tmp_path, monkeypatch
):
    path = tmp_path / "mismatched-generation.pdf"
    document = pymupdf.open()
    document.new_page()
    page_xref = document.page_xref(0)
    document.xref_set_key(
        document.pdf_catalog(), "OpenAction", f"[{page_xref} 99 R /Fit]"
    )
    document.save(path)
    document.close()

    result = _inspection(path, monkeypatch)

    assert result.inspection_status == "OK"
    assert result.active_content_xrefs


def test_worker_reports_structure_inspection_failure_as_failed(tmp_path, monkeypatch):
    path = tmp_path / "inspection-failure.pdf"
    _pdf(path)

    def fail_inspection(_document):
        assert _document.page_count == 1
        raise RuntimeError("inspection failed")

    monkeypatch.setattr(worker, "inspect_pdf_active_content", fail_inspection)
    result = _inspection(path, monkeypatch)

    assert result.inspection_status == "FAILED"
    assert result.error_code == "RuntimeError"


def test_worker_admits_pdf_without_actions(tmp_path, monkeypatch):
    path = tmp_path / "plain.pdf"
    _pdf(path)

    result = _inspection(path, monkeypatch)

    assert result.inspection_status == "OK"
    assert result.active_content_xrefs == ()


def test_worker_ignores_action_like_text_in_content_and_metadata(
    tmp_path, monkeypatch
):
    path = tmp_path / "action-like-text.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (20, 20), "/OpenAction /JavaScript /AA /Launch /EmbeddedFiles"
    )
    document.set_metadata(
        {"title": "/OpenAction /JavaScript /AA /Launch /EmbeddedFiles"}
    )
    document.save(path)
    document.close()

    result = _inspection(path, monkeypatch)

    assert result.inspection_status == "OK"
    assert result.active_content_xrefs == ()


def test_worker_admits_cyber_security_pdf_internal_navigation(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "test_data" / "cyber_security.pdf"
    if not path.is_file():
        pytest.skip("test_data/cyber_security.pdf is unavailable")

    result = _inspection(path, monkeypatch)

    assert result.inspection_status == "OK"
    assert result.active_content_xrefs == ()


def test_worker_crops_an_image_region_and_returns_provenance(tmp_path, monkeypatch):
    path = tmp_path / "source.pdf"
    _pdf(path)
    _patch_pdf_path(monkeypatch, path)
    request = ImageCropRequest(
        request_id="b" * 32,
        filename=f"{'b' * 32}.pdf",
        images=(
            ImageCropSpec(
                page_number=1,
                block_id="page-1-block-0",
                reading_order=0,
                source_coordinates=SourceCoordinates(
                    x0=10,
                    y0=10,
                    x1=100,
                    y1=80,
                    unit="PDF_POINT",
                    origin="TOP_LEFT",
                ),
            ),
        ),
    )

    result = worker._handle_crop_images(request.model_dump_json().encode())

    assert isinstance(result, ImageCropResponse)
    image = result.images[0]
    crop_bytes = base64.b64decode(image.data_base64)
    assert crop_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert image.block_id == "page-1-block-0"
    assert image.source_coordinates == request.images[0].source_coordinates
