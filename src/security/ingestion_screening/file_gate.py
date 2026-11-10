import hashlib
import os
import shlex
import subprocess
from pathlib import Path
from typing import Protocol
from src.ingestion.parser_worker.result import PdfInspectionResponse
from src.security.ingestion_screening.contracts import FileScreeningRecord, ScreeningFlag
from src.security.ingestion_screening.enums import AdmissionDecision,DetectionView,FlagSeverity,ScreeningStage
from src.security.ingestion_screening.policy import ScreeningPolicy

class MalwareScanner(Protocol):
    name: str
    version: str
    database_version: str

    def scan(self, path: Path) -> bool:
        """Return true only when the scanner confirms the file is clean."""
        ...

class ClamAVScanner:
    """Adapter for the installed ClamAV `clamscan` command-line scanner."""

    def __init__(self, command: str | None = None, timeout_seconds: int = 60) -> None:
        bundled = (
            Path(__file__).resolve().parents[3]
            / "tools"
            / "clamav"
            / "1.5.4"
            / "windows-x64"
            / "clamscan.exe"
        )
        default_command = (
            str(bundled)
            if os.name == "nt" and bundled.is_file()
            else "clamscan"
        )
        configured_binary = os.environ.get("CLAMAV_BINARY", "").strip()
        environment_command = os.environ.get("CLAMAV_SCAN_COMMAND")
        if command is None and configured_binary:
            self.command = [configured_binary]
        elif command is None and environment_command is None:
            self.command = [default_command]
        else:
            configured = command or environment_command or default_command
            self.command = (
                [part.strip('"') for part in shlex.split(configured, posix=False)]
                if os.name == "nt"
                else shlex.split(configured)
            )
        if not self.command:
            raise ValueError("CLAMAV_SCAN_COMMAND must name a scanner executable")
        self.timeout_seconds = timeout_seconds
        self.name = "ClamAV"
        self.version = "unknown"
        self.database_version = "unknown"
        configured_database = os.environ.get("CLAMAV_DATABASE_DIRECTORY", "").strip()
        self.database_directory = (
            Path(configured_database)
            if configured_database
            else (
                bundled.parent / "database"
                if self.command == [str(bundled)]
                else None
            )
        )

    def _database_args(self) -> list[str]:
        if self.database_directory is None:
            return []
        if not self.database_directory.is_dir():
            raise FileNotFoundError(
                f"ClamAV database directory does not exist: {self.database_directory}"
            )
        return [f"--database={self.database_directory}"]

    def initialize(self) -> None:
        result = subprocess.run(
            [*self.command, *self._database_args(), "--version"],
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("ClamAV version check failed")
        output = result.stdout.strip()
        engine_and_database = output.split("/", 1)
        self.version = engine_and_database[0][:128] or "unknown"
        self.database_version = (
            engine_and_database[1][:128] if len(engine_and_database) == 2 else "unknown"
        )
        if self.database_version == "unknown":
            raise RuntimeError("ClamAV signature database version is unavailable")

    def scan(self, path: Path) -> bool:
        result = subprocess.run(
            [*self.command, *self._database_args(), "--no-summary", str(path)],
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
        raise RuntimeError("ClamAV scan failed")

class FileGateError(ValueError):
    """Raised when G0 blocks a file before any parser can receive it."""

    def __init__(self, message: str, record: FileScreeningRecord) -> None:
        super().__init__(message)
        self.record = record

def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while data := source.read(1024 * 1024):
            size += len(data)
            digest.update(data)
    return digest.hexdigest(), size

def _record(
    *,
    source_sha256: str | None,
    file_size_bytes: int,
    policy: ScreeningPolicy,
    decision: AdmissionDecision,
    flags: tuple[ScreeningFlag, ...],
    justification: str,
    page_count: int | None = None,
    scanner: MalwareScanner | None = None,
) -> FileScreeningRecord:
    return FileScreeningRecord(
        source_sha256=source_sha256,
        file_size_bytes=file_size_bytes,
        page_count=page_count,
        decision=decision,
        flags=flags,
        scanner=scanner.name if scanner else None,
        scanner_version=scanner.version if scanner else None,
        scanner_database_version=(
            getattr(scanner, "database_version", "unknown") if scanner else None
        ),
        screening_policy_version=policy.version,
        screening_policy_digest=policy.digest(),
        justification_codes=(justification,),
    )

def screen_pdf_file(
    path: Path,
    *,
    policy: ScreeningPolicy,
    scanner: MalwareScanner,
    filename: str | None = None,
) -> FileScreeningRecord:
    """Check raw PDF identity and malware before isolated PDF inspection.

    This function deliberately does not parse or open the PDF. The parser
    worker performs page, object, encryption, and active-content inspection.
    """
    file_size = path.stat().st_size
    if file_size > policy.max_file_bytes:
        flag = ScreeningFlag(
            code="FILE_SIZE_LIMIT_EXCEEDED",
            stage=ScreeningStage.G0_FILE,
            severity=FlagSeverity.HIGH,
            detection_view=DetectionView.RAW,
        )
        record = _record(
            source_sha256=None,
            file_size_bytes=file_size,
            policy=policy,
            decision=AdmissionDecision.REJECTED,
            flags=(flag,),
            justification="FILE_SIZE_LIMIT_EXCEEDED",
            scanner=scanner,
        )
        raise FileGateError("PDF exceeds the configured file-size limit", record)

    source_sha256, file_size = _hash_file(path)
    claimed_name = filename if filename is not None else path.name
    if Path(claimed_name).suffix.casefold() != ".pdf":
        flag = ScreeningFlag(
            code="FILE_EXTENSION_MISMATCH",
            stage=ScreeningStage.G0_FILE,
            severity=FlagSeverity.HIGH,
            detection_view=DetectionView.RAW,
        )
        record = _record(
            source_sha256=source_sha256,
            file_size_bytes=file_size,
            policy=policy,
            decision=AdmissionDecision.REJECTED,
            flags=(flag,),
            justification="FILE_EXTENSION_MISMATCH",
            scanner=scanner,
        )
        raise FileGateError("Uploaded filename must use the .pdf extension", record)

    with path.open("rb") as source:
        magic = source.read(5)
    if magic != b"%PDF-":
        flag = ScreeningFlag(
            code="PDF_MAGIC_MISMATCH",
            stage=ScreeningStage.G0_FILE,
            severity=FlagSeverity.CRITICAL,
            detection_view=DetectionView.RAW,
        )
        record = _record(
            source_sha256=source_sha256,
            file_size_bytes=file_size,
            policy=policy,
            decision=AdmissionDecision.REJECTED,
            flags=(flag,),
            justification="PDF_MAGIC_MISMATCH",
            scanner=scanner,
        )
        raise FileGateError("File extension and PDF magic bytes do not match", record)

    try:
        is_clean = scanner.scan(path)
    except Exception as exc:
        flag = ScreeningFlag(
            code="MALWARE_SCANNER_FAILURE",
            stage=ScreeningStage.G0_FILE,
            severity=FlagSeverity.CRITICAL,
            detection_view=DetectionView.RAW,
            matched_value=type(exc).__name__,
        )
        record = _record(
            source_sha256=source_sha256,
            file_size_bytes=file_size,
            policy=policy,
            decision=AdmissionDecision.QUARANTINED,
            flags=(flag,),
            justification="MALWARE_SCANNER_FAILURE",
            scanner=scanner,
        )
        raise FileGateError("Malware scanner failed; file is blocked", record) from exc
    if not is_clean:
        flag = ScreeningFlag(
            code="MALWARE_DETECTED",
            stage=ScreeningStage.G0_FILE,
            severity=FlagSeverity.CRITICAL,
            detection_view=DetectionView.RAW,
        )
        record = _record(
            source_sha256=source_sha256,
            file_size_bytes=file_size,
            policy=policy,
            decision=AdmissionDecision.REJECTED,
            flags=(flag,),
            justification="MALWARE_DETECTED",
            scanner=scanner,
        )
        raise FileGateError("Malware scanner detected a threat", record)

    return _record(
        source_sha256=source_sha256,
        file_size_bytes=file_size,
        policy=policy,
        decision=AdmissionDecision.ADMITTED,
        flags=(),
        justification="G0_RAW_FILE_CHECKS_PASSED",
        scanner=scanner,
    )

def apply_pdf_structure_inspection(
    record: FileScreeningRecord,
    inspection: PdfInspectionResponse,
    *,
    policy: ScreeningPolicy,
    scanner: MalwareScanner,
) -> FileScreeningRecord:
    """Complete G0 using PDF structure inspected inside the parser worker."""
    if record.decision != AdmissionDecision.ADMITTED:
        raise ValueError("Structure inspection requires clean raw-file checks")

    def block(
        code: str,
        message: str,
        decision: AdmissionDecision,
        matched_value: str | None = None,
    ) -> None:
        failed = _record(
            source_sha256=record.source_sha256,
            file_size_bytes=record.file_size_bytes,
            policy=policy,
            decision=decision,
            flags=(
                ScreeningFlag(
                    code=code,
                    stage=ScreeningStage.G0_FILE,
                    severity=(
                        FlagSeverity.HIGH
                        if decision == AdmissionDecision.REJECTED
                        else FlagSeverity.CRITICAL
                    ),
                    detection_view=DetectionView.RAW,
                    matched_value=matched_value,
                ),
            ),
            justification=code,
            page_count=inspection.page_count,
            scanner=scanner,
        )
        raise FileGateError(message, failed)

    if inspection.inspection_status != "OK":
        block(
            "PDF_STRUCTURE_INSPECTION_FAILED",
            "Isolated PDF structure inspection failed",
            AdmissionDecision.QUARANTINED,
            inspection.error_code,
        )
    assert inspection.page_count is not None
    assert inspection.object_count is not None
    if inspection.encrypted or inspection.page_count < 1:
        block(
            "PDF_STRUCTURE_INVALID_OR_ENCRYPTED",
            "PDF is encrypted or has no pages",
            AdmissionDecision.QUARANTINED,
        )
    if inspection.page_count > policy.max_pdf_pages:
        block(
            "PDF_PAGE_LIMIT_EXCEEDED",
            "PDF exceeds the configured page-count limit",
            AdmissionDecision.REJECTED,
            str(inspection.page_count),
        )
    if inspection.object_count > policy.max_pdf_objects:
        block(
            "PDF_OBJECT_LIMIT_EXCEEDED",
            "PDF exceeds the configured object-count limit",
            AdmissionDecision.REJECTED,
            str(inspection.object_count),
        )
    if inspection.active_content_xrefs:
        block(
            "PDF_ACTIVE_CONTENT_OR_EMBEDDED_FILES",
            "PDF contains active content or embedded files",
            AdmissionDecision.QUARANTINED,
            ",".join(str(xref) for xref in inspection.active_content_xrefs),
        )
    return record.model_copy(
        update={
            "page_count": inspection.page_count,
            "justification_codes": ("G0_CHECKS_PASSED",),
        }
    )
