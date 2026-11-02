from pathlib import Path

import pymupdf

from src.ingestion.contracts import DocumentRoutePlan, PageRouteDecision
from src.ingestion.parser_contracts import (
    CanonicalDocument,
    CanonicalPage,
    ParserFailure,
)
from src.ingestion.parsers import FastTextParser, OcrParser, PageParser, StructuredParser


def _failed_page(
    page_number: int,
    parser: str,
    code: str,
    message: str,
) -> CanonicalPage:
    return CanonicalPage(
        page_number=page_number,
        parser=parser,
        status="FAILED",
        failure=ParserFailure(parser=parser, code=code, message=message),
    )


def _document_failure(
    plan: DocumentRoutePlan, code: str, message: str
) -> CanonicalDocument:
    failure = ParserFailure(parser="DISPATCHER", code=code, message=message)
    pages = tuple(
        _failed_page(page_number, "DISPATCHER", code, message)
        for page_number in range(1, plan.page_count + 1)
    )
    return CanonicalDocument(
        page_count=plan.page_count,
        document_disposition=plan.document_disposition,
        pages=pages,
        document_failure=failure,
    )


def _adapter_for_route(
    decision: PageRouteDecision,
    *,
    fast_text_parser: PageParser | None,
    structured_parser: PageParser | None,
    ocr_parser: PageParser | None,
) -> PageParser | None:
    if decision.route == "FAST_TEXT":
        return fast_text_parser if fast_text_parser is not None else FastTextParser()
    if decision.route == "STRUCTURED_PARSE":
        return (
            structured_parser
            if structured_parser is not None
            else StructuredParser()
        )
    if decision.route == "OCR":
        return ocr_parser if ocr_parser is not None else OcrParser()
    if decision.route == "LAYOUT_OCR":
        return (
            structured_parser
            if structured_parser is not None
            else StructuredParser(do_ocr=True)
        )
    return None


def dispatch_document(
    source: Path,
    route_plan: DocumentRoutePlan,
    *,
    fast_text_parser: PageParser | None = None,
    structured_parser: PageParser | None = None,
    ocr_parser: PageParser | None = None,
) -> CanonicalDocument:
    """Parse routed pages independently and retain failures in page order."""
    if not isinstance(route_plan, DocumentRoutePlan):
        raise TypeError("route_plan must be a DocumentRoutePlan instance")

    try:
        with pymupdf.open(source) as document:
            if document.needs_pass:
                return _document_failure(
                    route_plan,
                    "ENCRYPTED_DOCUMENT",
                    "PDF is encrypted and requires a password.",
                )
            if document.page_count != route_plan.page_count:
                return _document_failure(
                    route_plan,
                    "PAGE_COUNT_MISMATCH",
                    f"Route plan has {route_plan.page_count} pages but PDF has "
                    f"{document.page_count}.",
                )
    except Exception as exc:
        return _document_failure(
            route_plan,
            "DOCUMENT_OPEN_FAILED",
            f"Unable to open PDF: {type(exc).__name__}: {exc}",
        )

    pages: list[CanonicalPage] = []
    parser_by_route: dict[str, PageParser | None] = {}
    blank_validator = (
        fast_text_parser if fast_text_parser is not None else FastTextParser()
    )
    for decision in route_plan.decisions:
        if decision.route == "SKIP_BLANK":
            try:
                page = blank_validator.parse_page(source, decision.page_number)
                if page.status == "BLANK":
                    pages.append(
                        CanonicalPage(
                            page_number=decision.page_number,
                            parser="NONE",
                            status="BLANK",
                        )
                    )
                else:
                    warning = ParserFailure(
                        parser="DISPATCHER",
                        code="ROUTE_FEATURE_MISMATCH",
                        message=(
                            "The route plan marked this page blank, but PyMuPDF "
                            "detected content; native extraction was used."
                        ),
                    )
                    pages.append(
                        page.model_copy(update={"warnings": (warning,)})
                    )
            except Exception as exc:
                pages.append(
                    _failed_page(
                        decision.page_number,
                        "FAST_TEXT",
                        type(exc).__name__,
                        str(exc) or type(exc).__name__,
                    )
                )
            continue

        if decision.route not in parser_by_route:
            parser_by_route[decision.route] = _adapter_for_route(
                decision,
                fast_text_parser=fast_text_parser,
                structured_parser=structured_parser,
                ocr_parser=ocr_parser,
            )
        parser = parser_by_route[decision.route]
        if parser is None:
            pages.append(
                _failed_page(
                    decision.page_number,
                    "DISPATCHER",
                    "ROUTE_NOT_PARSABLE",
                    f"Route {decision.route} requires manual handling.",
                )
            )
            continue

        try:
            parsed_page = parser.parse_page(source, decision.page_number)
            if parsed_page.page_number != decision.page_number:
                raise ValueError(
                    f"Parser returned page {parsed_page.page_number} for "
                    f"requested page {decision.page_number}"
                )
            pages.append(parsed_page)
        except Exception as exc:
            parser_name = getattr(parser, "parser_name", type(parser).__name__)
            pages.append(
                _failed_page(
                    decision.page_number,
                    parser_name,
                    type(exc).__name__,
                    str(exc) or type(exc).__name__,
                )
            )

    document_disposition = route_plan.document_disposition
    if document_disposition == "REJECT_NO_CONTENT" and any(
        page.status in {"PARSED", "VECTOR_ONLY"} for page in pages
    ):
        document_disposition = "PROCESS"

    return CanonicalDocument(
        page_count=route_plan.page_count,
        document_disposition=document_disposition,
        pages=tuple(pages),
    )
