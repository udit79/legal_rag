from __future__ import annotations

from typing import Any

import pymupdf

_MAX_ACTION_DEPTH = 32
_MAX_ACTION_COUNT = 4096
_DESTINATION_MODES = {
    "Fit": 2,
    "FitB": 2,
    "FitH": 3,
    "FitV": 3,
    "FitBH": 3,
    "FitBV": 3,
    "FitR": 6,
    "XYZ": 5,
}
_NULLABLE_DESTINATION_PARAMETERS = {"FitH", "FitV", "FitBH", "FitBV", "XYZ"}
_ANNOTATION_SUBTYPES = {
    "3D",
    "Caret",
    "Circle",
    "FileAttachment",
    "FreeText",
    "Highlight",
    "Ink",
    "Line",
    "Link",
    "Movie",
    "Polygon",
    "PolyLine",
    "Popup",
    "PrinterMark",
    "Redact",
    "Screen",
    "Sound",
    "Square",
    "Squiggly",
    "Stamp",
    "StrikeOut",
    "Text",
    "TrapNet",
    "Underline",
    "Watermark",
    "Widget",
}


class _UnsafeAction(Exception):
    def __init__(self, xref: int) -> None:
        self.xref = xref


def inspect_pdf_active_content(document: pymupdf.Document) -> tuple[int, ...]:
    """Return object references containing unsafe actions or embedded files."""
    # PyMuPDF does not publish typing stubs for the low-level object API.
    mupdf: Any = pymupdf.mupdf  # pyright: ignore[reportPrivateImportUsage]
    pdf: Any = pymupdf._as_pdf_document(document)  # pyright: ignore[reportPrivateImportUsage]
    xref_length = document.xref_length()
    catalog_xref = document.pdf_catalog()
    page_xrefs = {document.page_xref(index) for index in range(document.page_count)}
    unsafe_xrefs: set[int] = set()
    named_destinations: dict[str, dict[str, object]] | None = None

    def is_null(value: Any) -> bool:
        return not value.m_internal or (
            not mupdf.pdf_is_indirect(value) and mupdf.pdf_is_null(value)
        )

    def resolve(
        value: Any,
        owner_xref: int,
        ancestors: frozenset[int] = frozenset(),
    ) -> tuple[Any, int | None, frozenset[int]]:
        if not value.m_internal:
            raise _UnsafeAction(owner_xref)
        if mupdf.pdf_is_indirect(value):
            xref = mupdf.pdf_to_num(value)
            if (
                xref < 1
                or xref >= xref_length
                or xref in ancestors
                or not mupdf.pdf_object_exists(pdf, xref)
            ):
                raise _UnsafeAction(owner_xref)
            xref_entry = mupdf.ll_pdf_get_xref_entry(pdf.m_internal, xref)
            if (
                xref_entry.type not in {"n", "o"}
                or (
                    xref_entry.type == "n"
                    and xref_entry.gen != mupdf.pdf_to_gen(value)
                )
                or (
                    xref_entry.type == "o"
                    and mupdf.pdf_to_gen(value) != 0
                )
            ):
                raise _UnsafeAction(owner_xref)
            resolved = mupdf.pdf_resolve_indirect(value)
            if is_null(resolved):
                raise _UnsafeAction(xref)
            return resolved, xref, ancestors | {xref}
        if mupdf.pdf_is_null(value):
            raise _UnsafeAction(owner_xref)
        return value, None, ancestors

    def dictionary_entries(
        value: Any, owner_xref: int
    ) -> list[tuple[str, Any]]:
        if not mupdf.pdf_is_dict(value):
            raise _UnsafeAction(owner_xref)
        entries = []
        for index in range(mupdf.pdf_dict_len(value)):
            key = mupdf.pdf_dict_get_key(value, index)
            if not mupdf.pdf_is_name(key):
                raise _UnsafeAction(owner_xref)
            entries.append(
                (mupdf.pdf_to_name(key), mupdf.pdf_dict_get_val(value, index))
            )
        return entries

    def is_local_destination(value: Any, owner_xref: int) -> bool:
        nonlocal named_destinations
        destination, _, _ = resolve(value, owner_xref)
        if mupdf.pdf_is_name(destination) or mupdf.pdf_is_string(destination):
            if named_destinations is None:
                named_destinations = document.resolve_names()
            name = (
                mupdf.pdf_to_name(destination)
                if mupdf.pdf_is_name(destination)
                else mupdf.pdf_to_text_string(destination)
            )
            resolved_name = named_destinations.get(name)
            page_number = resolved_name.get("page") if resolved_name else None
            return (
                isinstance(page_number, int)
                and not isinstance(page_number, bool)
                and 0 <= page_number < document.page_count
            )

        if not mupdf.pdf_is_array(destination):
            return False
        length = mupdf.pdf_array_len(destination)
        if length < 2:
            return False

        page_ref = mupdf.pdf_array_get(destination, 0)
        if not mupdf.pdf_is_indirect(page_ref):
            return False
        page_object, page_xref, _ = resolve(page_ref, owner_xref)
        if (
            page_xref not in page_xrefs
            or not mupdf.pdf_is_dict(page_object)
        ):
            return False

        mode_value = mupdf.pdf_array_get(destination, 1)
        if not mupdf.pdf_is_name(mode_value):
            return False
        mode = mupdf.pdf_to_name(mode_value)
        expected_length = _DESTINATION_MODES.get(mode)
        if expected_length is None or length != expected_length:
            return False

        parameters = [
            mupdf.pdf_array_get(destination, index)
            for index in range(2, length)
        ]
        if mode == "FitR":
            return all(mupdf.pdf_is_number(item) for item in parameters)
        if mode in _NULLABLE_DESTINATION_PARAMETERS:
            return all(
                mupdf.pdf_is_null(item) or mupdf.pdf_is_number(item)
                for item in parameters
            )
        return True

    def visit_action(
        value: Any,
        *,
        automatic: bool,
        owner_xref: int,
        depth: int,
        ancestors: frozenset[int] = frozenset(),
        allow_array: bool = False,
        budget: list[int] | None = None,
    ) -> None:
        if budget is None:
            budget = [0]
        budget[0] += 1
        if depth > _MAX_ACTION_DEPTH or budget[0] > _MAX_ACTION_COUNT:
            raise _UnsafeAction(owner_xref)

        action, action_xref, ancestry = resolve(value, owner_xref, ancestors)
        current_xref = action_xref if action_xref is not None else owner_xref
        if mupdf.pdf_is_array(action):
            if not allow_array or mupdf.pdf_array_len(action) == 0:
                raise _UnsafeAction(current_xref)
            for index in range(mupdf.pdf_array_len(action)):
                visit_action(
                    mupdf.pdf_array_get(action, index),
                    automatic=automatic,
                    owner_xref=current_xref,
                    depth=depth + 1,
                    ancestors=ancestry,
                    budget=budget,
                )
            return
        if not mupdf.pdf_is_dict(action):
            raise _UnsafeAction(current_xref)

        subtype, _, _ = resolve(
            mupdf.pdf_dict_gets(action, "S"), current_xref, ancestry
        )
        if not mupdf.pdf_is_name(subtype):
            raise _UnsafeAction(current_xref)
        action_type = mupdf.pdf_to_name(subtype)
        if action_type in {"JavaScript", "Launch"}:
            raise _UnsafeAction(current_xref)
        if automatic and action_type != "GoTo":
            raise _UnsafeAction(current_xref)
        if action_type == "GoTo":
            destination = mupdf.pdf_dict_gets(action, "D")
            if is_null(destination) or not is_local_destination(
                destination, current_xref
            ):
                raise _UnsafeAction(current_xref)

        action_entries = dict(dictionary_entries(action, current_xref))
        nested_aa = action_entries.get("AA")
        if nested_aa is not None:
            if is_null(nested_aa):
                raise _UnsafeAction(current_xref)
            visit_additional_actions(
                nested_aa,
                current_xref,
                ancestry,
                depth=depth,
                budget=budget,
            )

        next_action = action_entries.get("Next")
        if next_action is not None:
            if is_null(next_action):
                raise _UnsafeAction(current_xref)
            visit_action(
                next_action,
                automatic=automatic,
                owner_xref=current_xref,
                depth=depth + 1,
                ancestors=ancestry,
                allow_array=True,
                budget=budget,
            )

    def visit_additional_actions(
        value: Any,
        owner_xref: int,
        ancestors: frozenset[int] = frozenset(),
        *,
        depth: int = 0,
        budget: list[int] | None = None,
    ) -> None:
        if budget is None:
            budget = [0]
        additional_actions, _, ancestry = resolve(value, owner_xref, ancestors)
        for _, action in dictionary_entries(additional_actions, owner_xref):
            visit_action(
                action,
                automatic=True,
                owner_xref=owner_xref,
                depth=depth + 1,
                ancestors=ancestry,
                budget=budget,
            )

    catalog = mupdf.pdf_load_object(pdf, catalog_xref)
    catalog_entries = dict(dictionary_entries(catalog, catalog_xref))
    open_action = catalog_entries.get("OpenAction")
    if open_action is not None:
        if is_null(open_action):
            unsafe_xrefs.add(catalog_xref)
        else:
            try:
                resolved, _, _ = resolve(open_action, catalog_xref)
                if mupdf.pdf_is_dict(resolved):
                    visit_action(
                        open_action,
                        automatic=True,
                        owner_xref=catalog_xref,
                        depth=1,
                    )
                elif not is_local_destination(open_action, catalog_xref):
                    raise _UnsafeAction(catalog_xref)
            except _UnsafeAction as unsafe:
                unsafe_xrefs.add(unsafe.xref)

    for xref in range(1, xref_length):
        if not mupdf.pdf_object_exists(pdf, xref):
            continue
        value = mupdf.pdf_load_object(pdf, xref)
        if not mupdf.pdf_is_dict(value):
            continue

        entries = dict(dictionary_entries(value, xref))
        if "Type" in entries:
            embedded_type, _, _ = resolve(entries["Type"], xref)
            if mupdf.pdf_is_name(embedded_type) and mupdf.pdf_to_name(
                embedded_type
            ) == "EmbeddedFile":
                unsafe_xrefs.add(xref)
        if "EF" in entries and not is_null(entries["EF"]):
            unsafe_xrefs.add(xref)

        for key in ("A", "AA"):
            action_value = entries.get(key)
            if action_value is None:
                continue
            try:
                if key == "AA":
                    if is_null(action_value):
                        raise _UnsafeAction(xref)
                    visit_additional_actions(action_value, xref)
                else:
                    is_annotation = False
                    object_type = entries.get("Type")
                    if object_type is not None and not is_null(object_type):
                        resolved_type, _, _ = resolve(object_type, xref)
                        is_annotation = (
                            mupdf.pdf_is_name(resolved_type)
                            and mupdf.pdf_to_name(resolved_type) == "Annot"
                        )
                    subtype = entries.get("Subtype")
                    if subtype is not None and not is_null(subtype):
                        resolved_subtype, _, _ = resolve(subtype, xref)
                        is_annotation = is_annotation or (
                            mupdf.pdf_is_name(resolved_subtype)
                            and mupdf.pdf_to_name(resolved_subtype)
                            in _ANNOTATION_SUBTYPES
                        )
                    is_outline_item = "Title" in entries and "Parent" in entries
                    if is_null(action_value):
                        if is_annotation or is_outline_item:
                            raise _UnsafeAction(xref)
                        continue
                    action, _, _ = resolve(action_value, xref)
                    if (
                        not is_annotation
                        and not is_outline_item
                        and (
                            not mupdf.pdf_is_dict(action)
                            or "S"
                            not in {
                                name
                                for name, _ in dictionary_entries(action, xref)
                            }
                        )
                    ):
                        continue
                    visit_action(
                        action_value,
                        automatic=False,
                        owner_xref=xref,
                        depth=1,
                    )
            except _UnsafeAction as unsafe:
                unsafe_xrefs.add(unsafe.xref)

    if document.embfile_count() > 0:
        unsafe_xrefs.add(-1)
    return tuple(sorted(unsafe_xrefs))
