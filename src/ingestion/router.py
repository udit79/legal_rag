from src.ingestion.contracts import DocumentFeatures,DocumentRoutePlan,PageFeatures,PageRouteDecision,Route
MIN_NATIVE_TEXT_CHARACTERS = 50
def _route_page(page: PageFeatures) -> tuple[Route, str]:
    has_visual_content = page.image_count > 0 or page.drawing_count > 0

    if page.text_char_count == 0 and not has_visual_content:
        return "SKIP_BLANK", "No text, images, or drawings were detected."

    if page.text_char_count >= MIN_NATIVE_TEXT_CHARACTERS:
        if has_visual_content:
            return (
                "STRUCTURED_PARSE",
                "Sufficient native text and visual content were detected.",
            )
        return (
            "FAST_TEXT",
            "Sufficient native text was detected without visual content.",
        )
    if page.image_count > 0:
        return (
            "OCR",
            "Little native text and image content were detected.",
        )
    if page.drawing_count > 0:
        return (
            "LAYOUT_OCR",
            "Little native text and vector drawing content were detected.",
        )
    return (
        "ESCALATE",
        "Short native text was detected without visual content.",
    )
def route_document(
    features: DocumentFeatures, policy_version: str
) -> DocumentRoutePlan:
    if not isinstance(features, DocumentFeatures):
        raise TypeError("features must be a DocumentFeatures instance")
    if not isinstance(policy_version, str):
        raise TypeError("policy_version must be a string")
    if not policy_version.strip():
        raise ValueError("policy_version must not be blank")

    decisions: list[PageRouteDecision] = []
    for page in features.pages:
        route, reason = _route_page(page)
        decisions.append(
            PageRouteDecision(
                page_number=page.page_number,
                route=route,
                reason=reason,
                decision_source="RULES",
                policy_version=policy_version,
            )
        )
    return DocumentRoutePlan(
        page_count=features.page_count,
        decisions=tuple(decisions),
        document_disposition=(
            "REJECT_NO_CONTENT"
            if all(decision.route == "SKIP_BLANK" for decision in decisions)
            else "PROCESS"
        ),
    )