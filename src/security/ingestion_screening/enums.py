from enum import Enum
class AdmissionDecision(str, Enum):
    ADMITTED = "ADMITTED"
    ADMITTED_FLAGGED = "ADMITTED_FLAGGED"
    QUARANTINED = "QUARANTINED"
    REJECTED = "REJECTED"

class FlagSeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class ScreeningStage(str, Enum):
    G0_FILE = "G0_FILE"
    G1_INTEGRITY = "G1_INTEGRITY"
    G2_INJECTION = "G2_INJECTION"
    G3_LEARNED = "G3_LEARNED"
    G4_PROVENANCE = "G4_PROVENANCE"
    G5_ADMISSION = "G5_ADMISSION"
    G6_GRAPH_WRITE = "G6_GRAPH_WRITE"

class TrustTier(str, Enum):
    T0_CURATED = "T0"
    T1_ORGANIZATION_VERIFIED = "T1"
    T2_AUTHENTICATED_UPLOAD = "T2"
    T3_UNAUTHENTICATED_EXTERNAL = "T3"

class DetectionView(str, Enum):
    RAW = "RAW"
    NORMALIZED = "NORMALIZED"
    BOUNDARY = "BOUNDARY"
