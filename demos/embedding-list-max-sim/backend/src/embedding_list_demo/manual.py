"""Build and verify the fixed NASA handbook visual retrieval dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

DATASET_ID = "nasa-systems-engineering-handbook-rev2"
DATASET_TITLE = "NASA Systems Engineering Handbook"
DATASET_REVISION = "NASA/SP-2016-6105 Rev 2"
DOCUMENT_IDENTIFIER = "NASA/SP-2016-6105 Rev 2"
NTRS_ID = 20170001761
NTRS_RECORD_URL = "https://ntrs.nasa.gov/citations/20170001761"
OFFICIAL_PDF_URL = "https://ntrs.nasa.gov/api/citations/20170001761/downloads/20170001761.pdf"
DISTRIBUTION = "PUBLIC"
RIGHTS_DETERMINATION = "PUBLIC_USE_PERMITTED"
CONTAINS_THIRD_PARTY_MATERIAL = False
ATTRIBUTION = (
    "NASA Systems Engineering Handbook, Rev 2 (NASA/SP-2016-6105 Rev 2), "
    "Steven R. Hirshorn, Linda D. Voss, and Linda K. Bromley. "
    "NASA Technical Reports Server record 20170001761."
)
RENDER_STATEMENT = "Page images are unmodified 144-DPI renders from the official PDF."
ENDORSEMENT_STATEMENT = "NASA does not endorse Milvus or this demo."
PDF_FILENAME = "nasa-systems-engineering-handbook-rev2.pdf"
MANIFEST_FILENAME = "dataset-manifest.json"
QUERIES_FILENAME = "queries.json"
HASH_MANIFEST_FILENAME = "SHA256SUMS"
RENDER_DPI = 144
PDF_PAGE_INDEX_BASE = 1
RENDERER = "PyMuPDF"
RENDERER_VERSION = "1.28.2"
GENERATOR = "Pinned official PDF + PyMuPDF RGB page render"
GENERATOR_VERSION = "2.0.0"
SOURCE_PDF_SHA256 = "3153ae2e53e29452d5997efafe280a5f05cd21b43a047e988a17e1dd5207a38e"
SOURCE_PDF_BYTES = 4_122_125
SOURCE_PDF_PAGE_COUNT = 356
PAGE_COUNT = 40
TRACKED_MANIFEST_SHA256 = "5a5c529f4de7181384a6a65bb89caf36ec4a25c94a9787f625e194da842cd2e1"
TRACKED_HASH_MANIFEST_SHA256 = "440ece4df464a7a671789162c6c02c5fd97bfe6c12e24a47dc0663af53052516"


class ManualContractError(RuntimeError):
    """Raised when dataset evidence differs from the fixed contract."""


@dataclass(frozen=True)
class AuthorRecord:
    name: str
    affiliation: str

    def public_dict(self) -> dict[str, str]:
        return asdict(self)


AUTHORS = (
    AuthorRecord("Steven R. Hirshorn", "NASA Headquarters, Washington, DC"),
    AuthorRecord(
        "Linda D. Voss",
        "ASRC Research and Technology Solutions, LLC, Greenbelt, Maryland",
    ),
    AuthorRecord(
        "Linda K. Bromley",
        "ASRC Research and Technology Solutions, LLC, Greenbelt, Maryland",
    ),
)


@dataclass(frozen=True)
class PageSpec:
    pdf_page_index: int
    printed_page: int
    section: str
    title: str
    visual_type: str

    @property
    def page_id(self) -> str:
        return f"nasa-seh-printed-{self.printed_page:03d}"

    @property
    def image_filename(self) -> str:
        return f"pages/pdf-{self.pdf_page_index:03d}-printed-{self.printed_page:03d}.png"

    def source_dict(self) -> dict[str, object]:
        return {
            "page_id": self.page_id,
            "pdf_page_index": self.pdf_page_index,
            "printed_page": self.printed_page,
            "section": self.section,
            "title": self.title,
            "visual_type": self.visual_type,
        }


PAGE_SPECS = (
    PageSpec(
        64,
        51,
        "4.0 System Design Processes",
        "Interrelationships among the System Design Processes",
        "process relationship diagram",
    ),
    PageSpec(
        66,
        53,
        "4.1 Stakeholder Expectations Definition",
        "Stakeholder Expectations Definition Process",
        "process flow diagram",
    ),
    PageSpec(
        69,
        56,
        "4.1 Stakeholder Expectations Definition",
        "Information Flow for Stakeholder Expectations",
        "information flow diagram",
    ),
    PageSpec(
        73,
        60,
        "4.1 Stakeholder Expectations Definition",
        "Example of a Lunar Sortie Design Reference Mission",
        "mission scenario diagram",
    ),
    PageSpec(
        77,
        64,
        "4.2 Technical Requirements Definition",
        "Technical Requirements Definition Process",
        "process flow diagram",
    ),
    PageSpec(
        79,
        66,
        "4.2 Technical Requirements Definition",
        "Flow, Type, and Ownership of Requirements",
        "requirements classification diagram",
    ),
    PageSpec(
        81,
        68,
        "4.2 Technical Requirements Definition",
        "The Flowdown of Requirements",
        "requirements hierarchy diagram",
    ),
    PageSpec(
        86, 73, "4.3 Logical Decomposition", "Logical Decomposition Process", "process flow diagram"
    ),
    PageSpec(
        91,
        78,
        "4.4 Design Solution Definition",
        "Design Solution Definition Process",
        "process flow diagram",
    ),
    PageSpec(
        92,
        79,
        "4.4 Design Solution Definition",
        "The Doctrine of Successive Refinement",
        "iterative design diagram",
    ),
    PageSpec(
        96,
        83,
        "4.4 Design Solution Definition",
        "A Quantitative Objective Function",
        "cost-effectiveness chart",
    ),
    PageSpec(
        102, 89, "5.0 Product Realization", "Product Realization", "process relationship diagram"
    ),
    PageSpec(
        104,
        91,
        "5.1 Product Implementation",
        "Product Implementation Process",
        "process flow diagram",
    ),
    PageSpec(
        112, 99, "5.2 Product Integration", "Product Integration Process", "process flow diagram"
    ),
    PageSpec(
        116,
        103,
        "5.3 Product Verification",
        "Differences between Verification and Validation Testing",
        "comparison guidance box",
    ),
    PageSpec(
        118, 105, "5.3 Product Verification", "Product Verification Process", "process flow diagram"
    ),
    PageSpec(
        125,
        112,
        "5.3 Product Verification",
        "End-to-End Data Flow for a Scientific Satellite Mission",
        "system data-flow diagram",
    ),
    PageSpec(
        130, 117, "5.4 Product Validation", "Product Validation Process", "process flow diagram"
    ),
    PageSpec(
        138, 125, "5.5 Product Transition", "Product Transition Process", "process flow diagram"
    ),
    PageSpec(
        146, 133, "6.1 Technical Planning", "Technical Planning Process", "process flow diagram"
    ),
    PageSpec(
        164,
        151,
        "6.2 Requirements Management",
        "Requirements Management Process",
        "process flow diagram",
    ),
    PageSpec(
        170, 157, "6.3 Interface Management", "Interface Management Process", "process flow diagram"
    ),
    PageSpec(
        174,
        161,
        "6.4 Technical Risk Management",
        "Risk Scenario Development and Risk Triplets",
        "paired risk diagrams",
    ),
    PageSpec(
        176, 163, "6.4 Technical Risk Management", "Risk Management Process", "process flow diagram"
    ),
    PageSpec(
        178,
        165,
        "6.4 Technical Risk Management",
        "Risk-Informed Decision Making and Continuous Risk Management",
        "risk interaction diagram",
    ),
    PageSpec(
        181,
        168,
        "6.5 Configuration Management",
        "Configuration Management Process",
        "process flow diagram",
    ),
    PageSpec(
        182,
        169,
        "6.5 Configuration Management",
        "Five Elements of Configuration Management",
        "management elements diagram",
    ),
    PageSpec(
        184,
        171,
        "6.5 Configuration Management",
        "Evolution of Technical Baseline",
        "baseline evolution diagram",
    ),
    PageSpec(
        185,
        172,
        "6.5 Configuration Management",
        "Typical Change Control Process",
        "change-control flow diagram",
    ),
    PageSpec(
        189,
        176,
        "6.6 Technical Data Management",
        "Technical Data Management Process",
        "process flow diagram",
    ),
    PageSpec(
        197, 184, "6.7 Technical Assessment", "Technical Assessment Process", "process flow diagram"
    ),
    PageSpec(
        198,
        185,
        "6.7 Technical Assessment",
        "Planning and Status Reporting Feedback Loop",
        "feedback-loop diagram",
    ),
    PageSpec(
        206, 193, "6.8 Decision Analysis", "Decision Analysis Process", "process flow diagram"
    ),
    PageSpec(
        208,
        195,
        "6.8 Decision Analysis",
        "Risk Analysis of Decision Alternatives",
        "decision and risk flow diagram",
    ),
    PageSpec(
        249,
        236,
        "Appendix C",
        "How to Write a Good Requirement - Checklist",
        "requirements checklist",
    ),
    PageSpec(253, 240, "Appendix D", "Requirements Verification Matrix", "verification matrix"),
    PageSpec(256, 243, "Appendix E", "Validation Requirements Matrix", "validation matrix"),
    PageSpec(
        261, 248, "Appendix G", "Product Breakdown Structure Example", "product breakdown diagram"
    ),
    PageSpec(269, 256, "Appendix H", "Integration Plan Outline", "integration checklist"),
    PageSpec(272, 259, "Appendix I", "Verification and Validation Plan Outline", "plan outline"),
)


@dataclass(frozen=True)
class PageRecord:
    page_id: str
    pdf_page_index: int
    printed_page: int
    title: str
    section: str
    visual_type: str
    image_filename: str
    image_sha256: str
    width: int
    height: int

    def public_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HardNegative:
    page_id: str
    confusion_reason: str

    def source_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class QueryGroundTruth:
    query_id: str
    text: str
    target_page_ids: tuple[str, ...]
    target_rationale: str
    hard_negatives: tuple[HardNegative, ...]

    def public_dict(self) -> dict[str, str]:
        """Return only the natural query preset; evaluation mappings stay internal."""

        return {"query_id": self.query_id, "text": self.text}

    def source_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "text": self.text,
            "target_page_ids": list(self.target_page_ids),
            "target_rationale": self.target_rationale,
            "hard_negatives": [item.source_dict() for item in self.hard_negatives],
        }


def _page_id(printed_page: int) -> str:
    return f"nasa-seh-printed-{printed_page:03d}"


QUERY_SPECS = (
    QueryGroundTruth(
        "Q01",
        "How should systems engineers identify stakeholders, elicit their needs, validate "
        "expectations against operational scenarios, and baseline the result before defining "
        "technical requirements?",
        (_page_id(53),),
        "The page starts with customer and stakeholder expectations and follows identification, "
        "elicitation, analysis, validation, and baselining before downstream technical work.",
        (
            HardNegative(
                _page_id(64),
                "This related process consumes baselined stakeholder expectations but converts "
                "them into technical requirements.",
            ),
        ),
    ),
    QueryGroundTruth(
        "Q02",
        "How are system-level requirements flowed down into allocated and derived subsystem "
        "requirements while maintaining traceability to their parent requirements?",
        (_page_id(68),),
        "The hierarchy depicts system requirements transformed into subsystem requirements and "
        "then allocated and derived requirements.",
        (
            HardNegative(
                _page_id(73),
                "Logical decomposition allocates derived requirements but is a general process "
                "flow rather than the requested hierarchy.",
            ),
        ),
    ),
    QueryGroundTruth(
        "Q03",
        "What evidence should engineers collect to show both that a built product conforms to "
        "its specified requirements and that typical users can succeed with it in a realistic "
        "operational setting?",
        (_page_id(103),),
        "The page contrasts controlled requirement-based evidence with realistic-use evidence "
        "tied to the operational concept.",
        (
            HardNegative(
                _page_id(117),
                "Product validation covers realistic use but does not provide the same "
                "side-by-side distinction.",
            ),
        ),
    ),
    QueryGroundTruth(
        "Q04",
        "How should a team structure a technical risk so that initiating events, uncertain "
        "likelihood, and uncertain consequences can be analyzed together?",
        (_page_id(161),),
        "The diagrams connect initiating events and scenario models with likelihood and "
        "consequence uncertainty.",
        (
            HardNegative(
                _page_id(163),
                "The overall risk workflow does not define the internal scenario triplet "
                "requested.",
            ),
        ),
    ),
    QueryGroundTruth(
        "Q05",
        "After someone proposes a baseline change, what sequence of review, board decision, "
        "release, implementation, and status accounting should control it?",
        (_page_id(172),),
        "The diagram follows a change through originator, configuration-management, reviewer, "
        "board, and actionee responsibilities.",
        (
            HardNegative(
                _page_id(168),
                "The overall configuration-management workflow lacks the detailed actors and "
                "decision sequence.",
            ),
        ),
    ),
    QueryGroundTruth(
        "Q06",
        "When competing engineering alternatives rank closely, how can qualitative and "
        "quantitative analysis be combined to decide whether further uncertainty reduction is "
        "worthwhile?",
        (_page_id(195),),
        "The diagram combines qualitative, quantitative, and risk analysis, tests ranking "
        "robustness, and applies a net-benefit gate.",
        (
            HardNegative(
                _page_id(193),
                "The high-level decision workflow lacks the robustness and uncertainty-reduction "
                "loop.",
            ),
        ),
    ),
    QueryGroundTruth(
        "Q07",
        "Which diagram shows a spacecraft traveling from Earth orbit down to a lunar landing "
        "site along a drawn flight trajectory?",
        (_page_id(60),),
        "The pictorial mission scenario draws the flight path from Earth orbit to the lunar "
        "surface with spacecraft figures rather than descriptive text.",
        (
            HardNegative(
                _page_id(112),
                "This also depicts a space system but as an abstract data-flow diagram rather "
                "than a drawn flight trajectory.",
            ),
        ),
    ),
    QueryGroundTruth(
        "Q08",
        "Which chart plots a curve that rises as cost increases against effectiveness, with "
        "labeled axes?",
        (_page_id(83),),
        "The cost-effectiveness chart draws a labeled x-y curve that rises with cost versus "
        "effectiveness.",
        (
            HardNegative(
                _page_id(79),
                "This is a graphic loop of successive refinement rather than a labeled x-y "
                "cost-effectiveness curve.",
            ),
        ),
    ),
)


@dataclass(frozen=True)
class ManualManifest:
    schema_version: int
    dataset_id: str
    title: str
    revision: str
    document_identifier: str
    ntrs_id: int
    ntrs_record_url: str
    official_pdf_url: str
    distribution: str
    rights_determination: str
    contains_third_party_material: bool
    attribution: str
    authors: tuple[AuthorRecord, ...]
    language: str
    render_statement: str
    endorsement_statement: str
    generator: str
    generator_version: str
    renderer: str
    renderer_version: str
    source_spec_sha256: str
    source_pdf_page_count: int
    source_pdf_bytes: int
    pdf_page_index_base: int
    page_count: int
    pdf_filename: str
    pdf_sha256: str
    render_dpi: int
    queries_filename: str
    queries_sha256: str
    pages: tuple[PageRecord, ...]
    manifest_sha256: str = ""

    def source_dict(self) -> dict[str, object]:
        value = asdict(self)
        value.pop("manifest_sha256")
        value["authors"] = [author.public_dict() for author in self.authors]
        value["pages"] = [page.public_dict() for page in self.pages]
        return value

    def public_dict(self) -> dict[str, object]:
        return {**self.source_dict(), "manifest_sha256": self.manifest_sha256}

    @classmethod
    def from_dict(cls, value: dict[str, object], *, manifest_sha256: str) -> ManualManifest:
        authors = tuple(AuthorRecord(**item) for item in value["authors"])  # type: ignore[arg-type]
        pages = tuple(PageRecord(**item) for item in value["pages"])  # type: ignore[arg-type]
        return cls(
            **{key: item for key, item in value.items() if key not in {"authors", "pages"}},
            authors=authors,
            pages=pages,
            manifest_sha256=manifest_sha256,
        )  # type: ignore[arg-type]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_spec_sha256() -> str:
    payload = {
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "source_pdf_sha256": SOURCE_PDF_SHA256,
        "render_dpi": RENDER_DPI,
        "pdf_page_index_base": PDF_PAGE_INDEX_BASE,
        "renderer": RENDERER,
        "renderer_version": RENDERER_VERSION,
        "pages": [item.source_dict() for item in PAGE_SPECS],
        "queries": [item.source_dict() for item in QUERY_SPECS],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _queries_payload() -> dict[str, object]:
    return {
        "schema_version": 2,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "queries": [query.source_dict() for query in QUERY_SPECS],
    }


def _validate_source_pdf(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise ManualContractError(f"Source PDF must be a regular file: {path}")
    if path.stat().st_size != SOURCE_PDF_BYTES or sha256_file(path) != SOURCE_PDF_SHA256:
        raise ManualContractError("Source PDF bytes or SHA-256 differ from the fixed NTRS object")
    with pymupdf.open(path) as document:
        if document.page_count != SOURCE_PDF_PAGE_COUNT:
            raise ManualContractError("Source PDF page count differs from the fixed NTRS object")


def _write_hash_manifest(root: Path) -> None:
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != HASH_MANIFEST_FILENAME
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in paths]
    (root / HASH_MANIFEST_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verify_hash_manifest(root: Path) -> None:
    manifest_path = root / HASH_MANIFEST_FILENAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ManualContractError("Dataset SHA256SUMS is missing or unsafe")
    if sha256_file(manifest_path) != TRACKED_HASH_MANIFEST_SHA256:
        raise ManualContractError("Dataset SHA256SUMS differs from the fixed tracked dataset")
    declared: set[str] = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if (
            separator != "  "
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or relative in declared
        ):
            raise ManualContractError("Dataset SHA256SUMS has an invalid entry")
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
            raise ManualContractError(f"Unsafe dataset hash path: {relative}")
        if sha256_file(path) != digest:
            raise ManualContractError(f"Dataset SHA-256 verification failed: {relative}")
        declared.add(relative)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != HASH_MANIFEST_FILENAME
    }
    if declared != actual:
        raise ManualContractError("Dataset SHA256SUMS file set differs from the dataset")


def generate_manual(destination: Path, source_pdf: Path) -> ManualManifest:
    """Build the fixed dataset without overwriting any existing path."""

    if pymupdf.__version__ != RENDERER_VERSION:
        raise ManualContractError(
            f"Dataset rebuild requires PyMuPDF {RENDERER_VERSION}, got {pymupdf.__version__}"
        )
    destination = destination.resolve()
    source_pdf = source_pdf.resolve(strict=True)
    _validate_source_pdf(source_pdf)
    if destination.exists() or destination.is_symlink():
        raise ManualContractError(f"Refusing to overwrite dataset directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".nasa-seh-rev2-", dir=destination.parent))
    try:
        pdf_path = temporary / PDF_FILENAME
        shutil.copyfile(source_pdf, pdf_path)
        pages_dir = temporary / "pages"
        pages_dir.mkdir()
        records: list[PageRecord] = []
        with pymupdf.open(pdf_path) as document:
            for spec in PAGE_SPECS:
                page = document[spec.pdf_page_index - 1]
                pixmap = page.get_pixmap(dpi=RENDER_DPI, colorspace=pymupdf.csRGB, alpha=False)
                image_path = temporary / spec.image_filename
                pixmap.save(image_path)
                records.append(
                    PageRecord(
                        **spec.source_dict(),
                        image_filename=spec.image_filename,
                        image_sha256=sha256_file(image_path),
                        width=pixmap.width,
                        height=pixmap.height,
                    )
                )
        queries_path = temporary / QUERIES_FILENAME
        queries_path.write_text(
            json.dumps(_queries_payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = ManualManifest(
            schema_version=3,
            dataset_id=DATASET_ID,
            title=DATASET_TITLE,
            revision=DATASET_REVISION,
            document_identifier=DOCUMENT_IDENTIFIER,
            ntrs_id=NTRS_ID,
            ntrs_record_url=NTRS_RECORD_URL,
            official_pdf_url=OFFICIAL_PDF_URL,
            distribution=DISTRIBUTION,
            rights_determination=RIGHTS_DETERMINATION,
            contains_third_party_material=CONTAINS_THIRD_PARTY_MATERIAL,
            attribution=ATTRIBUTION,
            authors=AUTHORS,
            language="en",
            render_statement=RENDER_STATEMENT,
            endorsement_statement=ENDORSEMENT_STATEMENT,
            generator=GENERATOR,
            generator_version=GENERATOR_VERSION,
            renderer=RENDERER,
            renderer_version=RENDERER_VERSION,
            source_spec_sha256=source_spec_sha256(),
            source_pdf_page_count=SOURCE_PDF_PAGE_COUNT,
            source_pdf_bytes=SOURCE_PDF_BYTES,
            pdf_page_index_base=PDF_PAGE_INDEX_BASE,
            page_count=len(records),
            pdf_filename=PDF_FILENAME,
            pdf_sha256=sha256_file(pdf_path),
            render_dpi=RENDER_DPI,
            queries_filename=QUERIES_FILENAME,
            queries_sha256=sha256_file(queries_path),
            pages=tuple(records),
        )
        (temporary / MANIFEST_FILENAME).write_text(
            json.dumps(manifest.source_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_hash_manifest(temporary)
        os.replace(temporary, destination)
        return load_manual(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_queries(root: Path, manifest: ManualManifest) -> tuple[QueryGroundTruth, ...]:
    """Load and verify every natural query plus internal evaluation mappings."""

    path = (root / manifest.queries_filename).resolve(strict=True)
    if not path.is_relative_to(root) or sha256_file(path) != manifest.queries_sha256:
        raise ManualContractError("Query SHA-256 verification failed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value != _queries_payload():
        raise ManualContractError("Query presets or internal evaluation mappings differ")
    page_ids = {page.page_id for page in manifest.pages}
    for query in QUERY_SPECS:
        if not set(query.target_page_ids).issubset(page_ids):
            raise ManualContractError(f"Query target is outside the dataset: {query.query_id}")
        negatives = {item.page_id for item in query.hard_negatives}
        if not negatives.issubset(page_ids) or negatives & set(query.target_page_ids):
            raise ManualContractError(f"Query hard negatives are invalid: {query.query_id}")
        lowered = query.text.casefold()
        for page_id in (*query.target_page_ids, *negatives):
            page = next(item for item in manifest.pages if item.page_id == page_id)
            if str(page.pdf_page_index) in lowered or str(page.printed_page) in lowered:
                raise ManualContractError(f"Query leaks a page number: {query.query_id}")
            if page.title.casefold() in lowered:
                raise ManualContractError(f"Query leaks a full page title: {query.query_id}")
    return QUERY_SPECS


def load_manual(root: Path) -> ManualManifest:
    """Load the tracked dataset only after every declared hash is verified."""

    root = root.resolve(strict=True)
    _verify_hash_manifest(root)
    manifest_path = (root / MANIFEST_FILENAME).resolve(strict=True)
    if not manifest_path.is_relative_to(root):
        raise ManualContractError("Unsafe dataset manifest path")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = ManualManifest.from_dict(value, manifest_sha256=sha256_file(manifest_path))
    fixed = (
        manifest.manifest_sha256 == TRACKED_MANIFEST_SHA256
        and manifest.schema_version == 3
        and manifest.dataset_id == DATASET_ID
        and manifest.title == DATASET_TITLE
        and manifest.revision == DATASET_REVISION
        and manifest.document_identifier == DOCUMENT_IDENTIFIER
        and manifest.ntrs_id == NTRS_ID
        and manifest.ntrs_record_url == NTRS_RECORD_URL
        and manifest.official_pdf_url == OFFICIAL_PDF_URL
        and manifest.distribution == DISTRIBUTION
        and manifest.rights_determination == RIGHTS_DETERMINATION
        and manifest.contains_third_party_material is False
        and manifest.attribution == ATTRIBUTION
        and manifest.authors == AUTHORS
        and manifest.language == "en"
        and manifest.render_statement == RENDER_STATEMENT
        and manifest.endorsement_statement == ENDORSEMENT_STATEMENT
        and manifest.generator == GENERATOR
        and manifest.generator_version == GENERATOR_VERSION
        and manifest.page_count == PAGE_COUNT
        and manifest.source_pdf_page_count == SOURCE_PDF_PAGE_COUNT
        and manifest.source_pdf_bytes == SOURCE_PDF_BYTES
        and manifest.pdf_page_index_base == PDF_PAGE_INDEX_BASE
        and manifest.pdf_filename == PDF_FILENAME
        and manifest.pdf_sha256 == SOURCE_PDF_SHA256
        and manifest.render_dpi == RENDER_DPI
        and manifest.renderer == RENDERER
        and manifest.renderer_version == RENDERER_VERSION
        and manifest.queries_filename == QUERIES_FILENAME
        and manifest.source_spec_sha256 == source_spec_sha256()
    )
    if not fixed:
        raise ManualContractError("Dataset source, rights, renderer, or identity differs")
    pdf_path = (root / manifest.pdf_filename).resolve(strict=True)
    if not pdf_path.is_relative_to(root):
        raise ManualContractError("Unsafe dataset PDF path")
    _validate_source_pdf(pdf_path)
    expected = [spec.source_dict() for spec in PAGE_SPECS]
    actual = [
        {
            "page_id": page.page_id,
            "pdf_page_index": page.pdf_page_index,
            "printed_page": page.printed_page,
            "section": page.section,
            "title": page.title,
            "visual_type": page.visual_type,
        }
        for page in manifest.pages
    ]
    if actual != expected or len({page.page_id for page in manifest.pages}) != PAGE_COUNT:
        raise ManualContractError("Dataset page selection, order, or metadata differs")
    for page in manifest.pages:
        image_path = (root / page.image_filename).resolve(strict=True)
        if not image_path.is_relative_to(root) or sha256_file(image_path) != page.image_sha256:
            raise ManualContractError(f"Dataset page SHA-256 verification failed: {page.page_id}")
        if (page.width, page.height) not in {(1224, 1584), (1584, 1224)}:
            raise ManualContractError(f"Dataset page dimensions differ: {page.page_id}")
    load_queries(root, manifest)
    return manifest


def resolve_page_image(root: Path, manifest: ManualManifest, page_id: str) -> Path:
    """Resolve one allowlisted unmodified page render."""

    record = next((page for page in manifest.pages if page.page_id == page_id), None)
    if record is None:
        raise ManualContractError(f"Unknown NASA handbook page ID: {page_id}")
    resolved_root = root.resolve(strict=True)
    path = (resolved_root / record.image_filename).resolve(strict=True)
    if not path.is_relative_to(resolved_root) or not path.is_file():
        raise ManualContractError(f"Unsafe NASA handbook page path: {page_id}")
    return path


def query_ground_truth(
    queries: Sequence[QueryGroundTruth], query_text: str
) -> QueryGroundTruth | None:
    """Resolve internal evaluation metadata for an exact approved natural query."""

    normalized = " ".join(query_text.split()).casefold()
    return next(
        (item for item in queries if " ".join(item.text.split()).casefold() == normalized),
        None,
    )


def compare_dataset_trees(expected: Path, rebuilt: Path) -> None:
    """Require two dataset directories to have identical files and bytes."""

    expected = expected.resolve(strict=True)
    rebuilt = rebuilt.resolve(strict=True)
    expected_files = sorted(
        path.relative_to(expected) for path in expected.rglob("*") if path.is_file()
    )
    rebuilt_files = sorted(
        path.relative_to(rebuilt) for path in rebuilt.rglob("*") if path.is_file()
    )
    if expected_files != rebuilt_files:
        raise ManualContractError("Rebuilt dataset file set differs from the tracked dataset")
    for relative in expected_files:
        if sha256_file(expected / relative) != sha256_file(rebuilt / relative):
            raise ManualContractError(f"Rebuilt dataset bytes differ: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or verify the fixed NASA handbook dataset")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("source_pdf", type=Path)
    build.add_argument("destination", type=Path)
    validate = subparsers.add_parser("validate")
    validate.add_argument("dataset_root", type=Path)
    rebuild = subparsers.add_parser("rebuild-check")
    rebuild.add_argument("dataset_root", type=Path)
    rebuild.add_argument("destination", type=Path)
    args = parser.parse_args()

    if args.command == "build":
        result = generate_manual(args.destination, args.source_pdf)
    elif args.command == "validate":
        result = load_manual(args.dataset_root)
    else:
        tracked = load_manual(args.dataset_root)
        rebuilt = generate_manual(args.destination, args.dataset_root / tracked.pdf_filename)
        compare_dataset_trees(args.dataset_root, args.destination)
        result = rebuilt
    print(
        json.dumps(
            {
                "status": "passed",
                "dataset_id": result.dataset_id,
                "revision": result.revision,
                "page_count": result.page_count,
                "pdf_sha256": result.pdf_sha256,
                "manifest_sha256": result.manifest_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
