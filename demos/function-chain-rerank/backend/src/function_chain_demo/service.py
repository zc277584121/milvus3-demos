"""Milvus provisioning and search paths for the Function Chain demo."""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from pymilvus import DataType, FunctionChain, FunctionChainStage, MilvusClient
from pymilvus.function_chain import col, fn

from function_chain_demo.catalog import (
    DATASET,
    DATASET_IDENTITY,
    EMBEDDING_DIMENSION,
    FEATURE_NAMES,
    PRODUCTS,
)
from function_chain_demo.chain_features import (
    FRESHNESS_DECAY_FUNCTION,
    FRESHNESS_DECAY_OFFSET,
    FRESHNESS_DECAY_ORIGIN,
    FRESHNESS_DECAY_SCALE,
    FRESHNESS_DECAY_VALUE,
    POPULARITY_CLICKS_COEFF,
    POPULARITY_SALES_COEFF,
    PRICE_DECAY_FUNCTION,
    PRICE_DECAY_OFFSET,
    PRICE_DECAY_ORIGIN,
    PRICE_DECAY_SCALE,
    PRICE_DECAY_VALUE,
)
from function_chain_demo.chain_features import (
    freshness as chain_freshness,
)
from function_chain_demo.chain_features import (
    popularity as chain_popularity,
)
from function_chain_demo.chain_features import (
    price_affinity as chain_price_affinity,
)
from function_chain_demo.config import DemoSettings
from function_chain_demo.dataset import SEMANTIC_SCORE_DECIMALS, SIMULATED_FIELD, Product
from function_chain_demo.embedding_onnx import encode_query
from function_chain_demo.modeling import train_model
from function_chain_demo.object_store import MinioModelObjectStore, ModelObjectStore

OUTPUT_FIELDS = ["item_id"]
SEARCH_LIMIT = 20

# A human-readable model of the collection schema, focused on the columns the
# five-step Function Chain actually reads and writes. This is returned verbatim
# in the search response so the UI can render the data model without guessing.
COLLECTION_SCHEMA_FIELDS: Final = (
    {"name": "id", "type": "INT64", "role": "primary_key", "note": "stable product id"},
    {
        "name": "item_id",
        "type": "VARCHAR",
        "role": "identity",
        "note": "synthetic catalog item id",
    },
    {
        "name": "embedding",
        "type": f"FLOAT_VECTOR({EMBEDDING_DIMENSION})",
        "role": "vector",
        "note": "pure BGE-M3 dense vector, L2-normalized",
    },
    {
        "name": "clicks_30d",
        "type": "INT64",
        "role": "chain_input",
        "note": "simulated 30-day clicks → popularity",
    },
    {
        "name": "sales_30d",
        "type": "INT64",
        "role": "chain_input",
        "note": "simulated 30-day sales → popularity",
    },
    {
        "name": "display_price_usd",
        "type": "FLOAT",
        "role": "chain_input",
        "note": "simulated display price → price affinity",
    },
    {
        "name": "release_epoch",
        "type": "INT64",
        "role": "chain_input",
        "note": "simulated release epoch → freshness",
    },
    {
        "name": "rating",
        "type": "FLOAT",
        "role": "model_feature",
        "note": "derived rating [0,1] → XGBoost feature",
    },
    {
        "name": "title",
        "type": "VARCHAR",
        "role": "text",
        "note": "authored synthetic title",
    },
    {
        "name": "description",
        "type": "VARCHAR",
        "role": "text",
        "note": "authored synthetic description + bullet points",
    },
)


class MilvusClientLike(Protocol):
    def close(self) -> None: ...

    def get_server_version(self) -> str: ...

    def list_file_resources(self) -> list[Any]: ...

    def remove_file_resource(self, name: str) -> Any: ...

    def add_file_resource(self, name: str, path: str) -> Any: ...

    def has_collection(self, collection_name: str) -> bool: ...

    def drop_collection(self, collection_name: str) -> Any: ...

    def create_schema(self, **kwargs: Any) -> Any: ...

    def prepare_index_params(self) -> Any: ...

    def create_collection(self, **kwargs: Any) -> Any: ...

    def insert(self, **kwargs: Any) -> Any: ...

    def flush(self, **kwargs: Any) -> Any: ...

    def load_collection(self, **kwargs: Any) -> Any: ...

    def search(self, **kwargs: Any) -> list[list[dict[str, Any]]]: ...


ClientFactory = Callable[[], MilvusClientLike]


def file_resource_names(resources: Iterable[object]) -> set[str]:
    """Return names from PyMilvus FileResourceInfo values or string test fakes."""
    names: set[str] = set()
    for resource in resources:
        name = resource if isinstance(resource, str) else getattr(resource, "name", None)
        if not isinstance(name, str) or not name:
            raise TypeError(
                "list_file_resources() returned an item without a non-empty string name"
            )
        names.add(name)
    return names


def default_client_factory(settings: DemoSettings) -> ClientFactory:
    def create_client() -> MilvusClient:
        return MilvusClient(
            uri=settings.milvus_uri,
            token=settings.token_value(),
            timeout=settings.milvus_timeout_seconds,
        )

    return create_client


@dataclass(frozen=True)
class RuntimeManifest:
    collection_name: str
    model_resource_name: str
    model_object_name: str
    milvus_version: str
    product_count: int
    dataset: dict[str, object]
    model: dict[str, object]


@dataclass(frozen=True)
class RuntimeState:
    collection_exists: bool
    file_resource_exists: bool
    model_object_exists: bool


@dataclass(frozen=True)
class RankedProduct:
    id: int
    rank: int
    score: float
    item_id: str
    title: str | None
    product_type: str
    brand: str | None
    color: str | None
    material: str | None
    style: str | None
    node_name: str | None
    description: str | None
    bullet_points: tuple[str, ...]
    main_image_id: str
    selected_image_id: str
    image_role: str
    source_object_path: str
    source_url: str
    display_price_usd: float
    rating_value: float
    clicks_30d: int
    sales_30d: int
    inventory_units: int
    inventory_capacity: int
    release_date: str
    release_epoch: int
    rating: float
    inventory: float
    return_rate: float
    freshness: float
    image_path: str
    image_mime: str
    image_width: int
    image_height: int
    image_sha256: str
    operational_signal_provenance: str
    field_provenance: dict[str, str]

    @classmethod
    def from_product(cls, product: Product, rank: int, score: float) -> RankedProduct:
        return cls(
            id=product.id,
            rank=rank,
            score=score,
            item_id=product.item_id,
            title=product.title,
            product_type=product.product_type,
            brand=product.brand,
            color=product.color,
            material=product.material,
            style=product.style,
            node_name=product.node_name,
            description=product.description,
            bullet_points=product.bullet_points,
            main_image_id=product.main_image_id,
            selected_image_id=product.selected_image_id,
            image_role=product.image_role,
            source_object_path=product.source_object_path,
            source_url=product.source_url,
            display_price_usd=product.display_price_usd,
            rating_value=product.rating_value,
            clicks_30d=product.clicks_30d,
            sales_30d=product.sales_30d,
            inventory_units=product.inventory_units,
            inventory_capacity=product.inventory_capacity,
            release_date=product.release_date,
            release_epoch=product.release_epoch,
            rating=product.rating,
            inventory=product.inventory,
            return_rate=product.return_rate,
            freshness=product.freshness,
            image_path=product.image_path,
            image_mime=product.image_mime,
            image_width=product.image_width,
            image_height=product.image_height,
            image_sha256=product.image_sha256,
            operational_signal_provenance=SIMULATED_FIELD,
            field_provenance=product.field_provenance,
        )


@dataclass(frozen=True)
class SearchComparison:
    query_id: str | None
    query_text: str
    execution_path: str
    model_resource_name: str
    function_chain: dict[str, object]
    dataset: dict[str, object]
    data_model: dict[str, object]
    vector_order: tuple[RankedProduct, ...]
    business_order: tuple[RankedProduct, ...]

    def public_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "execution_path": self.execution_path,
            "model_resource_name": self.model_resource_name,
            "function_chain": self.function_chain,
            "dataset": self.dataset,
            "data_model": self.data_model,
            "vector_order": [asdict(product) for product in self.vector_order],
            "business_order": [asdict(product) for product in self.business_order],
        }


class DemoProvisioner:
    def __init__(
        self,
        settings: DemoSettings,
        *,
        client_factory: ClientFactory | None = None,
        object_store: ModelObjectStore | None = None,
    ) -> None:
        self.settings = settings
        self._client_factory = client_factory or default_client_factory(settings)
        self._object_store = object_store or MinioModelObjectStore(settings)

    def prepare(self) -> RuntimeManifest:
        client = self._client_factory()
        uploaded = False
        resource_registered = False
        try:
            server_version = client.get_server_version()
            if server_version != self.settings.milvus_expected_version:
                raise RuntimeError(
                    f"Expected Milvus {self.settings.milvus_expected_version}, got {server_version}"
                )
            with tempfile.TemporaryDirectory(prefix="function-chain-rerank-") as temp_dir:
                artifact = train_model(Path(temp_dir) / "xgb-reranker.ubj")
                self._object_store.upload(artifact.path)
                uploaded = True
                self._replace_file_resource(client)
                resource_registered = True
                self._replace_collection(client)
            return RuntimeManifest(
                collection_name=self.settings.collection_name,
                model_resource_name=self.settings.model_resource_name,
                model_object_name=self.settings.model_object_name,
                milvus_version=server_version,
                product_count=len(PRODUCTS),
                dataset=DATASET_IDENTITY,
                model=artifact.public_metadata(),
            )
        except Exception:
            if client.has_collection(collection_name=self.settings.collection_name):
                client.drop_collection(collection_name=self.settings.collection_name)
            if resource_registered:
                client.remove_file_resource(name=self.settings.model_resource_name)
            if uploaded:
                self._object_store.remove()
            raise
        finally:
            client.close()

    def cleanup(self) -> None:
        client = self._client_factory()
        errors: list[str] = []
        try:
            try:
                if client.has_collection(collection_name=self.settings.collection_name):
                    client.drop_collection(collection_name=self.settings.collection_name)
            except Exception as exc:
                errors.append(f"collection cleanup failed: {type(exc).__name__}")
            try:
                if self.settings.model_resource_name in file_resource_names(
                    client.list_file_resources()
                ):
                    client.remove_file_resource(name=self.settings.model_resource_name)
            except Exception as exc:
                errors.append(f"FileResource cleanup failed: {type(exc).__name__}")
            try:
                self._object_store.remove()
            except Exception as exc:
                errors.append(f"object cleanup failed: {type(exc).__name__}")
        finally:
            client.close()
        if errors:
            raise RuntimeError("; ".join(errors))

    def inspect_state(self) -> RuntimeState:
        client = self._client_factory()
        try:
            return RuntimeState(
                collection_exists=client.has_collection(
                    collection_name=self.settings.collection_name
                ),
                file_resource_exists=(
                    self.settings.model_resource_name
                    in file_resource_names(client.list_file_resources())
                ),
                model_object_exists=self._object_store.exists(),
            )
        finally:
            client.close()

    def _replace_file_resource(self, client: MilvusClientLike) -> None:
        if self.settings.model_resource_name in file_resource_names(client.list_file_resources()):
            client.remove_file_resource(name=self.settings.model_resource_name)
        client.add_file_resource(
            name=self.settings.model_resource_name,
            path=self.settings.model_object_name,
        )

    def _replace_collection(self, client: MilvusClientLike) -> None:
        if client.has_collection(collection_name=self.settings.collection_name):
            client.drop_collection(collection_name=self.settings.collection_name)

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("item_id", DataType.VARCHAR, max_length=32)
        schema.add_field("title", DataType.VARCHAR, max_length=256)
        schema.add_field("product_type", DataType.VARCHAR, max_length=64)
        schema.add_field("brand", DataType.VARCHAR, max_length=128)
        schema.add_field("color", DataType.VARCHAR, max_length=128)
        schema.add_field("material", DataType.VARCHAR, max_length=128)
        schema.add_field("style", DataType.VARCHAR, max_length=128)
        schema.add_field("node_name", DataType.VARCHAR, max_length=256)
        schema.add_field("description", DataType.VARCHAR, max_length=4096)
        schema.add_field("main_image_id", DataType.VARCHAR, max_length=64)
        schema.add_field("display_price_usd", DataType.FLOAT)
        schema.add_field("rating_value", DataType.FLOAT)
        schema.add_field("clicks_30d", DataType.INT64)
        schema.add_field("sales_30d", DataType.INT64)
        schema.add_field("inventory_units", DataType.INT64)
        schema.add_field("inventory_capacity", DataType.INT64)
        schema.add_field("release_date", DataType.VARCHAR, max_length=10)
        schema.add_field("release_epoch", DataType.INT64)
        schema.add_field("rating", DataType.FLOAT)
        schema.add_field("inventory", DataType.FLOAT)
        schema.add_field("return_rate", DataType.FLOAT)
        schema.add_field("freshness", DataType.FLOAT)
        schema.add_field("image_path", DataType.VARCHAR, max_length=128)
        schema.add_field("image_mime", DataType.VARCHAR, max_length=32)
        schema.add_field("image_width", DataType.INT64)
        schema.add_field("image_height", DataType.INT64)
        schema.add_field("image_sha256", DataType.VARCHAR, max_length=64)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 8, "efConstruction": 64},
        )
        client.create_collection(
            collection_name=self.settings.collection_name,
            schema=schema,
            index_params=index_params,
        )
        client.insert(
            collection_name=self.settings.collection_name,
            data=[product.as_milvus_row() for product in PRODUCTS],
        )
        client.flush(collection_name=self.settings.collection_name)
        client.load_collection(collection_name=self.settings.collection_name)


def build_chain_steps(model_resource_name: str) -> list[dict[str, str]]:
    """Describe each Function Chain step with real, executable Python code.

    These code strings are the exact ``pymilvus.function_chain`` builder calls the
    backend runs, so hovering a step in the UI shows the real query-time
    computation -- not a mock animation.
    """
    return [
        {
            "name": "popularity",
            "operation": "num_combine",
            "output": "popularity",
            "description": "Weighted blend of 30-day clicks and sales.",
            "code": (
                ".map(\n"
                '    "popularity",\n'
                "    fn.num_combine(\n"
                '        col("clicks_30d"),\n'
                '        col("sales_30d"),\n'
                '        mode="weighted",\n'
                f"        weights=[{format(POPULARITY_CLICKS_COEFF, '.8g')}, "
                f"{format(POPULARITY_SALES_COEFF, '.8g')}],  # 0.3/12000, 0.7/850\n"
                "    ),\n"
                ")"
            ),
        },
        {
            "name": "price_affinity",
            "operation": "decay",
            "output": "price_affinity",
            "description": "Linear price decay: cheaper listings score higher.",
            "code": (
                ".map(\n"
                '    "price_affinity",\n'
                "    fn.decay(\n"
                '        col("display_price_usd"),\n'
                f'        function="{PRICE_DECAY_FUNCTION}",\n'
                f"        origin={PRICE_DECAY_ORIGIN},\n"
                f"        scale={PRICE_DECAY_SCALE},\n"
                f"        offset={PRICE_DECAY_OFFSET},\n"
                f"        decay={PRICE_DECAY_VALUE},\n"
                "    ),\n"
                ")"
            ),
        },
        {
            "name": "freshness",
            "operation": "decay",
            "output": "freshness",
            "description": "Exponential recency decay from the release epoch.",
            "code": (
                ".map(\n"
                '    "freshness",\n'
                "    fn.decay(\n"
                '        col("release_epoch"),\n'
                f'        function="{FRESHNESS_DECAY_FUNCTION}",\n'
                f"        origin={FRESHNESS_DECAY_ORIGIN},\n"
                f"        scale={FRESHNESS_DECAY_SCALE},\n"
                f"        offset={FRESHNESS_DECAY_OFFSET},\n"
                f"        decay={FRESHNESS_DECAY_VALUE},\n"
                "    ),\n"
                ")"
            ),
        },
        {
            "name": "semantic_round",
            "operation": "round_decimal",
            "output": "normalized_semantic_score",
            "description": "Round the vector similarity score for stable ranking.",
            "code": (
                ".map(\n"
                '    "normalized_semantic_score",\n'
                '    fn.round_decimal(col("$score"), '
                f"decimal={SEMANTIC_SCORE_DECIMALS}),\n"
                ")"
            ),
        },
        {
            "name": "xgboost_rerank",
            "operation": "xgboost",
            "output": "$score",
            "description": "XGBoost combines the five features into a business score.",
            "code": (
                ".map(\n"
                '    "$score",\n'
                "    fn.xgboost(\n"
                '        col("normalized_semantic_score"),\n'
                '        col("rating"),\n'
                '        col("popularity"),\n'
                '        col("price_affinity"),\n'
                '        col("freshness"),\n'
                f"        model_resource={model_resource_name!r},\n"
                '        output="default",\n'
                "    ),\n"
                ")"
            ),
        },
    ]


class SearchService:
    def __init__(
        self,
        settings: DemoSettings,
        *,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.settings = settings
        self._client_factory = client_factory or default_client_factory(settings)

    def compare(self, query_text: str, query_id: str | None = None) -> SearchComparison:
        normalized_text = query_text.strip()
        if not normalized_text:
            raise ValueError("query_text cannot be empty")
        selected_query = None
        if query_id is not None:
            selected_query = next(
                (query for query in DATASET.queries if query.id == query_id), None
            )
            if selected_query is None:
                raise KeyError(query_id)
            if selected_query.query_text != normalized_text:
                raise ValueError("query_id and query_text do not identify the same fixed query")
        else:
            selected_query = next(
                (query for query in DATASET.queries if query.query_text == normalized_text), None
            )
        query_embedding = encode_query(normalized_text, DATASET.embedding_contract)
        client = self._client_factory()
        try:
            search_args = {
                "collection_name": self.settings.collection_name,
                "data": [list(query_embedding)],
                "anns_field": "embedding",
                "search_params": {"metric_type": "COSINE", "params": {"ef": 32}},
                "limit": SEARCH_LIMIT,
                "output_fields": OUTPUT_FIELDS,
            }
            vector_results = client.search(**search_args)
            business_results = client.search(
                **search_args,
                function_chains=self._function_chain(),
            )
        finally:
            client.close()
        vector_order = self._ranked_products(vector_results)
        business_order = self._ranked_products(business_results)
        return SearchComparison(
            query_id=selected_query.id if selected_query else None,
            query_text=normalized_text,
            execution_path="milvus_l0_xgboost_function_chain",
            model_resource_name=self.settings.model_resource_name,
            function_chain={
                "stage": "L0_RERANK",
                "operation": "xgboost",
                "feature_order": list(FEATURE_NAMES),
                "parallel_inputs": False,
                "intermediate_feature_orders": False,
                "vector_order_score": "semantic_score",
                "business_order_score": "business_score",
                "chain_steps": self._chain_steps(),
            },
            dataset=DATASET_IDENTITY,
            data_model=self._data_model(vector_order, business_order),
            vector_order=vector_order,
            business_order=business_order,
        )

    def _chain_steps(self) -> list[dict[str, str]]:
        return build_chain_steps(self.settings.model_resource_name)

    @staticmethod
    def _chain_trace(
        semantic_score: float, product: RankedProduct, business_score: float
    ) -> dict[str, object]:
        """Compute one real candidate's values through every chain step.

        This mirrors the on-chain formulas in plain Python so the UI can show the
        exact intermediate values Milvus produced for a real ranked product.
        """
        popularity = round(chain_popularity(product.clicks_30d, product.sales_30d), 8)
        price_affinity = round(chain_price_affinity(product.display_price_usd), 8)
        freshness = round(chain_freshness(product.release_epoch), 8)
        rounded_semantic = round(float(semantic_score), SEMANTIC_SCORE_DECIMALS)
        return {
            "item_id": product.item_id,
            "title": product.title or product.product_type,
            "steps": [
                {
                    "name": "semantic_score",
                    "value": rounded_semantic,
                    "inputs": {"$score": f"{float(semantic_score):.8f}"},
                    "note": "vector search cosine similarity",
                },
                {
                    "name": "popularity",
                    "value": popularity,
                    "inputs": {
                        "clicks_30d": product.clicks_30d,
                        "sales_30d": product.sales_30d,
                    },
                    "note": "num_combine weighted blend",
                },
                {
                    "name": "price_affinity",
                    "value": price_affinity,
                    "inputs": {"display_price_usd": product.display_price_usd},
                    "note": "linear decay, cheaper scores higher",
                },
                {
                    "name": "freshness",
                    "value": freshness,
                    "inputs": {"release_epoch": product.release_epoch},
                    "note": "exponential decay, newer scores higher",
                },
                {
                    "name": "rating",
                    "value": round(product.rating, 8),
                    "inputs": {"rating": product.rating},
                    "note": "derived rating feature",
                },
                {
                    "name": "business_score",
                    "value": round(float(business_score), 6),
                    "inputs": {
                        "semantic_score": rounded_semantic,
                        "rating": round(product.rating, 8),
                        "popularity": popularity,
                        "price_affinity": price_affinity,
                        "freshness": freshness,
                    },
                    "note": "XGBoost combines five features",
                },
            ],
        }

    def _data_model(self, vector_order, business_order) -> dict[str, object]:
        """Return the collection schema plus a real top-1 chain trace."""
        top_vector = vector_order[0]
        top_business = business_order[0]
        return {
            "fields": list(COLLECTION_SCHEMA_FIELDS),
            "chain_trace": self._chain_trace(top_vector.score, top_business, top_business.score),
        }

    def _function_chain(self) -> FunctionChain:
        return (
            FunctionChain(
                FunctionChainStage.L0_RERANK,
                name="xgb_business_rerank",
            )
            .map(
                "popularity",
                fn.num_combine(
                    col("clicks_30d"),
                    col("sales_30d"),
                    mode="weighted",
                    weights=[POPULARITY_CLICKS_COEFF, POPULARITY_SALES_COEFF],
                ),
            )
            .map(
                "price_affinity",
                fn.decay(
                    col("display_price_usd"),
                    function=PRICE_DECAY_FUNCTION,
                    origin=PRICE_DECAY_ORIGIN,
                    scale=PRICE_DECAY_SCALE,
                    offset=PRICE_DECAY_OFFSET,
                    decay=PRICE_DECAY_VALUE,
                ),
            )
            .map(
                "freshness",
                fn.decay(
                    col("release_epoch"),
                    function=FRESHNESS_DECAY_FUNCTION,
                    origin=FRESHNESS_DECAY_ORIGIN,
                    scale=FRESHNESS_DECAY_SCALE,
                    offset=FRESHNESS_DECAY_OFFSET,
                    decay=FRESHNESS_DECAY_VALUE,
                ),
            )
            .map(
                "normalized_semantic_score",
                fn.round_decimal(col("$score"), decimal=SEMANTIC_SCORE_DECIMALS),
            )
            .map(
                "$score",
                fn.xgboost(
                    col("normalized_semantic_score"),
                    col("rating"),
                    col("popularity"),
                    col("price_affinity"),
                    col("freshness"),
                    model_resource=self.settings.model_resource_name,
                    output="default",
                ),
            )
        )

    @staticmethod
    def _ranked_products(results: list[list[dict[str, Any]]]) -> tuple[RankedProduct, ...]:
        if not results or not results[0]:
            raise RuntimeError("Milvus returned no search results")
        ranked: list[RankedProduct] = []
        for rank, hit in enumerate(results[0], start=1):
            product = DATASET.product_by_id(int(hit["id"]))
            ranked.append(RankedProduct.from_product(product, rank, float(hit["distance"])))
        return tuple(ranked)
