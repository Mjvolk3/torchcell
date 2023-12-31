from _typeshed import Incomplete
from collections import MutableMapping as DictMixin
from intermine import idresolution as idresolution
from intermine.decorators import requires_version as requires_version
from intermine.errors import ServiceError as ServiceError, WebserviceError as WebserviceError
from intermine.lists.listmanager import ListManager as ListManager
from intermine.model import Attribute as Attribute, Collection as Collection, Column as Column, Model as Model, Reference as Reference
from intermine.query import Query as Query, Template as Template
from intermine.results import InterMineURLOpener as InterMineURLOpener, ResultIterator as ResultIterator

__organization__: str
__contact__: str

class Registry(DictMixin):
    MINES_PATH: str
    registry_url: Incomplete
    def __init__(self, registry_url: str = ...) -> None: ...
    def __contains__(self, name) -> bool: ...
    def __getitem__(self, name): ...
    def __setitem__(self, name, item) -> None: ...
    def __delitem__(self, name) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def keys(self): ...

def ensure_str(stringlike): ...

class Service:
    QUERY_PATH: str
    LIST_ENRICHMENT_PATH: str
    WIDGETS_PATH: str
    SEARCH_PATH: str
    QUERY_LIST_UPLOAD_PATH: str
    QUERY_LIST_APPEND_PATH: str
    MODEL_PATH: str
    TEMPLATES_PATH: str
    TEMPLATEQUERY_PATH: str
    LIST_PATH: str
    LIST_CREATION_PATH: str
    LIST_RENAME_PATH: str
    LIST_APPENDING_PATH: str
    LIST_TAG_PATH: str
    SAVEDQUERY_PATH: str
    VERSION_PATH: str
    RELEASE_PATH: str
    SCHEME: str
    SERVICE_RESOLUTION_PATH: str
    IDS_PATH: str
    USERS_PATH: str
    root: Incomplete
    prefetch_depth: Incomplete
    prefetch_id_only: Incomplete
    opener: Incomplete
    query: Incomplete
    def __init__(self, root, username: Incomplete | None = ..., password: Incomplete | None = ..., token: Incomplete | None = ..., prefetch_depth: int = ..., prefetch_id_only: bool = ...) -> None: ...
    LIST_MANAGER_METHODS: Incomplete
    def get_anonymous_token(self, url): ...
    def list_manager(self): ...
    def __getattribute__(self, name): ...
    def __getattr__(self, name): ...
    def __del__(self) -> None: ...
    @property
    def version(self): ...
    def resolve_service_path(self, variant): ...
    @property
    def release(self): ...
    def load_query(self, xml, root: Incomplete | None = ...): ...
    def select(self, *columns, **kwargs): ...
    new_query = select
    def get_template(self, name): ...
    def search(self, term, **facets): ...
    @property
    def widgets(self): ...
    def resolve_ids(self, data_type, identifiers, extra: str = ..., case_sensitive: bool = ..., wildcards: bool = ...): ...
    def flush(self) -> None: ...
    @property
    def templates(self): ...
    @property
    def model(self): ...
    def get_results(self, path, params, rowformat, view, cld: Incomplete | None = ...): ...
    def register(self, username, password): ...
    def get_deregistration_token(self, validity: int = ...): ...
    def deregister(self, deregistration_token): ...
