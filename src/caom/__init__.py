"""caom — ChIP-Atlas Ontology Mapper."""

from caom.api import map_chipatlas
from caom.ontologies.update import update_ontologies
from caom.version import __version__

__all__ = ["map_chipatlas", "update_ontologies", "__version__"]
