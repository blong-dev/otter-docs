"""Infrastructure-section renderers.

One renderer per `InfraSurface` type. Each renderer is a pure function
of the surface record → markdown string. The bridge in
`otter_docs.render.infra.adapter` wires each detector + renderer pair
into the existing Renderer registry so `Repo.render_document` treats
infra sections like any other section.
"""

from otter_docs.render.infra.adapter import (
    InfraSectionRenderer,
    register_infra_sections,
)

__all__ = [
    "InfraSectionRenderer",
    "register_infra_sections",
]
