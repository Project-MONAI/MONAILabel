"""Read-only discovery of hosted models compatible with vision annotation."""

from monailabel.providers.catalog.discovery import Catalog, discover
from monailabel.providers.catalog.services import SERVICES, HostedService, ServiceId

__all__ = ["SERVICES", "Catalog", "HostedService", "ServiceId", "discover"]
