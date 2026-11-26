"""Entity extraction contracts shared with instructor-rs."""

from enum import Enum

import pytest
from pydantic import BaseModel

import instructor

pytestmark = pytest.mark.paired(only=True)


class ShipmentState(str, Enum):
    PACKED = "packed"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"


class Package(BaseModel):
    tracking_id: str
    state: ShipmentState
    carrier: str | None = None


class Manifest(BaseModel):
    depot: str
    packages: list[Package]


@pytest.mark.asyncio
async def test_enum_extraction(provider_config: tuple[str, instructor.Mode]) -> None:
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    state = await client.create(
        response_model=ShipmentState,
        messages=[
            {
                "role": "user",
                "content": "The package is in transit. Extract its shipment state.",
            }
        ],
    )

    assert state is ShipmentState.IN_TRANSIT


@pytest.mark.asyncio
async def test_nested_enum_collection(
    provider_config: tuple[str, instructor.Mode],
) -> None:
    model, mode = provider_config
    client = instructor.from_provider(model, mode=mode, async_client=True)

    manifest = await client.create(
        response_model=Manifest,
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract the manifest for the North depot in the given order. "
                    "Package A1 is packed, carrier unknown. Package B2 is in transit "
                    "with carrier Swift. Use null for an unknown carrier."
                ),
            }
        ],
    )

    assert manifest.depot == "North"
    assert len(manifest.packages) == 2
    first, second = manifest.packages
    assert first.tracking_id == "A1"
    assert first.state is ShipmentState.PACKED
    assert first.carrier is None
    assert second.tracking_id == "B2"
    assert second.state is ShipmentState.IN_TRANSIT
    assert second.carrier == "Swift"
