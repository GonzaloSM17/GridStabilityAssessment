"""Published bus measurements and named power-transfer signals."""

from dataclasses import dataclass
import re

from data.time_series import TimeSeriesDataset


VOLTAGE_RMS = "voltage_rms"
FREQUENCY = "frequency"
SUPPORTED_MEASUREMENTS = (VOLTAGE_RMS, FREQUENCY)
MEASUREMENT_SUFFIXES = {
    VOLTAGE_RMS: "u_rms",
    FREQUENCY: "Frequency_Meter/f",
}


NORTH_TO_SOUTH_TRANSFER_PATHS = (
    ("KIMAL", "CHANGOS"),
    ("CHANGOS", "JADRESIC"),
    ("JADRESIC", "CUMBRE"),
    ("CUMBRE", "NCARDONES"),
    ("NCARDONES", "NMAITENCILLO"),
    ("NMAITENCILLO", "NPAZUCAR"),
    ("NPAZUCAR", "POLPAICO"),
)

NORTH_TO_SOUTH_500_KV_BUSES = (
    ("KIMAL",),
    ("LOSCHANGOS", "LCHANGOS"),
    ("JADRESIC",),
    ("CUMBRE",),
    ("NCARDONES",),
    ("NMAITENCILLO",),
    ("NPAZUCAR",),
    ("POLPAICO",),
    ("ALTOJAHUEL", "AJAHUEL"),
    ("LOAGUIRRE",),
    ("ANCOA",),
    ("ERIOS",),
    ("CHARRUA",),
)


@dataclass(frozen=True)
class Measurement:
    """One published measurement at a bus."""

    bus_name: str
    source_name: str
    kind: str
    column_name: str


@dataclass(frozen=True)
class Bus:
    """One electrical bus with the supported measurements found in a study."""

    name: str
    source_name: str
    measurements: tuple[Measurement, ...]

    def measurement(self, kind: str) -> Measurement:
        """Return one supported measurement available at this bus."""
        for measurement in self.measurements:
            if measurement.kind == kind:
                return measurement
        raise KeyError(f"Bus {self.name} does not provide measurement: {kind}")

    def voltage_rms(self) -> Measurement:
        """Return the RMS-voltage measurement at this bus."""
        return self.measurement(VOLTAGE_RMS)

    def frequency(self) -> Measurement:
        """Return the frequency measurement at this bus."""
        return self.measurement(FREQUENCY)


@dataclass(frozen=True)
class PowerTransfer:
    """One explicitly published active-power transfer monitor."""

    name: str
    column_name: str


class SignalCatalog:
    """Discover buses and supported measurements from one published schema."""

    def __init__(
        self,
        buses: tuple[Bus, ...],
        power_transfers: tuple[PowerTransfer, ...],
    ) -> None:
        self._buses = buses
        self._power_transfers = power_transfers
        self._buses_by_key = {
            self._normalise_bus_name(bus.name): bus for bus in buses
        }

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataset) -> "SignalCatalog":
        """Build a bus catalog from the schema of one representative run."""
        dataset.inspect()
        column_names = {series.name for series in dataset.series}
        source_names = sorted(
            column_name.removesuffix("/u_rms")
            for column_name in column_names
            if column_name.endswith("/u_rms")
        )

        buses: list[Bus] = []
        for source_name in source_names:
            measurements = tuple(
                Measurement(
                    bus_name=cls._display_bus_name(source_name),
                    source_name=source_name,
                    kind=kind,
                    column_name=f"{source_name}/{suffix}",
                )
                for kind, suffix in MEASUREMENT_SUFFIXES.items()
                if f"{source_name}/{suffix}" in column_names
            )
            buses.append(
                Bus(
                    name=cls._display_bus_name(source_name),
                    source_name=source_name,
                    measurements=measurements,
                )
            )

        power_transfers = tuple(
            PowerTransfer(
                name=column_name.removesuffix("/P").removeprefix("PQ_"),
                column_name=column_name,
            )
            for column_name in sorted(column_names)
            if column_name.startswith("PQ_") and column_name.endswith("/P")
        )
        return cls(tuple(buses), power_transfers)

    @property
    def buses(self) -> tuple[Bus, ...]:
        """Return every discovered bus with RMS-voltage measurements."""
        return self._buses

    @property
    def power_transfers(self) -> tuple[PowerTransfer, ...]:
        """Return active-power transfer monitors explicitly published as ``PQ_* /P``."""
        return self._power_transfers

    def bus(self, name: str) -> Bus:
        """Resolve a logical bus name such as KIMAL_220 to a published source."""
        key = self._normalise_bus_name(name)
        try:
            return self._buses_by_key[key]
        except KeyError as error:
            raise KeyError(f"Bus not found in the signal catalog: {name}") from error

    @staticmethod
    def _display_bus_name(source_name: str) -> str:
        """Derive the logical bus name from an EMTP meter source name."""
        return source_name.removesuffix("_MTR")

    @staticmethod
    def _normalise_bus_name(name: str) -> str:
        """Compare bus names independently of separators and letter case."""
        return re.sub(r"[^A-Z0-9]", "", name.upper())


def order_power_transfers_north_to_south(
    transfers: tuple[PowerTransfer, ...],
) -> tuple[PowerTransfer, ...]:
    """Order known 500 kV transfer corridors from north to south.

    Any transfer outside the declared corridor remains available after the
    ordered list, sorted by its published name.
    """
    return tuple(sorted(transfers, key=lambda transfer: transfer_north_to_south_order_key(transfer.name)))


def transfer_north_to_south_order_key(name: str) -> tuple[int, str]:
    """Return the north-to-south sorting key for a published transfer name."""
    normalized_name = re.sub(r"[^A-Z0-9]", "", name.upper())
    for index, endpoints in enumerate(NORTH_TO_SOUTH_TRANSFER_PATHS):
        if all(endpoint in normalized_name for endpoint in endpoints):
            return index, normalized_name
    return len(NORTH_TO_SOUTH_TRANSFER_PATHS), normalized_name


def order_500_kv_buses_north_to_south(buses: tuple[Bus, ...]) -> tuple[Bus, ...]:
    """Order known 500 kV buses from north to south."""
    return tuple(sorted(buses, key=lambda bus: bus_north_to_south_order_key(bus.name)))


def bus_north_to_south_order_key(name: str) -> tuple[int, str]:
    """Return the north-to-south sorting key for a 500 kV bus name."""
    normalized_name = re.sub(r"[^A-Z0-9]", "", name.upper())
    for index, aliases in enumerate(NORTH_TO_SOUTH_500_KV_BUSES):
        if any(alias in normalized_name for alias in aliases):
            return index, normalized_name
    return len(NORTH_TO_SOUTH_500_KV_BUSES), normalized_name
