# Unit tests for the PowerGrid KPI computation.
import numpy as np

from integrations.powergrid.kpi import compute_kpis


class FakeObs:
    """Fake observation exposing a fixed worst line loading."""

    def __init__(self, max_rho):
        """
        Stores the worst line loading this observation reports.

        :param float max_rho: The worst line loading to expose.
        """
        self.rho = np.array([max_rho], dtype=np.float32)


def test_compute_kpis_uses_reached_max_rho():
    """Ensure the KPI is the worst line loading of the reached observation."""
    obs = FakeObs(max_rho=0.73)

    kpis = compute_kpis(obs)

    assert kpis["efficiency_of_the_reco"] == np.float32(0.73)
