from pyfme.aircrafts.aircraft import Aircraft
from pyfme.models.constants import slugft2_2_kgm2, lbs2kg


class Phoenix2400(Aircraft):
    # Mass & Inertia
    self.mass = 1.160  # kg
    self.inertia = np.diag([948, 1346, 1967]) * slugft2_2_kgm2 * self.mass / (2300 * lbs2kg)  # kg·m²

    # Aircraft geometry
    self.chord = 0.10  # m
    self.span = 2.0  # m
    self.propeller_radius = 0.04  # 4cm
    self.Sw = self.span * 1.2  # m2
