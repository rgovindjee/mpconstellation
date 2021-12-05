MU_EARTH = 3.986004418E14  # m3 s−2
R_EARTH = 6.371E6 # m -> mean radius of earth
#R_EARTH = 6.378E6  # m -> equatorial radius of earth
J2 = 1.08262668E-3 # J2 constant for earth
G0 = 9.80665  # m/s2
ISP = 500  # s
C_D = 2.5 # satellite C_Ds generally range from 2.0 (for a spherical body) to 3.0
SA = 55.44 # m^2 -> rough area of Hubble Space Telescope, sourced from Wikipedia


class Constants:
    def __init__(self, MU, R_E, J2, G0, ISP, SA, R0, RHO):
        self.MU = MU
        self.R_E = R_E
        self.J2 = J2
        self.G0 = G0
        self.ISP = ISP
        self.SA = SA
        self.R0 = R0
        self.RHO = RHO