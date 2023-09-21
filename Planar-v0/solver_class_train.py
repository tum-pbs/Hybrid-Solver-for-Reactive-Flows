from phi.flow import *
from phi import math, struct
import tensorflow as tf

@struct.definition()
class SpFluid(DomainState):
    """
    A Fluid state consists of a density field (centered grid) and a velocity field (staggered grid).
    """

    def __init__(self, domain, velocity=0.0, temperature=0.0, pressure=101325, Yf=0.0, Yo=0.0, Wt=0.0, Wkf=0.0, Wko=0.0, buoyancy_factor=0.0, amp=1.0, eq=1.0, tags=('fluid', 'velocityfield'), name='fluid', **kwargs):
        DomainState.__init__(self, **struct.kwargs(locals()))
        self.solve_info = {}

    @struct.variable(default=0, dependencies=DomainState.domain)
    def temperature(self, temperature):
        """
The marker temperature is stored in a CenteredGrid with dimensions matching the domain.
        """
        return self.centered_grid('temperature', temperature)

    @struct.variable(default=101325, dependencies=DomainState.domain)
    def pressure(self, pressure):
        """
The marker pressure is stored in a CenteredGrid with dimensions matching the domain.
        """
        return self.centered_grid('pressure', pressure)

    @struct.variable(default=0, dependencies=DomainState.domain)
    def Yf(self, Yf):
        """
The marker pressure is stored in a CenteredGrid with dimensions matching the domain.
        """
        return self.centered_grid('Yf', Yf)

    @struct.variable(default=0, dependencies=DomainState.domain)
    def Yo(self, Yo):
        """
The marker pressure is stored in a CenteredGrid with dimensions matching the domain.
        """
        return self.centered_grid('Yo', Yo)

    @struct.variable(default=0, dependencies=DomainState.domain)
    def Wt(self, Wt):
        """
The marker pressure is stored in a CenteredGrid with dimensions matching the domain.
        """
        return self.centered_grid('Wt', Wt)

    @struct.variable(default=0, dependencies=DomainState.domain)
    def Wkf(self, Wkf):
        """
The marker pressure is stored in a CenteredGrid with dimensions matching the domain.
        """
        return self.centered_grid('Wkf', Wkf)

    @struct.variable(default=0, dependencies=DomainState.domain)
    def Wko(self, Wko):
        """
The marker pressure is stored in a CenteredGrid with dimensions matching the domain.
        """
        return self.centered_grid('Wko', Wko)

    @struct.variable(default=0, dependencies=DomainState.domain)
    def velocity(self, velocity):
        """
The velocity is stored in a StaggeredGrid with dimensions matching the domain.
        """
        return self.staggered_grid('velocity', velocity)

    @struct.constant(default=1.0)
    def amp(self, amp):
        return amp

    @struct.constant(default=1.0)
    def eq(self, eq):
        return eq

    @struct.constant(default=0.0)
    def buoyancy_factor(self, fac):
        """
The default fluid physics can apply Boussinesq buoyancy as an upward force, proportional to the density.
This force is scaled with the buoyancy_factor (float).
        """
        return fac

    def __repr__(self):
        return "Fluid[velocity: %s, temperature: %s, pressure: %s, Yf: %s, Yo: %s, Wt: %s, Wkf: %s, Wko: %s]" % (self.velocity, self.temperature, self.pressure, self.Yf, self.Yo, self.Wt, self.Wkf, self.Wko)

