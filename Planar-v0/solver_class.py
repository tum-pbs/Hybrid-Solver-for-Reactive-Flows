from phi.flow import *
from phi import math, struct

@struct.definition()
class SpFluid(DomainState):
    """
    A Fluid state consists of a density field (centered grid) and a velocity field (staggered grid).
    """

    def __init__(self, domain, velocity=0.0, temperature=0.0, pressure=101325.0, Yf=0.0, Yo=0.0, Wt=0.0, Wkf=0.0, Wko=0.0, buoyancy_factor=0.0, tags=('fluid', 'velocityfield'), name='fluid', **kwargs):
        DomainState.__init__(self, **struct.kwargs(locals()))
        self.solve_info = {}

    def default_physics(self): return SpeciesEnergy

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

    @struct.constant(default=0.0)
    def buoyancy_factor(self, fac):
        """
The default fluid physics can apply Boussinesq buoyancy as an upward force, proportional to the density.
This force is scaled with the buoyancy_factor (float).
        """
        return fac

    def __repr__(self):
        return "Fluid[velocity: %s, temperature: %s, pressure: %s, Yf: %s, Yo: %s, Wt: %s, Wkf: %s, Wko: %s]" % (self.velocity, self.temperature, self.pressure, self.Yf, self.Yo, self.Wt, self.Wkf, self.Wko)

def divergence_free_cr(velocity, density, domain=None, obstacles=(), pressure_solver=None, return_info=False):
    """
Projects the given velocity field by solving for and subtracting the pressure.
    :param return_info: if True, returns a dict holding information about the solve as a second object
    :param velocity: StaggeredGrid
    :param domain: Domain matching the velocity field, used for boundary conditions
    :param obstacles: list of Obstacles
    :param pressure_solver: PressureSolver. Uses default solver if none provided.
    :return: divergence-free velocity as StaggeredGrid
    """
    assert isinstance(velocity, StaggeredGrid)
    # --- Set up FluidDomain ---
    if domain is None:
        domain = Domain(velocity.resolution, OPEN)
    obstacle_mask = union_mask([obstacle.geometry for obstacle in obstacles])
    if obstacle_mask is not None:
        obstacle_grid = obstacle_mask.at(velocity.center_points).copied_with(extrapolation='constant')
        active_mask = 1 - obstacle_grid
    else:
        active_mask = math.ones(domain.centered_shape(name='active', extrapolation='constant'))
    accessible_mask = active_mask.copied_with(extrapolation=Material.accessible_extrapolation_mode(domain.boundaries))
    fluiddomain = FluidDomain(domain, active=active_mask, accessible=accessible_mask)
    # --- Boundary Conditions, Pressure Solve ---
    rho_u = velocity * density.at(velocity)
    divergence_field = rho_u.divergence(physical_units=False)
    pressure_c, iterations = solve_pressure(divergence_field, fluiddomain, pressure_solver=pressure_solver)
    pressure_c *= velocity.dx[0]
    gradp = StaggeredGrid.gradient(pressure_c)
    velocity -= fluiddomain.with_hard_boundary_conditions(gradp / density.at(velocity))
    #print('velocity 2', np.mean(velocity._data[0]._data))
    return velocity if not return_info else (velocity, {'pressure_c': pressure_c, 'iterations': iterations, 'divergence': divergence_field})

fuel_type = 'methane'
if fuel_type == 'methane':
    A_m, n_m, E_m = 1.1E7, 0, 83600
    hk, cp = 5.01E7, 1450
    Wf, Wo, Wp = 0.016, 0.032, 0.062
    Vf, Vo = -1, -2
elif fuel_type == 'propane':
    A_m, n_m, E_m = 2.75e8, 0, 130317
    hk, cp = 4.66E7, 1300
    Wf, Wo, Wp = 0.044, 0.032, 0.062
    Vf, Vo = -1, -5

class SpEnergy(Physics):
    """
Physics modelling the incompressible Navier-Stokes equations.
Supports buoyancy proportional to the marker density.
Supports obstacles, density effects, velocity effects, global gravity.
    """

    def __init__(self, pressure_solver=None):
        Physics.__init__(self, [StateDependency('obstacles', 'obstacle'),
                                StateDependency('gravity', 'gravity', single_state=True),
                                StateDependency('density_effects', 'density_effect', blocking=True),
                                StateDependency('velocity_effects', 'velocity_effect', blocking=True)])
        self.pressure_solver = pressure_solver

    def step(self, fluid, dt=1.0, obstacles=(), gravity=Gravity(), density_effects=(), velocity_effects=()):
        # pylint: disable-msg = arguments-differ
        diffusion_substeps = 1
        gravity = gravity_tensor(gravity, fluid.rank)
        velocity = fluid.velocity
        temperature = fluid.temperature
        pressure = fluid.pressure
        Yf, Yo = fluid.Yf, fluid.Yo
        wt, wkf, wko = fluid.Wt, fluid.Wkf, fluid.Wko

        # --- update density using state equation ---
        density = (1.0 / 8.314) * pressure * (((Yf * (1 / Wf) + Yo * (1 / Wo) + (1-Yf-Yo) * (1 / Wp)) * temperature) ** (-1))

        # --- momentum equation : Advection and diffusion velocity---
        velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)
        velocity = diffuse(velocity, 0.1*dt, substeps=diffusion_substeps)

        velocity += (density * gravity * fluid.buoyancy_factor * dt).at(velocity)
        # --- Effects ---
        for effect in density_effects:
            density = effect_applied(effect, density, dt)
        for effect in velocity_effects:
            velocity = effect_applied(effect, velocity, dt)

        # --- Pressure solve ---
        velocity, fluid.solve_info = divergence_free_cr(velocity, density, fluid.domain, obstacles, self.pressure_solver, return_info=True)
        pressure_c = fluid.solve_info.get('pressure_c', None)
        pressure += pressure_c
        #print(np.mean(pressure))
        # --- energy equation : Advection and diffusion temperature---


        rho_cpT = density * (cp) * temperature
        rho_cpT = advect.semi_lagrangian(rho_cpT, velocity, dt=dt)
        temperature = rho_cpT * ((density * cp) ** (-1))
        temperature = diffuse(temperature, dt*0.1, substeps=diffusion_substeps)
        temperature -= (wt * dt * ((density * cp)**(-1)))

        # --- species transport ---
        Yf = advect.semi_lagrangian(Yf, velocity, dt=dt)
        Yf = diffuse(Yf, 0.1*dt, substeps=diffusion_substeps)
        Yf += wkf * dt * (density ** -1)

        Yo = advect.semi_lagrangian(Yo, velocity, dt=dt)
        Yo = diffuse(Yo, 0.1 * dt, substeps=diffusion_substeps)
        Yo += wko * dt * (density ** -1)

        return fluid.copied_with(velocity=velocity, temperature= temperature, pressure = pressure, Yf = Yf, Yo = Yo, Wt = wt, Wkf = wkf, Wko = wko, age=fluid.age + dt)


SpeciesEnergy = SpEnergy()
