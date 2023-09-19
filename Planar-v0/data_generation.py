# ----------------------------------------------------------------------------
#
# Planar-v0 hybrid NN-PDE framework
# Copyright 2023 Nilam T, Nils Thuerey
#
# This program is free software, distributed under the terms of the
# Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# Data generation
#
# ----------------------------------------------------------------------------

import os, sys, logging, argparse, pickle, glob, random, distutils.dir_util
from solver_class import SpEnergy, SpFluid

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

log = logging.getLogger()
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

params = {}
parser = argparse.ArgumentParser(description='Parameter Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu',             default='0',             help='visible GPUs')
parser.add_argument('--cuda',            action='store_true',     help='enable CUDA for solver')
parser.add_argument('-o', '--output',    default=None,            help='path to an output directory')
parser.add_argument('-t', '--simsteps',  default=300, type=int,  help='simulation steps: an epoch')
parser.add_argument('-s', '--skipsteps', default=0, type=int,   help='skip first steps; (vortices may not form)')
parser.add_argument('-r', '--res',       default=32, type=int,    help='resolution of the reference axis')
parser.add_argument('--er',              default=1.0, type=float, help='fuel-air equivalence ratio')
parser.add_argument('-b', '--bx',       default=0.05, type=float,   help='length of combustion chamber (box)')
parser.add_argument('--seed',            default=0, type=int,     help='seed for random number generator')
sys.argv += ['--' + p for p in params if isinstance(params[p], bool) and params[p]]
pargs = parser.parse_args()
params.update(vars(pargs))

os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']

if params['cuda']: from phi.tf.tf_cuda_pressuresolver import CUDASolver

from phi.tf.flow import *
import phi.tf.util

import tensorflow as tf
from tensorflow import keras

random.seed(params['seed'])
np.random.seed(params['seed'])
tf.compat.v1.set_random_seed(params['seed'])

class Chemical_reaction(SpEnergy):
    def __init__(self, pressure_solver=GeometricCG()):
        SpEnergy.__init__(self, pressure_solver)


    def step(self, smoke, react, dt=1.0, gravity=Gravity()):
        density = (1.0 / 8.314) * smoke.pressure * (((smoke.Yf * (1 / Wf) + smoke.Yo * (1 / Wo) + (1 - smoke.Yf - smoke.Yo) * (1 / Wp)) * smoke.temperature) ** (-1))
        if react == 1:
            Q1 = A_m * (density * smoke.Yf/Wf) * ((density*smoke.Yo/Wo)**0.5) * keras.backend.exp(-E_m * ((8.314 * smoke.temperature.data) ** (-1)))
        else:
            Q1 = 1
        wkf = Q1 * (Vf) * Wf
        wko = Q1 * (Vo) * Wo

        wt = hk * (wkf)
        smoke = smoke.copied_with(velocity = 0.0, Wkf = wkf, Wko = wko, Wt = wt)
        return super().step(fluid=smoke, dt=dt, obstacles=(), gravity=gravity, density_effects=(), velocity_effects=())

rst = SpFluid(Domain([params['res'], params['res']], box=AABox(0, [params['bx'], params['bx']]), boundaries=CLOSED), buoyancy_factor=0)

Zf = 1/(1 + (4*4.29/params['er']))
Zo = 1-Zf
t1 = np.linspace(-15, 8, params['res'])
temp1 = 1200 + 800 * np.tanh(t1)
yf = Zf * (-np.tanh(t1) + 1) / 2
yo = Zo * (-np.tanh(t1) + 1) / 2

# sigmoid
for idx1 in range(params['res']):
    rst.Yf.data[:, idx1, :, :] = yf[idx1]
    rst.Yo.data[:, idx1, :, :] = yo[idx1]
    rst.temperature.data[:, idx1, :, :] = temp1[idx1]

Yf0, Yo0 = CenteredGrid(rst.Yf.data, rst.Yf.box), CenteredGrid(rst.Yo.data, rst.Yo.box)
T0 = CenteredGrid(rst.temperature.data, rst.temperature.box)
rst = rst.copied_with(temperature= T0, Yf = Yf0, Yo = Yo0)
# phiflow scene

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
tf_session = tf.Session(config=config)

scene = Scene.create(directory=params['output'])
sess  = Session(scene, session=tf_session)

log.addHandler(logging.FileHandler(os.path.normpath(scene.path)+'/run.log'))
log.info(params)
log.info('tensorflow-{} ({}, {}); keras-{} ({})'.format(tf.__version__, tf.sysconfig.get_include(), tf.sysconfig.get_lib(), keras.__version__, keras.__path__))

if params['output']:
    with open(os.path.normpath(scene.path)+'/params.pickle', 'wb') as f: pickle.dump(params, f)

simulator = Chemical_reaction()
tf_rst_in = phi.tf.util.placeholder_like(rst)
tf_rst = simulator.step(tf_rst_in, react=1, dt=0.000001)

if params['skipsteps']==0 and params['output'] is not None:

    scene.write(
        [rst.temperature, rst.Yf, rst.Yo, rst.pressure],
        ['temp', 'Yf', 'Yo', 'pressure'],
        0
    )

for i in range(1, params['simsteps']):

    my_feed_dict = {tf_rst_in: rst}
    rst = sess.run(tf_rst, my_feed_dict)

    log.info('Step {:06d}'.format(i))
    if params['skipsteps']<i and params['output'] is not None:
        scene.write(
            [rst.temperature, rst.Yf, rst.Yo, rst.pressure],
            ['temp', 'Yf', 'Yo', 'pressure'],
            i
        )

