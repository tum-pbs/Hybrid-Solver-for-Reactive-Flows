# ----------------------------------------------------------------------------
#
# nonUniform-Bunsen32 hybrid NN-PDE framework
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
from solver_class_flame import SpEnergy, SpFluid


fuel_type = 'methane'
if fuel_type == 'methane':
    A_m, n_m, E_m = 5.1E4, 0, 93600
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
parser.add_argument('--gpu',             default='2',             help='visible GPUs')
parser.add_argument('--cuda',            action='store_true',     help='enable CUDA for solver')
parser.add_argument('-o', '--output',    default=None,            help='path to an output directory')
parser.add_argument('-t', '--simsteps',  default=300, type=int,  help='simulation steps: an epoch')
parser.add_argument('-r', '--res',       default=32, type=int,    help='resolution of the reference axis')
parser.add_argument('--er',              default=1.0, type=float, help='fuel-air equivalence ratio')
parser.add_argument('--amp',              default=15, type=float, help='excitation amplitude')
parser.add_argument('--bx',       default=1.0, type=float,   help='length of combustion chamber (box)')
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


class flame_case(SpEnergy):
    def __init__(self, pressure_solver=GeometricCG()):
        SpEnergy.__init__(self, pressure_solver)


    def step(self, smoke, react, rd_array, res, dt=1.0, gravity=Gravity()):

        density = (1.0 / 8.314) * smoke.pressure * (((smoke.Yf * (1 / Wf) + smoke.Yo * (1 / Wo) + (1 - smoke.Yf - smoke.Yo) * (1 / Wp)) * smoke.temperature) ** (-1))

        if ((react == 1) ) :
            print('here')
            Q1 = A_m * (density * smoke.Yf/Wf) * ((density*smoke.Yo/Wo)**0.5) * keras.backend.exp(-E_m * ((8.314 * smoke.temperature.data) ** (-1)))

        else:
            Q1 = 1

        wkf = Q1 * (Vf) * Wf
        wko = Q1 * (Vo) * Wo

        wt = hk * (wkf)
        smoke = smoke.copied_with(Wkf = wkf, Wko = wko, Wt = wt)
        return super().step(fluid=smoke, rd_array=rd_array, res=res, dt=dt, obstacles=(), gravity=gravity, density_effects=(), velocity_effects=())

rst = SpFluid(Domain([params['res'], params['res']], box=AABox(0, [params['bx'], params['bx']]), boundaries=[OPEN,CLOSED]), rd=Noise(channels=1), buoyancy_factor=0, amp=params['amp'], eq=params['er'])
# init temperature and mass fraction

def InitialVelocity(field, uy, res):
    vn = field.staggered_tensor()
    vn[:, :, :, 0] = uy # bottom uy
    vn[:, :, :, 1] = 0.0 # bottom ux
    vn[:, int(res / 4):, 0:1, 0] = 0.0  # upper left wall
    vn[:, int(res / 4):, (res - 1):, 0] = 0.0  # upper right wall
    field = StaggeredGrid(vn, field.box)
    return field

v0 = InitialVelocity(rst.velocity, uy=params['amp'], res=params['res'])

Zf = 1/(1 + (4*4.29/params['er']))
Zo = 1-Zf
t1 = np.linspace(-4, 8, params['res'])
temp1 = 1400 + 600 * np.tanh(t1)
yf = Zf * (-np.tanh(t1) + 1) / 2
yo = Zo * (-np.tanh(t1) + 1) / 2

# sigmoid
for idx1 in range(params['res']):
    rst.Yf.data[:, idx1, :, :] = yf[idx1]
    rst.Yo.data[:, idx1, :, :] = yo[idx1]
    rst.temperature.data[:, idx1, :, :] = temp1[idx1]

Yf0, Yo0 = CenteredGrid(rst.Yf.data, rst.Yf.box), CenteredGrid(rst.Yo.data, rst.Yo.box)
T0 = CenteredGrid(rst.temperature.data, rst.temperature.box)
rst = rst.copied_with(temperature= T0, Yf = Yf0, Yo = Yo0, velocity = v0)
# phiflow scene

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
tf_session = tf.Session(config=config)

scene = Scene.create(directory=params['output'])
sess  = Session(scene, session=tf_session)
if params['output'] is not None:
    scene.write(
        [rst.temperature, rst.Yf, rst.Yo, rst.velocity, rst.pressure, rst.rd, rst.Wt],
        ['temp', 'Yf', 'Yo', 'vel', 'pressure', 'rd', 'wt'],
        0
    )

scene_list = os.listdir(params['output'])
rd_file = params['output'] + '/' + scene_list[-1] + "/rd_000000.npz"

r_file = np.load(rd_file)
rd_array = r_file[r_file.files[-1]]

print(np.mean(rd_array), np.min(rd_array), np.max(rd_array))
log.addHandler(logging.FileHandler(os.path.normpath(scene.path)+'/run.log'))
log.info(params)
log.info('tensorflow-{} ({}, {}); keras-{} ({})'.format(tf.__version__, tf.sysconfig.get_include(), tf.sysconfig.get_lib(), keras.__version__, keras.__path__))

if params['output']:
    with open(os.path.normpath(scene.path)+'/params.pickle', 'wb') as f: pickle.dump(params, f)

simulator = flame_case()
with struct.VARIABLES:
    tf_rst_in = placeholder(rst.shape)

tf_rst = simulator.step(tf_rst_in, react=1, rd_array=rd_array, res=params['res'], dt=0.0005)


for i in range(1, params['simsteps']):
    my_feed_dict = {tf_rst_in: rst}
    rst = sess.run(tf_rst, my_feed_dict)

    if params['output'] is not None:
        scene.write(
            [rst.temperature, rst.Yf, rst.Yo, rst.velocity, rst.pressure, rst.Wt],
            ['temp', 'Yf', 'Yo', 'vel', 'pressure', 'wt'],
            i
        )
