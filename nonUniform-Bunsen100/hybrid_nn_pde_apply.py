# ----------------------------------------------------------------------------
#
# nonUniform-Bunsen100 hybrid NN-PDE framework
# Copyright 2023 Nilam T, Nils Thuerey
#
# This program is free software, distributed under the terms of the
# Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# Apply correction network model
#
# ----------------------------------------------------------------------------

import os, sys, logging, argparse, pickle
from solver_class_flame_new import SpEnergy, SpFluid

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
parser.add_argument('--gpu',            default='0',                                  help='visible GPUs')
parser.add_argument('-s', '--scale',    default=4, type=int,                          help='simulation scale for high-res')
parser.add_argument('-r', '--res',      default=100, type=int,                         help='resolution of the reference axis')
parser.add_argument('--testset',          default=None,              help='training; will load data from this simulation folder (no reaction)')
parser.add_argument('--bx',       default=1.5, type=float,   help='length of combustion chamber (box)')
parser.add_argument('-l', '--len',      default=100, type=int,                        help='length of the reference axis')
parser.add_argument('-b', '--sbatch',   default=1, type=int,       help='size of a batch; when 10 simulations with the size of 5, 5 simulations are into two batches')
parser.add_argument('--bIdx',   default=0, type=int,       help='batch index to compute simulation number')
parser.add_argument('--er',             default=1.0, type=float,                      help='Equivalence ratio')
parser.add_argument('--amp',              default=280, type=float, help='excitation amplitude')
parser.add_argument('--rd',         default=None,                                help='load rd  (e.g., rd_000000.npz)')
parser.add_argument('--initTnr',         default=None,                                help='load no-reaction temperature  (e.g., nr_temp_0000.npz)')
parser.add_argument('--initYfnr',         default=None,                               help='load no-reaction Yf (e.g., nr_Yf_0000.npz)')
parser.add_argument('--initYonr',         default=None,                               help='load no-reaction Yf (e.g., nr_Yo_0000.npz)')
parser.add_argument('--initVnr',         default=None,                               help='load no-reaction vel (e.g., nr_vel_0000.npz)')
parser.add_argument('-t', '--simsteps', default=300, type=int,                        help='simulation steps')
parser.add_argument('-o', '--output',   default='/tmp/phiflow/run',                   help='path to an output directory')
parser.add_argument('--stats',          default='/tmp/phiflow/data/dataStats.pickle', help='path to datastats')
parser.add_argument('--model',          default='/tmp/phiflow/tf/model.h5',           help='path to a tensorflow model')
sys.argv += ['--' + p for p in params if isinstance(params[p], bool) and params[p]]
pargs = parser.parse_args()
params.update(vars(pargs))

os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']

from phi.flow import *

import tensorflow as tf
from tensorflow import keras

def to_staggered(tensor_cen, box):
    with tf.name_scope('to_staggered') as scope:
        return StaggeredGrid(math.pad(tensor_cen, ((0,0), (0,1), (0,1), (0,0))), box=box)

def get_zf_zo(eq):
    Zf = 1/(1 + (4*4.29/eq))
    Zo = 1-Zf
    return Zf, Zo

def to_feature(smokestates, ext_const_channel):
    # input feature; drop the unused edges of the staggered velocity grid making its dim same to the centered grid's
    #print('smokestate length ',len(smokestates))
    #print(ext_const_channel)
    #velocity_x = np.zeros(smokestates[j].velocity.staggered_tensor()[:, :-1:, :, 1:2])
    velocity_x0 = [smokestates[j].velocity.staggered_tensor()[:, :-1:, :, 1:2] for j in range(len(smokestates))]
    velocity_x1 = [smokestates[j].velocity.staggered_tensor()[:, :-1:, 1:, 1:2] for j in range(len(smokestates))]
    velocity_x2 = [np.zeros([smokestates[j].velocity.data[1].data.shape[0], 100, 1, 1]) for j in range(len(smokestates))]
    #print(velocity_x0[0].shape, velocity_x1[0].shape, velocity_x2[0].shape)
    velocity_x3 = [np.concatenate((velocity_x1[j], velocity_x2[j]), axis=2) for j in range(len(smokestates))]
    velocity_x = [((velocity_x0[j] + velocity_x3[j]) / 2) for j in range(len(smokestates))]
    #print(velocity_x0[0].shape, velocity_x1[0].shape, velocity_x2[0].shape, velocity_x3[0].shape, velocity_x[0].shape)
    #print(smokestates[0].temperature.data.shape, smokestates[0].Yf.data.shape, velocity_x[0][:, : , :-1:, 0:1].shape, smokestates[0].velocity.staggered_tensor()[:, :-1:, :-1:, 0:1].shape)
    with tf.name_scope('to_feature') as scope:
        return math.concat(
            [smokestates[j].temperature.data for j in range(len(smokestates))] +
            [smokestates[j].Yf.data for j in range(len(smokestates))] +
            [smokestates[j].Yo.data for j in range(len(smokestates))] +
            [(smokestates[j].velocity.staggered_tensor()[:, :-1:, :-1:, 0:1]) for j in range(len(smokestates))] +
            [(velocity_x[j][:, : , :-1:, 0:1]) for j in range(len(smokestates))] +
            [np.ones(shape=smokestates[0].temperature.data.shape)*ext_const_channel],  # equivalence ratio
            axis=-1
        )

def mask_mass_fraction_new(tf_st_co_prd, tf_cv_md, var_name, box):
    y_final =  tf_cv_md
    Zf, Zo = get_zf_zo(tf_st_co_prd[-1].eq)
    y_data = y_final.data
    if var_name == 'Yf':
        thres = Zf
    elif var_name == 'Yo':
        thres = Zo
    yd_data = np.clip(y_data, 0, thres)

    return CenteredGrid(yd_data, box=box)

def mask_temperature_max(tf_cv_md, box):
    y_final =  tf_cv_md
    T_min, T_max = 400, 4000
    y_data = y_final.data
    cond = np.greater(y_data, np.ones(y_data.shape) * T_max)
    mask = np.where(cond, np.ones(y_data.shape), np.zeros(y_data.shape))
    mask2 = np.ones(y_data.shape) - mask

    return CenteredGrid(y_data*mask2 + mask*T_max, box=box)

def mask_temperature_min(tf_cv_md, box):
    y_final =  tf_cv_md
    T_min, T_max = 400, 4000
    y_data = y_final.data
    cond = np.less(y_data, np.ones(y_data.shape) * T_min)
    mask = np.where(cond, np.ones(y_data.shape), np.zeros(y_data.shape))
    mask2 = np.ones(y_data.shape) - mask

    return CenteredGrid(y_data*mask2 + mask*T_min, box=box)

class flame_case(SpEnergy):
    def __init__(self, pressure_solver=GeometricCG()):
        SpEnergy.__init__(self, pressure_solver)


    def step(self, smoke, rd_array, res, dt=1.0, gravity=Gravity()):

        Q1 = 1

        wkf = Q1 * (Vf) * Wf
        wko = Q1 * (Vo) * Wo

        wt = hk * (wkf)
        smoke = smoke.copied_with(Wkf = wkf, Wko = wko, Wt = wt)
        return super().step(fluid=smoke, rd_array=rd_array, res=res, dt=dt, obstacles=(), gravity=gravity, density_effects=(), velocity_effects=())


# initialize
st = SpFluid(Domain([params['res'], params['res']], box=AABox(0, [params['bx'], params['bx']]), boundaries=[OPEN,CLOSED]), buoyancy_factor=0, batch_size=params['sbatch'], amp=params['amp'], eq=params['er'])
#print('output dir - ',params['output'])
def InitialVelocity(field, uy, res):
    vn = field.staggered_tensor()
    vn[:, :, :, 0] = uy # bottom uy
    vn[:, :, :, 1] = 0.0 # bottom ux
    vn[:, int(res / 4):, 0:1, 0] = 0.0  # upper left wall
    vn[:, int(res / 4):, (res - 1):, 0] = 0.0  # upper right wall
    field = StaggeredGrid(vn, field.box)
    return field

v0 = InitialVelocity(st.velocity, uy=params['amp'], res=params['res'])

Zf = 1/(1 + (4*4.29/params['er']))
Zo = 1-Zf
t1 = np.linspace(-4, 8, params['res'])
temp1 = 1400 + 600 * np.tanh(t1)
yf = Zf * (-np.tanh(t1) + 1) / 2
yo = Zo * (-np.tanh(t1) + 1) / 2

# sigmoid
for idx1 in range(params['res']):
    st.Yf.data[:, idx1, :, :] = yf[idx1]
    st.Yo.data[:, idx1, :, :] = yo[idx1]
    st.temperature.data[:, idx1, :, :] = temp1[idx1]
st = st.copied_with(velocity = v0)

for b in range(params['sbatch']):
    sim_no = str(params['bIdx']*params['sbatch'] + b).zfill(6)
    initTnr = params['testset'] + '/sim_' + sim_no + '/nr_temp_000000.npz'
    initYfnr = params['testset'] + '/sim_' + sim_no + '/nr_Yf_000000.npz'
    initYonr = params['testset'] + '/sim_' + sim_no + '/nr_Yo_000000.npz'
    initVnr = params['testset'] + '/sim_' + sim_no + '/nr_vel_000000.npz'
    rd_file = params['testset'] + '/sim_' + sim_no + '/rd_000000.npz'
    if b == 0:
        T0 = read_zipped_array(initTnr)
        Yf0 = read_zipped_array(initYfnr)
        Yo0 = read_zipped_array(initYonr)
        V0 = read_zipped_array(initVnr)
        r_file = np.load(rd_file)
        rd_array = r_file[r_file.files[-1]]
        rd_array = rd_array.reshape((1,params['res'],params['res'],1))
    else:
        T0 = np.concatenate((T0,read_zipped_array(initTnr)),axis=0)
        Yf0 = np.concatenate((Yf0,read_zipped_array(initYfnr)), axis=0)
        Yo0 = np.concatenate((Yo0, read_zipped_array(initYonr)), axis=0)
        V0 = read_zipped_array(initVnr)
        r_file = np.load(rd_file)
        r0_array = r_file[r_file.files[-1]]
        r0_array = r0_array.reshape((1, params['res'], params['res'], 1))
        rd_array = np.concatenate((rd_array, r0_array), axis=0)

print(T0.shape, Yf0.shape, V0.shape, rd_array.shape)
st = [ st.copied_with(temperature=T0, Yf=Yf0, Yo=Yo0, velocity=V0) for _ in range(1) ]  # NOTE: update according to the input feature!

# extra field
cv = st[0].staggered_grid(name="corr", value=0)

# phiflow scene
scene = Scene.create(directory=params['output'])
print('output dir - ',params['output'])
log.addHandler(logging.FileHandler(os.path.normpath(scene.path)+'/run.log'))
log.info(params)
log.info('tensorflow-{} ({}, {}); keras-{} ({})'.format(tf.__version__, tf.sysconfig.get_include(), tf.sysconfig.get_lib(), keras.__version__, keras.__path__))

if params['output']:
    with open(os.path.normpath(scene.path)+'/params.pickle', 'wb') as f: pickle.dump(params, f)


# load a tf model and stats used for data normalization
with open(params['stats'], 'rb') as f: data_stats = pickle.load(f)
log.info(data_stats)
model = keras.models.load_model(params['model'])  # Fully convolutional, so we can use trained weights regardless the input dimension
model.summary(print_fn=log.info)

scene.write(
    [st[-1].temperature, st[-1].velocity, cv],
    ['tempTf', 'velTf', 'corTf'],
    0
)

simulator = flame_case()
for i in range(1, params['simsteps']):
    for j in range(len(st)-1):
        st[j] = st[j+1]

    st[-1] = simulator.step(st[-1], rd_array=rd_array, res=params['res'], dt=0.0004)

    inputf = (to_feature(st, params['er']) - [*(data_stats['mean'][0:3]), *(data_stats['mean'][4]), 0.0])/[
        *(data_stats['std'][0:3]), *(data_stats['std'][4]),    # velocity
        1.0    # er
    ]
    cv_pred = model.predict(inputf) * ([*(data_stats['std'][0:3])]) + ([*(data_stats['mean'][0:3])])
    cv = cv_pred[:, :, :, 0]
    cv_temp = CenteredGrid(cv.reshape((-1, params['res'], params['res'], 1)), st[-1].temperature.box)
    cv = cv_pred[:, :, :, 1]
    cv_yf = CenteredGrid(cv.reshape((-1, params['res'], params['res'], 1)), st[-1].Yf.box)
    cv = cv_pred[:, :, :, 2]
    cv_yo = CenteredGrid(cv.reshape((-1, params['res'], params['res'], 1)), st[-1].Yo.box)
    final_yf = mask_mass_fraction_new(st, cv_yf, 'Yf', st[-1].Yf.box)
    final_yo = mask_mass_fraction_new(st, cv_yo, 'Yo', st[-1].Yo.box)

    # st[-1] = st[-1].copied_with(temperature=st[-1].temperature + cv_temp, Yf = st[-1].Yf+cv_yf, Yo=st[-1].Yo+cv_yo)
    st[-1] = st[-1].copied_with(temperature=cv_temp, Yf=final_yf, Yo=final_yo)

    log.info('step {:06d}'.format(i))
    scene.write(
        [st[-1].temperature, st[-1].Yf, st[-1].Yo, st[-1].velocity, cv],
        ['tempTf', 'YfTf', 'YoTf', 'velTf', 'corTf'],
        i
    )