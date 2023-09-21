# ----------------------------------------------------------------------------
#
# Phiflow chemical reaction solver framework
# Copyright 2020 Nilam T, Nils Thuerey
#
# This program is free software, distributed under the terms of the
# Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# Apply correction network model
#
# ----------------------------------------------------------------------------

import os, sys, logging, argparse, pickle
from solver_class_flame_v1 import SpEnergy, SpFluid

log = logging.getLogger()
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

params = {}
parser = argparse.ArgumentParser(description='Parameter Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu',            default='0',                                  help='visible GPUs')
parser.add_argument('-s', '--scale',    default=4, type=int,                          help='simulation scale for high-res')
parser.add_argument('-r', '--res',      default=32, type=int,                         help='resolution of the reference axis')
parser.add_argument('--testset',          default=None,              help='training; will load data from this simulation folder (no reaction)')
parser.add_argument('--bx',       default=1.0, type=float,   help='length of combustion chamber (box)')
parser.add_argument('-l', '--len',      default=100, type=int,                        help='length of the reference axis')
parser.add_argument('-b', '--sbatch',   default=1, type=int,       help='size of a batch; when 10 simulations with the size of 5, 5 simulations are into two batches')
parser.add_argument('--bIdx',   default=0, type=int,       help='batch index to compute simulation number')
parser.add_argument('--er',             default=1.0, type=float,                      help='Equivalence ratio')
parser.add_argument('--amp',              default=15, type=float, help='excitation amplitude')
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

def to_feature(smokestates, ext_const_channel, ext_const_channelA):
    for d in range(smokestates[0].temperature.data.shape[0]):
        y_data = np.ones(shape=(1,32,32,1))*ext_const_channelA[d]
        if d == 0:
            yd_data = y_data
        else:
            yd_data = np.concatenate((yd_data, y_data),axis=0)

    with tf.name_scope('to_feature') as scope:
        return math.concat(
            [smokestates[j].temperature.data for j in range(len(smokestates))] +
            [smokestates[j].Yf.data for j in range(len(smokestates))] +
            [smokestates[j].Yo.data for j in range(len(smokestates))] +
            [np.ones(shape=smokestates[0].temperature.data.shape)*ext_const_channel] +
            [yd_data],  # equivalence ratio
            axis=-1
        )

def mask_mass_fraction_new(tf_st_co_prd, tf_cv_md, var_name, box):
    y_final =  tf_cv_md
    for d in range(y_final.data.shape[0]):
        Zf, Zo = get_zf_zo(tf_st_co_prd[-1].eq[d])
        y_data = y_final.data[d:(d+1),:,:,:]
        if var_name == 'Yf':
            thres = Zf
        elif var_name == 'Yo':
            thres = Zo
        if d == 0:
            yd_data = np.clip(y_data, 0, thres)
        else:
            yd_data = np.concatenate((yd_data, np.clip(y_data, 0, thres)),axis=0)

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


# initialize
st = SpFluid(Domain([params['res'], params['res']], box=AABox(0, [params['bx'], params['bx']]), boundaries=[OPEN,CLOSED]), buoyancy_factor=0, batch_size=params['sbatch'], amp=params['amp'], eq=params['er'])
print(st.amp)
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
A0, E0 = [], []
for b in range(params['sbatch']):
    sim_no = str(params['bIdx']*params['sbatch'] + b).zfill(6)
    initTnr = params['testset'] + '/sim_' + sim_no + '/temp_000000.npz'
    initYfnr = params['testset'] + '/sim_' + sim_no + '/Yf_000000.npz'
    initYonr = params['testset'] + '/sim_' + sim_no + '/Yo_000000.npz'
    initVnr = params['testset'] + '/sim_' + sim_no + '/vel_000000.npz'
    with open(params['testset'] + '/sim_' + sim_no + '/params.pickle', 'rb') as f:
        params0 = pickle.load(f)

    if b == 0:
        T0 = read_zipped_array(initTnr)
        Yf0 = read_zipped_array(initYfnr)
        Yo0 = read_zipped_array(initYonr)
        V0 = read_zipped_array(initVnr)
        #A0 = [params0['amp']]
    else:
        T0 = np.concatenate((T0,read_zipped_array(initTnr)),axis=0)
        Yf0 = np.concatenate((Yf0,read_zipped_array(initYfnr)), axis=0)
        Yo0 = np.concatenate((Yo0, read_zipped_array(initYonr)), axis=0)
        V0 = np.concatenate((V0, read_zipped_array(initVnr)), axis=0)#read_zipped_array(initVnr)
    A0.append(params0['amp']) #np.concatenate((A0, params0['amp']), axis=0)
    E0.append(params0['er'])

print(T0.shape, Yf0.shape, V0.shape, A0)
st = [ st.copied_with(temperature=T0, Yf=Yf0, Yo=Yo0, velocity=V0, amp=A0, eq=E0) for _ in range(1) ]  # NOTE: update according to the input feature!

# extra field
cv = st[0].staggered_grid(name="corr", value=0)

# phiflow scene
scene = Scene.create(directory=params['output'])
print('output dir - ',params['output'])
log.addHandler(logging.FileHandler(os.path.normpath(scene.path)+'/run.log'))
log.info(params)
log.info('tensorflow-{} ({}, {}); keras-{} ({})'.format(tf.__version__, tf.sysconfig.get_include(), tf.sysconfig.get_lib(), keras.__version__, keras.__path__))
log.info(A0)
log.info(E0)
if params['output']:
    with open(os.path.normpath(scene.path)+'/params.pickle', 'wb') as f: pickle.dump(params, f)


# load a tf model and stats used for data normalization
with open(params['stats'], 'rb') as f: data_stats = pickle.load(f)
log.info(data_stats)
model = keras.models.load_model(params['model'])  # Fully convolutional, so we can use trained weights regardless the input dimension
model.summary(print_fn=log.info)

scene.write(
    [st[-1].temperature, st[-1].Yf, st[-1].Yo, st[-1].velocity, cv],
    ['tempTf', 'YfTf', 'YoTf', 'velTf', 'corTf'],
    0
)

for i in range(1, params['simsteps']):
    for j in range(len(st)-1):
        st[j] = st[j+1]

    inputf = (to_feature(st, E0[0], A0) - [*(data_stats['mean'][0:3]), 0.0, data_stats['ext.mean'][0]])/[
        *(data_stats['std'][0:3]), 1.0, data_stats['ext.std'][0]
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

    st[-1] = st[-1].copied_with(temperature=cv_temp, Yf=final_yf, Yo=final_yo)

    log.info('step {:06d}'.format(i))
    scene.write(
        [st[-1].temperature, st[-1].Yf, st[-1].Yo, cv],
        ['tempTf', 'YfTf', 'YoTf', 'corTf'],
        i
    )