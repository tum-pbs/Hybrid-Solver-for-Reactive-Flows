# ----------------------------------------------------------------------------
#
# Planar-v0 PDD framework
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
from solver_class_train import SpFluid

log = logging.getLogger()
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

params = {}
parser = argparse.ArgumentParser(description='Parameter Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu',            default='0',                                  help='visible GPUs')
parser.add_argument('-s', '--scale',    default=4, type=int,                          help='simulation scale for high-res')
parser.add_argument('-r', '--res',      default=32, type=int,                         help='resolution of the reference axis')
parser.add_argument('--testset',          default=None,              help='training; will load data from this simulation folder (no reaction)')
parser.add_argument('--bx',       default=0.05, type=float,   help='length of combustion chamber (box)')
parser.add_argument('-l', '--len',      default=100, type=int,                        help='length of the reference axis')
parser.add_argument('-b', '--sbatch',   default=1, type=int,       help='size of a batch; when 10 simulations with the size of 5, 5 simulations are into two batches')
parser.add_argument('--bIdx',   default=0, type=int,       help='batch index to compute simulation number')
parser.add_argument('--er',             default=1.0, type=float,                      help='Equivalence ratio')
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

def to_feature(smokestates, E0):
    # input feature; drop the unused edges of the staggered velocity grid making its dim same to the centered grid's
    #print(len(smokestates))
    with tf.name_scope('to_feature') as scope:
        return math.concat(
            [smokestates[j].temperature.data for j in range(len(smokestates))] +
            [smokestates[j].Yf.data for j in range(len(smokestates))] +
            [smokestates[j].Yo.data for j in range(len(smokestates))] +
            [E0],  # equivalence ratio
            axis=-1
        )

def mask_mass_fraction_new(tf_st_co_prd, tf_cv_md, e0, var_name, box):
    y_final =  tf_cv_md
    for d in range(y_final.data.shape[0]):
        Zf, Zo = get_zf_zo(e0[d])
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


# initialize
st = SpFluid(Domain([params['res'], params['res']], box=AABox(0, [params['bx'], params['bx']]), boundaries=CLOSED), buoyancy_factor=0, batch_size=params['sbatch'])
e0 = []
for b in range(params['sbatch']):
    sim_no = str(params['bIdx']*params['sbatch'] + b).zfill(6)
    if int(params['bIdx']*params['sbatch'] + b) > 3:
        sim_no = str(3).zfill(6)
    initTnr = params['testset'] + '/sim_' + sim_no + '/temp_000000.npz'
    initYfnr = params['testset'] + '/sim_' + sim_no + '/Yf_000000.npz'
    initYonr = params['testset'] + '/sim_' + sim_no + '/Yo_000000.npz'
    with open(params['testset'] + '/sim_' + sim_no + '/params.pickle', 'rb') as f:
        params0 = pickle.load(f)

    if b == 0:
        T0 = read_zipped_array(initTnr)
        Yf0 = read_zipped_array(initYfnr)
        Yo0 = read_zipped_array(initYonr)
        E0 = np.ones((1,32,32,1))*params0['er']

    else:
        T0 = np.concatenate((T0,read_zipped_array(initTnr)),axis=0)
        Yf0 = np.concatenate((Yf0,read_zipped_array(initYfnr)), axis=0)
        Yo0 = np.concatenate((Yo0, read_zipped_array(initYonr)), axis=0)
        E0 = np.concatenate((E0, np.ones((1,32,32,1))*params0['er']), axis=0)
    e0.append(params0['er'])

print(T0.shape, Yf0.shape, E0.shape)
st = [ st.copied_with(temperature=T0, Yf=Yf0, Yo=Yo0) for _ in range(1) ]  # NOTE: update according to the input feature!

# extra field
cv = st[0].staggered_grid(name="corr", value=0)

# phiflow scene
scene = Scene.create(directory=params['output'])
print('output dir - ',params['output'])
log.addHandler(logging.FileHandler(os.path.normpath(scene.path)+'/run.log'))
log.info(params)
log.info('tensorflow-{} ({}, {}); keras-{} ({})'.format(tf.__version__, tf.sysconfig.get_include(), tf.sysconfig.get_lib(), keras.__version__, keras.__path__))

log.info(E0)
if params['output']:
    with open(os.path.normpath(scene.path)+'/params.pickle', 'wb') as f: pickle.dump(params, f)


# load a tf model and stats used for data normalization
with open(params['stats'], 'rb') as f: data_stats = pickle.load(f)
log.info(data_stats)
model = keras.models.load_model(params['model'])  # Fully convolutional, so we can use trained weights regardless the input dimension
model.summary(print_fn=log.info)

scene.write(
    [st[-1].temperature, st[-1].Yf, st[-1].Yo, cv],
    ['tempTf', 'YfTf', 'YoTf', 'corTf'],
    0
)

for i in range(1, params['simsteps']):
    for j in range(len(st)-1):
        st[j] = st[j+1]


    inputf = (to_feature(st, E0) - [*(data_stats['mean'][0:3]), 0.0])/[
        *(data_stats['std'][0:3]), 1.0]

    cv_pred = model.predict(inputf)*([*(data_stats['std'][0:3])]) + ([*(data_stats['mean'][0:3])])
    cv = cv_pred[:, :, :, 0]
    cv_temp = CenteredGrid(cv.reshape((-1,32,32,1)), st[-1].temperature.box)
    cv = cv_pred[:, :, :, 1]
    cv_yf = CenteredGrid(cv.reshape((-1,32,32,1)), st[-1].Yf.box)
    cv = cv_pred[:, :, :, 2]
    cv_yo = CenteredGrid(cv.reshape((-1,32,32,1)), st[-1].Yo.box)
    final_yf = mask_mass_fraction_new(st, cv_yf, e0, 'Yf', st[-1].Yf.box)
    final_yo = mask_mass_fraction_new(st, cv_yo, e0, 'Yo', st[-1].Yo.box)

    st[-1] = st[-1].copied_with(temperature=cv_temp, Yf=final_yf, Yo=final_yo)

    log.info('step {:06d}'.format(i))
    scene.write(
        [st[-1].temperature, st[-1].Yf, st[-1].Yo, cv],
        ['tempTf', 'YfTf', 'YoTf', 'corTf'],
        i
    )
