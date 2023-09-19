# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
#
# uniform-Bunsen hybrid NN-PDE framework
# Copyright 2023 Nilam T, Nils Thuerey
#
# This program is free software, distributed under the terms of the
# Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#
# Training
#
# ----------------------------------------------------------------------------

import os, sys, logging, argparse, pickle, glob, random, distutils.dir_util
from solver_class_train import SpEnergy, SpFluid

fuel_type = 'methane'
if fuel_type == 'methane':
    hk, cp = 5.01E7, 1450
    Wf, Wo, Wp = 0.016, 0.032, 0.062
    Vf, Vo = -1, -2
elif fuel_type == 'propane':
    hk, cp = 4.66E7, 1300
    Wf, Wo, Wp = 0.044, 0.032, 0.062
    Vf, Vo = -1, -5

log = logging.getLogger()
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

params = {}
parser = argparse.ArgumentParser(description='Parameter Parser', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--gpu',            default='0',               help='visible GPUs')
parser.add_argument('--cuda',           action='store_true',       help='enable CUDA for solver')
parser.add_argument('--train',          default=None,              help='training; will load data from this simulation folder (no reaction)')
parser.add_argument('--skip-ds',        action='store_true',       help='skip down-scaling; assume you have already saved')
parser.add_argument('--only-ds',        action='store_true',       help='exit after down-scaling and saving; use only for data pre-processing')
parser.add_argument('--res',       default=32, type=int,    help='resolution of the reference axis')
parser.add_argument('--log',            default=None,              help='path to a log file')
parser.add_argument('-s', '--scale',    default=4, type=int,       help='simulation scale for high-res')
parser.add_argument('--bx',       default=1, type=float,   help='length of combustion chamber (box)')
parser.add_argument('-n', '--nsims',    default=1, type=int,       help='number of simulations')
parser.add_argument('-b', '--sbatch',   default=1, type=int,       help='size of a batch; when 10 simulations with the size of 5, 5 simulations are into two batches')
parser.add_argument('-t', '--simsteps', default=299, type=int,    help='simulation steps; # of data samples (i.e. frames) per simulation')
parser.add_argument('-m', '--msteps',   default=2, type=int,       help='multi steps in training loss')
parser.add_argument('-e', '--epochs',   default=10, type=int,      help='training epochs')
parser.add_argument('--seed',           default=None, type=int,    help='seed for random number generator')
parser.add_argument('-l', '--len',      default=100, type=int,     help='length of the reference axis')  # FIXME: save and restore from the data
parser.add_argument('--model',          default='unet_v1',       help='(predefined) network model')
parser.add_argument('--reg-loss',       action='store_true',       help='turn on regularization loss')
parser.add_argument('--lr',             default=1e-3, type=float,  help='start learning rate')
parser.add_argument('--adplr',          action='store_true',       help='turn on adaptive learning rate')
parser.add_argument('--clip-grad',      action='store_true',       help='turn on clip gradients')
parser.add_argument('--resume',         default=100, type=int,      help='resume training epochs')
parser.add_argument('--inittf',         default=None,              help='load initial model weights (warm start)')
parser.add_argument('--pretf',          default=None,              help='load pre-trained weights (only for testing pre-trained supervised model; do not use for a warm start!)')
parser.add_argument('--tf',             default='/tmp/phiflow/tf', help='path to a tensorflow output dir (model, logs, etc.)')
sys.argv += ['--' + p for p in params if isinstance(params[p], bool) and params[p]]
pargs = parser.parse_args()
params.update(vars(pargs))

os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']

if params['cuda']: from phi.tf.tf_cuda_pressuresolver import CUDASolver

from phi.tf.flow import *
import phi.tf.util

import tensorflow as tf
from tensorflow import keras

if params['resume']>0 and params['log']:
    params['log'] = os.path.splitext(params['log'])[0] + '_resume{:04d}'.format(params['resume']) + os.path.splitext(params['log'])[1]

if params['log']:
    distutils.dir_util.mkpath(os.path.dirname(params['log']))
    log.addHandler(logging.FileHandler(params['log']))

if (params['nsims'] % params['sbatch']) != 0:
    params['nsims'] = (params['nsims']//params['sbatch'])*params['sbatch']
    log.info('Number of simulations is not divided by the batch size thus adjusted to {}'.format(params['nsims']))

log.info(params)
log.info('tensorflow-{} ({}, {}); keras-{} ({})'.format(tf.__version__, tf.sysconfig.get_include(), tf.sysconfig.get_lib(), keras.__version__, keras.__path__))

random.seed(0 if params['seed'] is None else params['seed'])
np.random.seed(0 if params['seed'] is None else params['seed'])
tf.compat.v1.set_random_seed(0 if params['seed'] is None else params['seed'])

def to_feature(smokestate, ext_const_channel):
    velocity_x0 = smokestate.velocity.staggered_tensor()[:, :-1:, :, 1:2]
    velocity_x1 = smokestate.velocity.staggered_tensor()[:, :-1:, 1: , 1:2]
    velocity_x2 = tf.zeros([smokestate.velocity.data[1].data.shape[0], 32, 1, 1])
    velocity_x3 = tf.concat((velocity_x1, velocity_x2), axis=2)
    velocity_x = (velocity_x0 + velocity_x3)/2
    with tf.name_scope('to_feature') as scope:
        return math.concat(
            [
                smokestate.temperature.data,
                smokestate.Yf.data,
                smokestate.Yo.data,
                smokestate.velocity.staggered_tensor()[:, :-1:, :-1:, 0:1],
                velocity_x[:,:,:-1:,:],
                tf.constant(shape=smokestate.temperature.data.shape, value=1.0)*math.reshape(value=ext_const_channel, shape=[smokestate._batch_size, 1, 1, 1]),
            ],
            axis=-1
        )

def reshaped_tensor(tensor, var_idx, target_shape, box):
    with tf.name_scope('reshaped_tensor') as scope:
        return CenteredGrid(tf.reshape(tensor[:,:,:,var_idx], target_shape), box=box)

def reshaped_tensor_v(tensor, var_idx, target_shape, box):
    with tf.name_scope('reshaped_tensor_v') as scope:
        return StaggeredGrid(tf.reshape(tensor[:,:,:,var_idx:(var_idx+2)], target_shape), box=box)

def get_zf_zo(eq):
    Zf = 1/(1 + (4*4.29/eq))
    Zo = 1-Zf
    return Zf, Zo

def mask_temperature_max(tf_st_co_prd, tf_cv_md, var_idx, box):
    T_min, T_max = 400, 4000
    y_final = reshaped_tensor(tf_cv_md[-1], var_idx, tf_st_co_prd[-1].temperature.data.shape, tf_st_co_prd[-1].temperature.box)
    y_data = y_final.data
    for d in range(y_final.data.shape[0]):
        yd = y_data[d, :, :, :]
        yd = tf.expand_dims(yd, axis=0)

        cond = tf.greater(yd, tf.ones(tf.shape(yd)) * T_max)
        mask = tf.where(cond, tf.ones(tf.shape(yd)), tf.zeros(tf.shape(yd)))
        mask2 = tf.ones(tf.shape(yd)) - mask
        if d == 0:
            Mask2 = mask2
            Mask = mask*T_max
        else:
            Mask2 = tf.concat((Mask2, mask2), axis=0)
            Mask = tf.concat((Mask, mask*T_max), axis=0)
        #y_data[d, :, :, :] = yd * mask2 + mask * thres
    return CenteredGrid(y_data*Mask2 + Mask, box=box)

def mask_temperature_min(y_final, box):
    T_min, T_max = 400, 4000
    y_data = y_final.data
    for d in range(y_final.data.shape[0]):
        yd = y_data[d, :, :, :]
        yd = tf.expand_dims(yd, axis=0)

        cond = tf.less(yd, tf.ones(tf.shape(yd)) * T_min)
        mask = tf.where(cond, tf.ones(tf.shape(yd)), tf.zeros(tf.shape(yd)))
        mask2 = tf.ones(tf.shape(yd)) - mask
        if d == 0:
            Mask2 = mask2
            Mask = mask*T_min
        else:
            Mask2 = tf.concat((Mask2, mask2), axis=0)
            Mask = tf.concat((Mask, mask*T_min), axis=0)
        #y_data[d, :, :, :] = yd * mask2 + mask * thres
    return CenteredGrid(y_data*Mask2 + Mask, box=box)

def mask_mass_fraction_new(tf_st_co_prd, tf_cv_md, var_idx, var_name, tf_st_er_in, box):
    if var_name == 'Yf':
        y_final = reshaped_tensor(tf_cv_md[-1], var_idx, tf_st_co_prd[-1].Yf.data.shape, tf_st_co_prd[-1].Yf.box)
    elif var_name == 'Yo':
        y_final = reshaped_tensor(tf_cv_md[-1], var_idx, tf_st_co_prd[-1].Yo.data.shape, tf_st_co_prd[-1].Yo.box)
    Zf, Zo = get_zf_zo(tf_st_er_in)
    y_data = y_final.data
    for d in range(y_final.data.shape[0]):
        yd = y_data[d, :, :, :]
        yd = tf.expand_dims(yd, axis=0)
        if var_name == 'Yf':
            thres = Zf[d]
        elif var_name == 'Yo':
            thres = Zo[d]
        cond = tf.greater(yd, tf.ones(tf.shape(yd)) * thres)
        mask = tf.where(cond, tf.ones(tf.shape(yd)), tf.zeros(tf.shape(yd)))
        mask2 = tf.ones(tf.shape(yd)) - mask
        if d == 0:
            Mask2 = mask2
            Mask = mask*thres
        else:
            Mask2 = tf.concat((Mask2, mask2), axis=0)
            Mask = tf.concat((Mask, mask*thres), axis=0)
        #y_data[d, :, :, :] = yd * mask2 + mask * thres
    return CenteredGrid(y_data*Mask2 + Mask, box=box)


def model_mercury(tensor_in):
    with tf.name_scope('model_mercury') as scope:
        return keras.Sequential([
            keras.layers.Input(tensor=tensor_in),
            keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation=tf.nn.relu),
            keras.layers.Conv2D(filters=64, kernel_size=5, padding='same', activation=tf.nn.relu),
            keras.layers.Conv2D(filters=2,  kernel_size=5, padding='same', activation=None),  # u, v
        ])

def conv2d_block(input_tensor, n_filters, kernel_size=5, batchnorm=False):
    x = keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal')(input_tensor)
    if batchnorm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal')(x)
    if batchnorm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    x = keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal')(x)
    if batchnorm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    return x

def model_unet_v1(tensor_in):
    n_filters = 16
    batchnorm = False
    with tf.name_scope('model_unet') as scope:

        l_input = keras.layers.Input(tensor=tensor_in)

        c1 = conv2d_block(l_input, n_filters = n_filters * 1, kernel_size=5, batchnorm=batchnorm)
        p1 = keras.layers.MaxPooling2D((2,2))(c1)

        c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=5, batchnorm=batchnorm)
        # expansive path
        u2 = keras.layers.Conv2DTranspose(n_filters*2, kernel_size=5, strides=2, padding='same')(c2)
        u2 = keras.layers.concatenate([u2, c1])
        c3 = conv2d_block(u2, n_filters=n_filters*1, kernel_size=5, batchnorm=batchnorm)

        l_output = keras.layers.Conv2D(filters=3, kernel_size=5, padding='same', kernel_initializer='he_normal')(c3)
        return keras.models.Model(inputs=l_input, outputs=l_output)

def lr_schedule(epoch, current_lr):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 10, 15, 20, 22 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = current_lr
    if   epoch == 23: lr *= 0.5
    elif epoch == 21: lr *= 1e-1
    elif epoch == 16: lr *= 1e-1
    elif epoch == 11: lr *= 1e-1
    return lr

class Chemical_reaction(SpEnergy):
    def __init__(self, pressure_solver=GeometricCG()):
        SpEnergy.__init__(self, pressure_solver)


    def step(self, smoke, eq=1.0, amp=1.0, dt=1.0, gravity=Gravity()):
        Q1 = 1
        wkf = Q1 * (Vf) * Wf
        wko = Q1 * (Vo) * Wo

        wt = hk * (wkf)
        smoke = smoke.copied_with(Wkf = wkf, Wko = wko, Wt = wt, amp= amp, eq=eq)
        return super().step(fluid=smoke, dt=dt, obstacles=(), gravity=gravity, density_effects=(), velocity_effects=())


class PhifDataset():
    def __init__(self, dirpath, num_frames, num_sims=None, batch_size=1, print_fn=print):
        self.dataSims      = sorted(glob.glob(dirpath + '/sim_0*'))[0:num_sims]
        self.pathsTemp     = [sorted(glob.glob(asim + '/temp_0*.npz')) for asim in self.dataSims]
        self.pathsYf       = [sorted(glob.glob(asim + '/Yf_0*.npz')) for asim in self.dataSims]
        self.pathsYo       = [sorted(glob.glob(asim + '/Yo_0*.npz')) for asim in self.dataSims]
        self.pathsVel      = [sorted(glob.glob(asim + '/vel_0*.npz')) for asim in self.dataSims]
        self.pathsPr       = [sorted(glob.glob(asim + '/pressure_0*.npz')) for asim in self.dataSims]
        self.dataFrms      = [ np.arange(num_frames) for _ in self.dataSims ]  # NOTE: may contain different numbers of frames
        self.batchSize     = batch_size
        self.epoch         = None
        self.epochIdx      = 0
        self.batch         = None
        self.batchIdx      = 0
        self.step          = None
        self.stepIdx       = 0
        self.dataPreloaded = None
        self.printFn       = print_fn

        self.numOfSims    = num_sims
        self.numOfBatchs  = self.numOfSims//self.batchSize
        self.numOfFrames  = num_frames
        self.numOfSteps   = num_frames

        self.printFn('Preload: Loading data from {} = {}'.format(dirpath, self.dataSims))
        self.dataPreloaded = {  # dataPreloaded['sim_key'][step #][0=temperature, 1=Yf]
            asim: [
                (
                    read_zipped_array(self.filenameToDownscaled(self.pathsTemp[j][i])),
                    read_zipped_array(self.filenameToDownscaled(self.pathsYf[j][i])),
                    read_zipped_array(self.filenameToDownscaled(self.pathsYo[j][i])),
                    read_zipped_array(self.filenameToDownscaled(self.pathsPr[j][i])),
                    read_zipped_array(self.filenameToDownscaled(self.pathsVel[j][i])),
                ) for i in range(num_frames)
            ] for j,asim in enumerate(self.dataSims)
        }

        self.resolution = self.dataPreloaded[self.dataSims[0]][0][0].shape  # [batch-size, y-size, x-size, dim]
        self.printFn(self.resolution)
        # TODO: need a sanity check for resolution over all data

        self.dataStats = {
            'mean': (
                np.mean(np.concatenate([np.absolute(self.dataPreloaded[asim][i][0].reshape(-1)) for asim in self.dataSims for i in range(num_frames)], axis=-1)),  # temp
                np.mean(np.concatenate([np.absolute(self.dataPreloaded[asim][i][1].reshape(-1)) for asim in self.dataSims for i in range(num_frames)], axis=-1)),  # Yf
                np.mean(np.concatenate([np.absolute(self.dataPreloaded[asim][i][2].reshape(-1)) for asim in self.dataSims for i in range(num_frames)], axis=-1)),  # Yo
                np.mean(np.concatenate([np.absolute(self.dataPreloaded[asim][i][3].reshape(-1)) for asim in self.dataSims for i in range(num_frames)], axis=-1)),  # Pr
                (
                    np.mean(np.concatenate([np.absolute(self.dataPreloaded[asim][i][4][..., 0].reshape(-1)) for asim in self.dataSims for i in range(num_frames)])),  # vel[0]
                    np.mean(np.concatenate([np.absolute(self.dataPreloaded[asim][i][4][..., 1].reshape(-1)) for asim in self.dataSims for i in range(num_frames)])),  # vel[1]
                ),
            ),
            'std': (
                np.std(np.concatenate([np.absolute(self.dataPreloaded[asim][i][0].reshape(-1)) for asim in self.dataSims for i in range(num_frames)], axis=-1)),  # temp
                np.std(np.concatenate([np.absolute(self.dataPreloaded[asim][i][1].reshape(-1)) for asim in self.dataSims for i in range(num_frames)], axis=-1)),  # Yf
                np.std(np.concatenate([np.absolute(self.dataPreloaded[asim][i][2].reshape(-1)) for asim in self.dataSims for i in range(num_frames)], axis=-1)),  # Yo
                np.std(np.concatenate([np.absolute(self.dataPreloaded[asim][i][3].reshape(-1)) for asim in self.dataSims for i in range(num_frames)], axis=-1)),  # Pr
                (
                    np.std(np.concatenate([np.absolute(self.dataPreloaded[asim][i][4][..., 0].reshape(-1)) for asim in self.dataSims for i in range(num_frames)])),  # vel[0]
                    np.std(np.concatenate([np.absolute(self.dataPreloaded[asim][i][4][..., 1].reshape(-1)) for asim in self.dataSims for i in range(num_frames)])),  # vel[1]
                ),
            )
        }

        self.extConstChannelPerSim = {}  # extConstChannelPerSim['sim_key'][0=first channel, ...]; for now, only equivalence ratio
        self.extConstChannelPerSim_amp = {}
        num_of_ext_channel = 1
        for j, asim in enumerate(self.dataSims):
            with open(asim+'/params.pickle', 'rb') as f:
                sim_params = pickle.load(f)
                self.extConstChannelPerSim[asim] = [ sim_params['er'] ] # equivalence ratio
                self.extConstChannelPerSim_amp[asim] = [sim_params['amp']]  # equivalence ratio

        self.dataStats.update({
            'ext.mean': [
                np.mean([np.absolute(self.extConstChannelPerSim[asim][i]) for asim in self.dataSims]) for i in range(num_of_ext_channel)  # equivalence ratio
            ],
            'ext.std': [
                np.std([np.absolute(self.extConstChannelPerSim[asim][i]) for asim in self.dataSims]) for i in range(num_of_ext_channel)  # equivalence ratio
            ]
        })
        self.printFn(self.dataStats)
        #self.printFn(self.extConstChannelPerSim_rd)

    def filenameToDownscaled(self, fname):
        return os.path.dirname(fname) + '/' + os.path.basename(fname)

    def newEpoch(self, exclude_tail=0, shuffle_data=True):
        self.numOfSteps = self.numOfFrames - exclude_tail
        simSteps = [ (asim, self.dataFrms[i][0:(len(self.dataFrms[i])-exclude_tail)]) for i,asim in enumerate(self.dataSims) ]
        sim_step_pair = []
        for i,_ in enumerate(simSteps):
            sim_step_pair += [ (i, astep) for astep in simSteps[i][1] ]  # (sim_idx, step) ...

        if shuffle_data: random.shuffle(sim_step_pair)
        self.epoch = [ list(sim_step_pair[i*self.numOfSteps:(i+1)*self.numOfSteps]) for i in range(self.batchSize*self.numOfBatchs) ]
        self.epochIdx += 1
        self.batchIdx = 0
        self.stepIdx = 0

    def nextBatch(self):  # batch size may be the number of simulations in a batch
        self.batchIdx += self.batchSize
        self.stepIdx = 0

    def nextStep(self):
        self.stepIdx += 1

    def getData(self, consecutive_frames, with_skip=1):
        t_r = [
            math.concat([
                self.dataPreloaded[
                    self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]]  # sim_key
                ][
                    self.epoch[self.batchIdx+i][self.stepIdx][1]+j*with_skip  # steps
                ][0]
                for i in range(self.batchSize)
            ], axis=0) for j in range(consecutive_frames+1)
        ]
        y_r = [
            math.concat([
                self.dataPreloaded[
                    self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]]  # sim_key
                ][
                    self.epoch[self.batchIdx+i][self.stepIdx][1]+j*with_skip  # steps
                ][1]
                for i in range(self.batchSize)
            ], axis=0) for j in range(consecutive_frames+1)
        ]
        yo_r = [
            math.concat([
                self.dataPreloaded[
                    self.dataSims[self.epoch[self.batchIdx + i][self.stepIdx][0]]  # sim_key
                ][
                    self.epoch[self.batchIdx + i][self.stepIdx][1] + j * with_skip  # steps
                    ][2]
                for i in range(self.batchSize)
            ], axis=0) for j in range(consecutive_frames + 1)
        ]
        p_r = [
            math.concat([
                self.dataPreloaded[
                    self.dataSims[self.epoch[self.batchIdx + i][self.stepIdx][0]]  # sim_key
                ][
                    self.epoch[self.batchIdx + i][self.stepIdx][1] + j * with_skip  # steps
                    ][3]
                for i in range(self.batchSize)
            ], axis=0) for j in range(consecutive_frames + 1)
        ]
        v_r = [
            math.concat([
                self.dataPreloaded[
                    self.dataSims[self.epoch[self.batchIdx + i][self.stepIdx][0]]  # sim_key
                ][
                    self.epoch[self.batchIdx + i][self.stepIdx][1] + j * with_skip  # steps
                    ][4]
                for i in range(self.batchSize)
            ], axis=0) for j in range(consecutive_frames + 1)
        ]
        ext = [
            self.extConstChannelPerSim[
                self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]]
            ][0] for i in range(self.batchSize)
        ]
        exA = [
            self.extConstChannelPerSim_amp[
                self.dataSims[self.epoch[self.batchIdx + i][self.stepIdx][0]]
            ][0] for i in range(self.batchSize)
        ]
        return [t_r, y_r, yo_r, p_r, v_r, ext, exA]


simulator_lo = Chemical_reaction()

dataset = PhifDataset(
    dirpath=params['train'],
    num_frames=params['simsteps'], num_sims=params['nsims'], batch_size=params['sbatch'],
    print_fn=log.info
)

if params['only_ds']: exit(0)

if params['pretf']:
    with open(os.path.dirname(params['pretf'])+'/stats.pickle', 'rb') as f: ld_stats = pickle.load(f)
    dataset.dataStats['in.std'] = (ld_stats['in.std'][0], (ld_stats['in.std'][1], ld_stats['in.std'][2]))
    dataset.dataStats['out.std'] = ld_stats['out.std']
    log.info(dataset.dataStats)

if params['resume']>0:
    with open(params['tf']+'/dataStats.pickle', 'rb') as f: dataset.dataStats = pickle.load(f)

Rlo = list(dataset.resolution)

st_co = SpFluid(Domain([params['res'], params['res']], box=AABox(0, [params['bx'], params['bx']]), boundaries=[OPEN,CLOSED]), buoyancy_factor=0, batch_size=params['sbatch'])
st_gt = [ SpFluid(Domain([params['res'], params['res']], box=AABox(0, [params['bx'], params['bx']]), boundaries=[OPEN,CLOSED]), buoyancy_factor=0, batch_size=params['sbatch']) for _ in range(params['msteps']) ]

with tf.name_scope('input') as scope:
    with tf.name_scope('co') as scope: tf_st_co_in =   phi.tf.util.placeholder_like(st_co)
    with tf.name_scope('lr') as scope: tf_vr_lr_in =   tf.placeholder(tf.float32, shape=[])  # learning rate
    with tf.name_scope('er') as scope: tf_st_er_in =   tf.placeholder(tf.float32, shape=[params['sbatch']])  # equivalence ratio
    with tf.name_scope('amp') as scope: tf_st_amp_in = tf.placeholder(tf.float32, shape=[params['sbatch']])  # amp
    with tf.name_scope('gt') as scope: tf_st_gt_in = [ phi.tf.util.placeholder_like(st_co) for _ in range(params['msteps']) ]

if (params['train'] is None):
    log.info(params['train'])
    log.info('No pre-loadable training data path is given.')
    exit(0)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
tf_session = tf.Session(config=config)

scene = Scene.create(params['train'], count=params['sbatch'], mkdir=False, copy_calling_script=False)
sess = Session(scene, session=tf_session)
tf.compat.v1.keras.backend.set_session(tf_session)

with tf.name_scope('training') as scope:
    netModel = eval('model_{}'.format(params['model']))
    model = netModel(to_feature(tf_st_co_in, tf_st_er_in))

    with tf.name_scope('corre') as scope:
        tf_st_co_prd, tf_cv_md = [], []
        for i in range(params['msteps']):
            with tf.name_scope('step_w_pred') as scope:
                with tf.name_scope('step') as scope:
                    tf_st_co_prd += [
                        simulator_lo.step(
                            tf_st_co_in if i == 0 else tf_st_co_prd[-1],
                            eq=tf_st_er_in,
                            amp=tf_st_amp_in,
                            dt=0.0005
                        )
                    ]

                with tf.name_scope('pred') as scope:
                    tf_cv_md += [
                        model(
                            (to_feature(tf_st_co_prd[-1], tf_st_er_in) - ([*(dataset.dataStats['mean'][0:3]), # temperature, Yf, Yo
                                                                           *(dataset.dataStats['mean'][4]),
                                                                           0.0])) / ([*(dataset.dataStats['std'][0:3]), # temperature, Yf, Yo
                                                                                      *(dataset.dataStats['std'][4]), 1.0])
                        )*([*(dataset.dataStats['std'][0:3])]) + ([*(dataset.dataStats['mean'][0:3])])
                    ]
                T_final = mask_temperature_max(tf_st_co_prd, tf_cv_md, 0, tf_st_co_prd[-1].temperature.box)
                T_final = mask_temperature_min(T_final, tf_st_co_prd[-1].temperature.box)
                Yf_final = mask_mass_fraction_new(tf_st_co_prd, tf_cv_md, 1, 'Yf', tf_st_er_in, tf_st_co_prd[-1].Yf.box)
                Yo_final = mask_mass_fraction_new(tf_st_co_prd, tf_cv_md, 2, 'Yo', tf_st_er_in, tf_st_co_prd[-1].Yo.box)
                tf_st_co_prd[-1] = tf_st_co_prd[-1].copied_with(temperature=T_final, Yf=Yf_final, Yo=Yo_final)

    with tf.name_scope('loss') as scope:
        loss_steps = [
            (tf.nn.l2_loss((tf_st_gt_in[i].temperature.data - tf_st_co_prd[i].temperature.data)/dataset.dataStats['std'][0]) +
            tf.nn.l2_loss((tf_st_gt_in[i].Yf.data - tf_st_co_prd[i].Yf.data)/dataset.dataStats['std'][1]) +
            tf.nn.l2_loss((tf_st_gt_in[i].Yo.data - tf_st_co_prd[i].Yo.data)/dataset.dataStats['std'][2]))
            for i in range(params['msteps'])
        ]

        loss = tf.reduce_sum(loss_steps)/params['msteps']
        for i,a_step_loss in enumerate(loss_steps): tf.compat.v1.summary.scalar(name='loss_step{:02d}'.format(i), tensor=a_step_loss)
        tf.compat.v1.summary.scalar(name='sum_step_loss', tensor=loss)

        total_loss = loss
        if params['reg_loss']:
            reg_loss = tf.reduce_sum(model.losses)
            total_loss += reg_loss
            tf.compat.v1.summary.scalar(name='regularization', tensor=reg_loss)

        tf.compat.v1.summary.scalar(name='total_loss', tensor=total_loss)
        tf.compat.v1.summary.scalar(name='lr', tensor=tf_vr_lr_in)

        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=tf_vr_lr_in)

        if params['clip_grad']:
            gvs = opt.compute_gradients(total_loss)
            clipped_gvs = [(tf.clip_by_norm(grad, clip_norm=0.001), var) for (grad, var) in gvs]
            train_step = opt.apply_gradients(clipped_gvs)

        else:
            train_step = opt.minimize(total_loss)

model.summary(print_fn=log.info)
sess.initialize_variables()

if params['pretf']:
    log.info('load a pre-trained model: {}'.format(params['pretf']))
    ld_model = keras.models.load_model(params['pretf'], compile=False)
    model.set_weights(ld_model.get_weights())

if params['inittf']:
    log.info('load an initial model (warm start): {}'.format(params['inittf']))
    ld_model = keras.models.load_model(params['inittf'], compile=False)
    model.set_weights(ld_model.get_weights())

if params['resume']<1:
    [ params['tf'] and distutils.dir_util.mkpath(params['tf']) ]
    with open(params['tf']+'/dataStats.pickle', 'wb') as f: pickle.dump(dataset.dataStats, f)
else:
    ld_model = keras.models.load_model(params['tf']+'/model_epoch{:04d}.h5'.format(params['resume']))
    model.set_weights(ld_model.get_weights())

tf_summary_merged = tf.compat.v1.summary.merge_all()
tf_writer_tr = tf.compat.v1.summary.FileWriter(params['tf']+'/summary/training')
if params['resume']<1: tf_writer_tr.add_graph(sess.graph)

current_lr = params['lr']
i_st = 0
for j in range(params['epochs']):  # training
    dataset.newEpoch(exclude_tail=params['msteps'])
    if j<params['resume']:
        log.info('resume: skipping {} epoch'.format(j+1))
        i_st += dataset.numOfSteps*dataset.numOfBatchs
        continue

    current_lr = lr_schedule(j, current_lr) if params['adplr'] else params['lr']
    for ib in range(dataset.numOfBatchs):   # for each batch
        for i in range(dataset.numOfSteps):  # for each step
            adata = dataset.getData(consecutive_frames=params['msteps'], with_skip=1)
            er_nr, amp_nr = adata[5], adata[6] # equivalence ratio, amp
            st_co = st_co.copied_with(temperature=adata[0][0], Yf=adata[1][0], Yo=adata[2][0], pressure=adata[3][0], velocity=adata[4][0])
            st_gt = [ st_gt[k].copied_with(temperature=adata[0][k+1], Yf=adata[1][k+1], Yo=adata[2][k+1], velocity=adata[4][k+1]) for k in range(params['msteps']) ]

            my_feed_dict = { tf_st_co_in: st_co, tf_st_er_in: er_nr, tf_st_amp_in: amp_nr, tf_vr_lr_in: current_lr }
            my_feed_dict.update(zip(tf_st_gt_in, st_gt))

            summary, _, l2 = sess.run([tf_summary_merged, train_step, total_loss], my_feed_dict)
            tf_writer_tr.add_summary(summary, i_st)
            i_st += 1

            log.info('epoch {:03d}/{:03d}, batch {:03d}/{:03d}, step {:04d}/{:04d}: loss={}'.format(
                j+1, params['epochs'], ib+1, dataset.numOfBatchs, i+1, dataset.numOfSteps, l2
            ))
            dataset.nextStep()

        dataset.nextBatch()

    if np.isfinite(l2): model.save(params['tf']+'/model_epoch{:04d}.h5'.format(j+1))

tf_writer_tr.close()
model.save(params['tf']+'/model.h5')
