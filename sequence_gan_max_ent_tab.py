import numpy as np
import tensorflow as tf
import random
import time
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout_max_ent import ROLLOUT
from target_lstm import TARGET_LSTM
from tabular_simple import TabularSimple
from gan_trainer import GanTrainer
import pickle
from enum import Enum
import yaml

class Runmode(Enum):
    fresh = 1
    con = 2
    skip = 3

with open("SeqGAN_ent_tab.yaml") as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)
#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = config['EMB_DIM'] # embedding dimension
HIDDEN_DIM = config['HIDDEN_DIM'] # hidden state dimension of lstm cell
SEQ_LENGTH = config['SEQ_LENGTH'] # sequence length
START_TOKEN = config['START_TOKEN']
PRE_GEN_EPOCH = config['PRE_GEN_EPOCH'] # supervise (maximum likelihood estimation) epochs for generator
PRE_DIS_EPOCH = config['PRE_DIS_EPOCH'] # supervise (maximum likelihood estimation) epochs for discriminator
SEED = config['SEED']
BATCH_SIZE = config['BATCH_SIZE']
ROLLOUT_UPDATE_RATE = config['ROLLOUT_UPDATE_RATE']
GENERATOR_LR = config['generator_lr']
REWARD_GAMMA = config['reward_gamma']
rollout_num = config['rollout_num']
#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = config['dis_embedding_dim']
dis_filter_sizes = config['dis_filter_sizes']
dis_num_filters = config['dis_num_filters']
dis_dropout_keep_prob = config['dis_dropout_keep_prob']
dis_l2_reg_lambda = config['dis_l2_reg_lambda']
dis_batch_size = config['dis_batch_size']
DIS_EPOCHS_PR_BATCH = config['epochs_discriminator_multiplier']
#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = config['TOTAL_BATCH']
# vocab size for our custom data
vocab_size = config['vocab_size']
# positive data, containing real music sequences
positive_file = config['positive_file']
# negative data from the generator, containing fake sequences
negative_file = config['negative_file']
valid_file = config['valid_file']
generated_num = config['generated_num']

epochs_generator = config['epochs_generator']
epochs_discriminator = config['epochs_discriminator']
pretrain = Runmode[config['runmode_pretrain']] #Enum('Runmode', config['runmode_pretrain'], module=__name__)
advtrain = Runmode[config['runmode_advtrain']]
pretrain_file = config['pretrain_file']
advtrain_file = config['advtrain_file']



def target_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)



def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    gen_data_loader = Gen_Data_loader(BATCH_SIZE,SEQ_LENGTH)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE,SEQ_LENGTH) # For testing
    dis_data_loader = Dis_dataloader(BATCH_SIZE,SEQ_LENGTH)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, learning_rate=GENERATOR_LR)
    #target_params = pickle.load(open('save/target_params_py3.pkl','rb'))
    #target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model
    target = TabularSimple(4,4,2)
    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim, 
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)
    saver = tf.train.Saver()

    
    rollout = ROLLOUT(generator, ROLLOUT_UPDATE_RATE)
    gan_trainer = GanTrainer(generator,discriminator, rollout, 
        gen_data_loader, dis_data_loader, target,pretrain_file, advtrain_file, 
        positive_file, negative_file, BATCH_SIZE)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)

    
    if pretrain == Runmode.fresh or pretrain == Runmode.skip:
        sess.run(tf.global_variables_initializer())
    if pretrain == Runmode.con:
        tf.reset_default_graph()
        saver.restore(sess, pretrain_file)
    if pretrain != Runmode.skip:
        sess.run(tf.global_variables_initializer())
        gan_trainer.pretrain(sess, PRE_GEN_EPOCH, PRE_DIS_EPOCH,DIS_EPOCHS_PR_BATCH,
            saver,dis_dropout_keep_prob,generated_num)
        # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
        # generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, positive_file)
    if advtrain == Runmode.con:
        tf.reset_default_graph()
        saver.restore(sess, advtrain_file)
    if advtrain != Runmode.skip:
        gan_trainer.advtrain(sess,saver,TOTAL_BATCH,BATCH_SIZE,epochs_generator,epochs_discriminator,
            DIS_EPOCHS_PR_BATCH,rollout_num,generated_num, dis_dropout_keep_prob)
    gan_trainer.log.close()

if __name__ == '__main__':
    main()
