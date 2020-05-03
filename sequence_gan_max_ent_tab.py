import numpy as np
import git
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import random
import time
from dataloader import *
from generator import Generator, MusicGenerator
from discriminator import Discriminator
from rollout_max_ent import ROLLOUT
from target_lstm import TARGET_LSTM
from tabular_simple import TabularSimple
from gan_trainer import GanTrainer
import pickle
from enum import Enum
import yaml
import shutil
import datetime
from tensorflow.python import debug as tf_debug
import argparse
from shutil import copyfile

class Runmode(Enum):
    fresh = 1 #run from scratch
    con = 2 # continue pretraining from cache 
    skip = 3 #  skip the pretraining step
    cache = 4 # load pretraining cache and skip pretraining





def main():
    parser=argparse.ArgumentParser(
    description='''Run's the SeqGan algorithm using Maximum Entropy Reinforcement Learning.''',
    epilog="""All's well that ends well.""")
    parser.add_argument('--config', type=str, default="", help='JSON config file',required=True)
    args = parser.parse_args()
    config_file = args.config
    with open(config_file) as stream:
        try:
            config = yaml.load(stream, Loader=yaml.Loader)
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
    
    
    repo = git.Repo(search_parent_directories=True)
    model_number = repo.head.object.hexsha
    if repo.is_dirty():
        print('Git repo is dirty which will result in poor tracking-logging')
        model_number += '-dirty'
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join('runs', model_number, current_time)
    save_dir = os.path.join(run_dir,'save')
    os.makedirs(save_dir,exist_ok=True)
    copyfile(src = config_file, dst = os.path.join(run_dir,'config.yaml'))
    
    # positive data, containing real music sequences
    positive_file = config['positive_file']
    #positive_file = os.path.join(save_dir, 'real_data.txt')
    # negative data from the generator, containing fake sequences
    #negative_file = config['negative_file']
    negative_file = os.path.join(save_dir, 'generator_sample.txt')
    valid_file = config['valid_file']
    #valid_file = os.path.join(save_dir, 'eval_file.txt')
    generated_num = config['generated_num']

    epochs_generator = config['epochs_generator']
    epochs_discriminator = config['epochs_discriminator']
    pretrain = Runmode[config['runmode_pretrain']] #Enum('Runmode', config['runmode_pretrain'], module=__name__)
    advtrain = Runmode[config['runmode_advtrain']]
    pretrain_cache_file = config['pretrain_file']
    advtrain_cache_file = config['advtrain_file']
    ent_temp = config['ent_temp']
    music = config['is_music_data']
    target_class = config['target_class'] if 'target_class' in config else False 
    number_model_save = int(config['number_model_save'])
    normalize_rewards = config['rewards_normalize']
    
    
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0
    tf.compat.v1.disable_eager_execution()
    if music:
        gen_data_loader = Music_Gen_Data_loader(BATCH_SIZE,SEQ_LENGTH)
        eval_data_loader = Music_Gen_Data_loader(BATCH_SIZE,SEQ_LENGTH)
        dis_data_loader = Music_Dis_dataloader(BATCH_SIZE,SEQ_LENGTH)
        generator = MusicGenerator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, learning_rate=GENERATOR_LR)
    
    else:
        gen_data_loader = Gen_Data_loader(BATCH_SIZE,SEQ_LENGTH)
        eval_data_loader = Gen_Data_loader(BATCH_SIZE,SEQ_LENGTH) # For testing
        dis_data_loader = Dis_dataloader(BATCH_SIZE,SEQ_LENGTH)
        generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, learning_rate=GENERATOR_LR)
    
    # generate real data from the true dataset
    gen_data_loader.create_batches(positive_file)
    # generate real validation data from true validation dataset
    eval_data_loader.create_batches(valid_file)

    #target_params = pickle.load(open('save/target_params_py3.pkl','rb'))
    #target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model
    if target_class == 'TARGET_LSTM':
        target_params = pickle.load(open(config['target_args'],'rb'))
        target = TARGET_LSTM(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model
    elif target_class:
        target = globals()[target_class](*config['target_args'])
        
    else:
        target = None 
    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim, 
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)
    
    
    rollout = ROLLOUT(generator, ROLLOUT_UPDATE_RATE, normalize_rewards)
    print("Logging in: {}".format(run_dir))
    gan_trainer = GanTrainer(generator,discriminator, rollout, 
        gen_data_loader, dis_data_loader, eval_data_loader, target, 
        positive_file, negative_file, BATCH_SIZE,START_TOKEN, music, number_model_save, run_dir)
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    saver = tf.compat.v1.train.Saver(max_to_keep=number_model_save)
    sess = tf.compat.v1.Session(config=tf_config)
    gan_trainer.init(sess)
    # Run from scratch
    if pretrain == Runmode.fresh or pretrain == Runmode.skip:
        sess.run(tf.compat.v1.global_variables_initializer())
    # Load cache
    elif pretrain == Runmode.con or pretrain == Runmode.cache:
        #tf.compat.v1.reset_default_graph()
        saver.restore(sess, pretrain_cache_file)
        #gan_trainer.init(sess)
    # Run the pretraining
    if pretrain != Runmode.skip or pretrain != Runmode.cache:
        gan_trainer.pretrain(sess, PRE_GEN_EPOCH, PRE_DIS_EPOCH,DIS_EPOCHS_PR_BATCH,
            saver,dis_dropout_keep_prob,generated_num)
        # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
        # generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, positive_file)
    if advtrain == Runmode.con:
        tf.compat.v1.reset_default_graph()
        saver.restore(sess, advtrain_cache_file)
        gan_trainer.init(sess)
    if advtrain != Runmode.skip:
        gan_trainer.advtrain(sess, saver, TOTAL_BATCH, BATCH_SIZE, epochs_generator, epochs_discriminator,
            DIS_EPOCHS_PR_BATCH, rollout_num,generated_num, dis_dropout_keep_prob, ent_temp)
    #gan_trainer.log.close()

if __name__ == '__main__':
    main()
