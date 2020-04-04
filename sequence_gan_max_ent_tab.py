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
import pickle
from torch.utils.tensorboard import SummaryWriter
from enum import Enum

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 4 # embedding dimension
HIDDEN_DIM = 4 # hidden state dimension of lstm cell
SEQ_LENGTH = 4 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 120 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64
vocab_size = 4

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 4
dis_filter_sizes = [1, 2, 3, 4]
#dis_num_filters = [200, 200, 200, 200]
dis_num_filters = [10, 10, 10, 10]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 200
positive_file = 'save/real_data_tab.txt'
negative_file = 'save/generator_sample_tab.txt'
eval_file = 'save/eval_file_tab.txt'
generated_num = 10000


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


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


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    gen_data_loader = Gen_Data_loader(BATCH_SIZE,SEQ_LENGTH)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE,SEQ_LENGTH) # For testing
    dis_data_loader = Dis_dataloader(BATCH_SIZE,SEQ_LENGTH)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, learning_rate=0.0025)
    #target_params = pickle.load(open('save/target_params_py3.pkl','rb'))
    #target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model
    target = TabularSimple(4,4,2)
    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim, 
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    class Runmode(Enum):
        fresh = 1
        con = 2
        skip = 3

    pretrain = Runmode.fresh
    pretrain_file = 'model/pretrain_max_ent_tab.ckpt'
    advtrain = Runmode.fresh 
    advtrain_file = 'model/advtrain_max_ent_tab.ckpt'
    saver = tf.train.Saver()

    log = open('save/experiment-log.txt', 'w')
    writer = SummaryWriter()
    if pretrain == Runmode.fresh or pretrain == Runmode.skip:
        sess.run(tf.global_variables_initializer())
    if pretrain == Runmode.con:
        tf.reset_default_graph()
        saver.restore(sess, pretrain_file)
    if pretrain != Runmode.skip:
        sess.run(tf.global_variables_initializer())
        # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
        # generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, positive_file)
        gen_data_loader.create_batches(positive_file)

        #test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        
        
        #  pre-train generator
        print('Start pre-training...')
        log.write('pre-training...\n')
        for epoch in range(PRE_EPOCH_NUM):
            loss = pre_train_epoch(sess, generator, gen_data_loader)
            if epoch % 5 == 0:
                #generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
                samples = generator.generate(sess)
                test_loss = - target.ll(samples)
                #likelihood_data_loader.create_batches(eval_file)
                #test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
                print('pre-train epoch ', epoch, 'test_loss ', test_loss)
                buffer = 'epoch:\t'+ str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
                writer.add_scalar('Loss/pre_oracle_nll', test_loss, epoch)
                log.write(buffer)
                saver.save(sess, pretrain_file)

        print('Start pre-training discriminator...')
        # Train 3 epoch on the generated data and do this for 50 times
        for epoch in range(50):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)
            for i in range(3):
                dis_data_loader.reset_pointer()
                for it in range(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _ = sess.run(discriminator.train_op, feed)
            saver.save(sess, pretrain_file)
            writer.add_scalar('Loss/pre_discrim_loss', sess.run(discriminator.loss, feed), epoch)
        print('pre-train entropy: ', sess.run(generator.pretrain_loss, {generator.x: generator.generate(sess)}))
    if advtrain == Runmode.con:
        tf.reset_default_graph()
        saver.restore(sess, advtrain_file)
    if advtrain != Runmode.skip:
        rollout = ROLLOUT(generator, 0.8)
        start_time = time.time()
        print('#########################################################################')
        print('Start Adversarial Training...')
        log.write('adversarial training...\n')
        for total_batch in range(TOTAL_BATCH):
            # Train the generator for one step
            
            for it in range(1):
                samples = generator.generate(sess)
                rewards = rollout.get_reward(sess, samples, 16, discriminator)
                feed = {generator.x: samples, generator.rewards: rewards}
                _ = sess.run(generator.g_updates, feed_dict=feed)
            

            # Test
            if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
                test_loss = - target.ll(samples)
                #generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
                #likelihood_data_loader.create_batches(eval_file)
                #test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
                buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
                policy_entropy = sess.run(generator.pretrain_loss, {generator.x: generator.generate(sess)})
                print('total_batch: ', total_batch, 'test_loss: ', test_loss, ' policy entropy: ',policy_entropy)
                writer.add_scalar('Loss/oracle_nll', test_loss, total_batch)
                log.write(buffer)

            #print("trained adv --- %s seconds ---" % (time.time() - start_time))

            # Update roll-out parameters
            rollout.update_params()
            #print("rollout --- %s seconds ---" % (time.time() - start_time))
            # Train the discriminator
            for _ in range(5):
                generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
                dis_data_loader.load_train_data(positive_file, negative_file)

                for _ in range(3):
                    dis_data_loader.reset_pointer()
                    for it in range(dis_data_loader.num_batch):
                        x_batch, y_batch = dis_data_loader.next_batch()
                        feed = {
                            discriminator.input_x: x_batch,
                            discriminator.input_y: y_batch,
                            discriminator.dropout_keep_prob: dis_dropout_keep_prob
                        }
                        _ = sess.run(discriminator.train_op, feed)
            saver.save(sess, advtrain_file)
            disc_loss = sess.run(discriminator.loss, feed)
            class_ = 0
            predictions = np.array([])
            for i in range(10):
                predictions = np.concatenate((predictions,sess.run(discriminator.ypred_for_auc, {discriminator.input_x: generator.generate(sess), discriminator.dropout_keep_prob: 1.0})[:,class_]))
            writer.add_scalar('Loss/discrim_loss', disc_loss, total_batch)
            print("discrim  --  min: {}, max: {}, max mean: {}, loss: {}".format(min(predictions),max(predictions),np.mean(predictions),disc_loss))
            #print("disc_loss: {}".format(disc_loss))
            #print("trained disc --- %s seconds ---" % (time.time() - start_time))

    log.close()


if __name__ == '__main__':
    main()
