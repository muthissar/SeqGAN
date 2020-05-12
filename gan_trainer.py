import numpy as np
#from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pathos.multiprocessing import ProcessingPool as Pool
import postprocessing as POST
import math
import os
import time



class GanTrainer:
    def __init__(self,generator,discriminator, rollout, 
        gen_data_loader, dis_data_loader, eval_data_loader, target, 
        positive_file, negative_file, BATCH_SIZE, START_TOKEN, music, save, run_dir,log_time=True):
        self.generator = generator
        self.discriminator = discriminator
        self.rollout = rollout
        self.gen_data_loader = gen_data_loader
        self.dis_data_loader = dis_data_loader
        self.eval_data_loader = eval_data_loader
        self.target = target
        self.run_dir = run_dir
        self.define_log()
        model_dir = os.path.join(run_dir, 'model')
        os.makedirs(model_dir, exist_ok = True)
        self.gen_dir = os.path.join(run_dir, 'gen')
        self.pretrain_file = os.path.join(model_dir,'pretrain')
        self.advtrain_file = os.path.join(model_dir,'advtrain')
        self.positive_file = positive_file
        self.negative_file = negative_file
        self.BATCH_SIZE = BATCH_SIZE
        self.START_TOKEN = START_TOKEN
        self.music = music
        self.save = save
        self.log_time = True
    def define_log(self):
        writer_dir = os.path.join(self.run_dir, 'log')
        os.makedirs(writer_dir, exist_ok = True)
        g =  tf.compat.v1.get_default_graph()
        with g.as_default():
            with tf.compat.v1.variable_scope('gan_trainer'):
                self._step_gen = tf.Variable(0, dtype=tf.int64)
                self._step_disc = tf.Variable(0, dtype=tf.int64)
                self._cross_p_q = tf.Variable(0, dtype=tf.float64)
                self._cross_p_q_2 = tf.Variable(0, dtype=tf.float64)
                self._cross_q_p = tf.Variable(0, dtype=tf.float64)
                self.ent_p = tf.Variable(0, dtype=tf.float64)
                self.rein_max_ent = tf.Variable(0, dtype=tf.float64)
                self.disc_loss = tf.Variable(0, dtype=tf.float64)
                self.writer =  tf.summary.create_file_writer(writer_dir)
                with self.writer.as_default():
                    #pretraining logs
                    #tf.summary.scalar('Loss/pre/cross_p_q', self._cross_p_q_pre, self._step_gen)
                    #tf.summary.scalar('Loss/pre/cross_p_q_2', self._cross_p_q_2_pre, self._step_gen)
                    #tf.summary.scalar('Loss/pre/cross_q_p', self._cross_q_p_pre, self._step_gen)
                    #tf.summary.scalar('Loss/pre/ent_p', self.ent_p_pre, self._step_gen)
                    #tf.summary.scalar('Loss/pre/discrim_loss', self.disc_loss_pre, self._step_disc)
                    #adv trainging logs
                    tf.summary.scalar('Loss/cross_p_q', self._cross_p_q, self._step_gen)
                    tf.summary.scalar('Loss/cross_p_q_2', self._cross_p_q_2, self._step_gen)
                    tf.summary.scalar('Loss/cross_q_p', self._cross_q_p, self._step_gen)
                    tf.summary.scalar('Loss/ent_p', self.ent_p, self._step_gen)
                    tf.summary.scalar('Loss/discrim_loss', self.disc_loss, self._step_disc)
                    tf.summary.scalar('Loss/rein_max_ent', self.rein_max_ent, self._step_gen)
                    self.all_summary_ops = tf.compat.v1.summary.all_v2_summary_ops()
                    self.writer_flush = self.writer.flush()
    def init(self,sess):
        sess.run(self.writer.init())
        
    def pre_train_epoch(self, sess, trainable_model, data_loader):
        # Pre-train the generator using MLE for one epoch
        supervised_g_losses = []
        data_loader.reset_pointer()

        for _ in range(data_loader.num_batch):
            batch = data_loader.next_batch()
            _, g_loss = trainable_model.pretrain_step(sess, batch)
            supervised_g_losses.append(g_loss)

        return np.mean(supervised_g_losses)
    def calculate_bleu(self,sess, trainable_model, data_loader):
        #TODO: ONLY GENERATE A FEW SAMPLES IN HYPOYHESIS
        # bleu score implementation
        # used for performance evaluation for pre-training & adv. training
        # separate true dataset to the valid set
        # conditionally generate samples from the start token of the valid set
        # measure similarity with nltk corpus BLEU
        smoother = SmoothingFunction()

        data_loader.reset_pointer()
        bleu_avg = 0

        references = []
        hypotheses = []

        for it in range(data_loader.num_batch):
            batch = data_loader.next_batch()
            # predict from the batch
            # TODO: which start tokens?
            #start_tokens = batch[:, 0]
            start_tokens = np.array([self.START_TOKEN] * self.BATCH_SIZE, dtype=np.float64)
            prediction = trainable_model.predict(sess, batch, start_tokens)
            # argmax to convert to vocab
            #prediction = np.argmax(prediction, axis=2)

            # cast batch and prediction to 2d list of strings
            batch_list = batch.astype(np.str).tolist()
            pred_list = prediction.astype(np.str).tolist()
            references.extend(batch_list)
            hypotheses.extend(pred_list)

        bleu = 0.
        # calculate bleu for each predicted seq
        # compare each predicted seq with the entire references
        # this is slow, use multiprocess
        def calc_sentence_bleu(hypothesis):
            return sentence_bleu(references, hypothesis, smoothing_function=smoother.method4)
        #if __name__ == '__main__':
        p = Pool()
        result = (p.map(calc_sentence_bleu, hypotheses))
        bleu = np.mean(result)

        return bleu
    
    def cross_p_q(self, sess):
        #samples = self.generator.generate(sess)
        #test_loss = - self.target.ll(samples=samples, sess=sess) if self.target is not None else math.nan
        #NOTE: in order for test loss to be stable we need a lot of samples. This should vary on model
        #TODO: make this a variable
        n_samples = 2**14
        #samples = np.concatenate([self.target.generate(sess=sess,batch_size = self.BATCH_SIZE) for  _ in range(max(1,int(n_samples/self.BATCH_SIZE)))])
        test_loss = np.mean([sess.run(self.generator.pretrain_loss, {self.generator.x: \
            self.target.generate(sess=sess,batch_size = self.BATCH_SIZE)}) if\
            self.target is not None else math.nan for  _ in range(int(math.ceil(n_samples/self.BATCH_SIZE)))])
        return test_loss

    def cross_p_q_2(self, sess):
        supervised_g_losses = []
        for _ in range(self.eval_data_loader.num_batch):
            batch = self.eval_data_loader.next_batch()
            g_loss = sess.run(self.generator.pretrain_loss,
                {self.generator.x: batch}
            )
            supervised_g_losses.append(g_loss)
        return np.mean(supervised_g_losses)
    def cross_q_p(self, sess):
        #samples = self.generator.generate(sess)
        #test_loss = - self.target.ll(samples=samples, sess=sess) if self.target is not None else math.nan
        #NOTE: in order for test loss to be stable we need a lot of samples. This should vary on model
        #TODO: make this a variable
        n_samples = 2**14
        #samples = np.concatenate([self.target.generate(sess=sess,batch_size = self.BATCH_SIZE) for  _ in range(max(1,int(n_samples/self.BATCH_SIZE)))])
        test_loss = np.mean([-self.target.ll(samples=self.generator.generate(sess), sess=sess) if\
            self.target is not None else math.nan for  _ in range(int(math.ceil(n_samples/self.BATCH_SIZE)))])
        return test_loss
    
    def log_gen(self, sess, epoch, cross_p_q_2,g_loss=0):
        cross_p_q = self.cross_p_q(sess)
        cross_q_p = self.cross_q_p(sess)
        ent_p = sess.run(self.generator.pretrain_loss,
            {self.generator.x: self.generator.generate(sess)})
        sess.run(self.all_summary_ops)
        #sess.run([self._step_gen.assign(epoch)])
        sess.run([self._step_gen.assign_add(1)])
        sess.run([self._cross_p_q.assign(cross_p_q),
            self._cross_p_q_2.assign(cross_p_q_2),
            self._cross_q_p.assign(cross_q_p),
            self.ent_p.assign(ent_p),
            self.rein_max_ent.assign(g_loss)
        ])
        sess.run(self.writer_flush)
            
        
            
        #sess.run(self.writer_flush)
        # measure bleu score with the validation set
        #bleu_score = self.calculate_bleu(sess, self.generator, self.eval_data_loader)
        #if epoch % 5 == 0:
        print('epoch: {}, cross(p,q): {}, cross_2(p,q): {}, cross(q,p): {}, ent(p): {}, g_loss: {}'.format(
            epoch,
            cross_p_q,
            cross_p_q_2,
            cross_q_p,
            ent_p,
            g_loss
            ))

    def log_disc(self, sess, epoch, disc_loss):
        sess.run(self.all_summary_ops)
        sess.run([self._step_disc.assign_add(1)])
        sess.run([self.disc_loss.assign(disc_loss)])
        sess.run(self.writer_flush)
        print("epoch: {}, discrim_loss: {}.".format(
            epoch,
            disc_loss
        ))
        #class_ = 0
        #predictions = np.array([])
        #TODO: not right loss as it's not taking into account the 
        #for i in range(10):
        #    predictions = np.concatenate((predictions,sess.run(self.discriminator.ypred_for_auc, {self.discriminator.input_x: self.generator.generate(sess), self.discriminator.dropout_keep_prob: dis_dropout_keep_prob})[:,class_]))
        #tf.summary.scalar('Loss/discrim_loss', disc_loss, total_batch)
        #print("discrim  --  min: {}, max: {}, ll: {}, loss: {}".format(min(predictions),max(predictions),,disc_loss))

    def pretrain_generator(self,sess,PRE_GEN_EPOCH,saver,generated_num):
        self.gen_data_loader.create_batches(self.positive_file)
        for epoch in range(PRE_GEN_EPOCH):
            cross_p_q_2 = self.pre_train_epoch(sess, self.generator, self.gen_data_loader)
            self.log_gen(sess, epoch, cross_p_q_2)
            if epoch % 5 == 0:
                if self.save:
                    saver.save(sess, self.pretrain_file)
                # generate 5 test samples per epoch
                # it automatically samples from the generator and postprocess to midi file
                # midi files are saved to the pre-defined folder
                #if self.music:
                #    if epoch == 0:
                #        self.generator.generate_samples(sess, self.BATCH_SIZE, generated_num, self.negative_file)
                #        #POST.main(self.negative_file, 5,  epoch, self.gen_dir + +str(-1)+'_vanilla_')
                #        #POST.main(self.negative_file, 5,  epoch, self.gen_dir)
                #    elif epoch == PRE_GEN_EPOCH - 1:
                #        self.generator.generate_samples(sess, self.BATCH_SIZE, generated_num, self.negative_file)
                #        #POST.main(self.negative_file, 5, epoch, self.gen_dir + str(-PRE_GEN_EPOCH)+'_vanilla_')
                #        #POST.main(self.negative_file, 5, epoch, self.gen_dir)


    def pretrain_discrim(self, sess, PRE_DIS_EPOCH,DIS_EPOCHS_PR_BATCH,
        saver,dis_dropout_keep_prob, generated_num):
        print('Start pre-training discriminator...')
        # Train 3 epoch on the generated data and do this for 50 times
        for epoch in range(PRE_DIS_EPOCH):
            self.generator.generate_samples(sess, self.BATCH_SIZE, generated_num, self.negative_file)
            self.dis_data_loader.load_train_data(self.positive_file, self.negative_file)
            for i in range(DIS_EPOCHS_PR_BATCH):
                self.dis_data_loader.reset_pointer()
                for it in range(self.dis_data_loader.num_batch):
                    x_batch, y_batch = self.dis_data_loader.next_batch()
                    feed = {
                        self.discriminator.input_x: x_batch,
                        self.discriminator.input_y: y_batch,
                        self.discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _ = sess.run(self.discriminator.train_op, feed)
            if self.save:
                saver.save(sess, self.pretrain_file)
            self.log_disc(sess, epoch, sess.run(self.discriminator.loss, feed))

    def pretrain(self,sess, PRE_GEN_EPOCH, PRE_DIS_EPOCH,DIS_EPOCHS_PR_BATCH,
        saver,dis_dropout_keep_prob,generated_num):
        #  pre-train generator
        print('Start pre-training...')
        #self.log.write('pre-training...\n')
        self.pretrain_generator(sess,PRE_GEN_EPOCH,saver,generated_num)
        self.pretrain_discrim(sess, PRE_DIS_EPOCH,DIS_EPOCHS_PR_BATCH,
        saver,dis_dropout_keep_prob,generated_num)
        self.gen_data_loader.create_batches(self.positive_file)
    def advtrain(self,sess,saver,TOTAL_BATCH,BATCH_SIZE,epochs_generator,epochs_discriminator,
        DIS_EPOCHS_PR_BATCH,rollout_num,generated_num, dis_dropout_keep_prob,ent_temp):
        print('#########################################################################')
        print('Start Adversarial Training...')
        #self.log.write('adversarial training...\n')
        for total_batch in range(TOTAL_BATCH):
            # Train the generator for one step
            t1  = time.time()
            g_loss = self.advtrain_gen(sess, epochs_generator, rollout_num, ent_temp)
            t2  = time.time()
            print('Time g_train: {}'.format(t2-t1))
            t2 = t1
            # Test
            #if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            t1  = time.time()
            self.log_gen(sess, total_batch, self.cross_p_q_2(sess), g_loss)
            t2  = time.time()
            print('Time log_gen: {}'.format(t2-t1))
            t2 = t1
            t1  = time.time()
            disc_loss = self.advtrain_disc(sess,saver,epochs_discriminator,DIS_EPOCHS_PR_BATCH,
                BATCH_SIZE, generated_num, self.positive_file, self.negative_file, dis_dropout_keep_prob)
            t2  = time.time()
            print('Time d_train: {}'.format(t2-t1))
            t2 = t1
            t1  = time.time()
            self.log_disc(sess, total_batch, disc_loss)
            t2  = time.time()
            print('Time log_disc: {}'.format(t2-t1))
            t2 = t1
            #if True: #config['infinite_loop']:
            #    if bleu_score < 0.5: #config['loop_threshold']:
            #        buffer = 'Mode collapse detected, restarting from pretrained model...'
            #        print(buffer)
            #        self.log.write(buffer + '\n')
            #        self.log.flush()
            #        #load_checkpoint(sess, saver)
            #if self.music:
                # generate random test samples and postprocess the sequence to midi file
                #self.generator.generate_samples(sess, BATCH_SIZE, generated_num, self.negative_file)
                #POST.main(self.negative_file, 5, total_batch, self.gen_dir + str(total_batch)+'_vanilla_')
                #POST.main(self.negative_file, 5, total_batch, self.gen_dir)
                
                #print("disc_loss: {}".format(disc_loss))
                #print("trained disc --- %s seconds ---" % (time.time() - start_time))



    def advtrain_gen(self,sess,epochs_generator,rollout_num,ent_temp):
        # Train the generator for one step
        if self.rewards_reduced_variance:
            for it in range(epochs_generator):
                samples = self.generator.generate(sess)
                rewards = self.rollout.get_reward(sess, samples, rollout_num, self.discriminator, ent_temp=ent_temp)
                feed = {self.generator.x: samples, self.generator.rewards: rewards}
                _ = sess.run(self.generator.g_updates, feed_dict=feed)
            # TODO: BU NE??? THIS TRAINS GENERATOR/LSTM. WHY???
            # Update roll-out parameters
            self.rollout.update_params()
        else:
            for it in range(epochs_generator):
                samples = self.generator.generate(sess)
                rewards = self.rollout.get_reward_2(sess, samples, rollout_num, self.discriminator, ent_temp=ent_temp)
                feed = {self.generator.x: samples, self.generator.rewards: rewards}
                _ = sess.run(self.generator.g_updates, feed_dict=feed)
        #test_loss = - self.target.ll(samples=samples,sess=sess) if self.target is not None else math.nan
        #print("rollout --- %s seconds ---" % (time.time() - start_time))
        # Train the discriminator
        g_loss = sess.run(self.generator.g_loss, feed_dict=feed)
        return g_loss

    def advtrain_disc(self,sess,saver,epochs_discriminator,DIS_EPOCHS_PR_BATCH,
        BATCH_SIZE, generated_num, positive_file, negative_file, dis_dropout_keep_prob):
        for _ in range(epochs_discriminator):
            self.generator.generate_samples(sess, BATCH_SIZE, generated_num, negative_file)
            self.dis_data_loader.load_train_data(positive_file, negative_file)
            for _ in range(DIS_EPOCHS_PR_BATCH):
                self.dis_data_loader.reset_pointer()
                for it in range(self.dis_data_loader.num_batch):
                    x_batch, y_batch = self.dis_data_loader.next_batch()
                    feed = {
                        self.discriminator.input_x: x_batch,
                        self.discriminator.input_y: y_batch,
                        self.discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _ = sess.run(self.discriminator.train_op, feed)
        if self.save:
            saver.save(sess, self.advtrain_file)
        disc_loss = sess.run(self.discriminator.loss, feed)
        return disc_loss

