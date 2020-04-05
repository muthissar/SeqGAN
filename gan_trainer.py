import numpy as np
from torch.utils.tensorboard import SummaryWriter
class GanTrainer:
    def __init__(self,generator,discriminator, rollout, 
        gen_data_loader, dis_data_loader, target,pretrain_file, advtrain_file, 
        positive_file, negative_file, BATCH_SIZE):
        self.generator = generator
        self.discriminator = discriminator
        self.rollout = rollout
        self.log = open('save/experiment-log.txt', 'w')
        self.gen_data_loader = gen_data_loader
        self.dis_data_loader = dis_data_loader
        self.target = target
        self.writer =  SummaryWriter()
        self.pretrain_file = pretrain_file
        self.advtrain_file = advtrain_file
        self.positive_file = positive_file
        self.negative_file = negative_file
        self.BATCH_SIZE = BATCH_SIZE
    def pre_train_epoch(self, sess, trainable_model, data_loader):
        # Pre-train the generator using MLE for one epoch
        supervised_g_losses = []
        data_loader.reset_pointer()

        for _ in range(data_loader.num_batch):
            batch = data_loader.next_batch()
            _, g_loss = trainable_model.pretrain_step(sess, batch)
            supervised_g_losses.append(g_loss)

        return np.mean(supervised_g_losses)
    def generate_samples(self,sess, trainable_model, batch_size, generated_num, output_file):
        # Generate Samples
        generated_samples = []
        for _ in range(int(generated_num / batch_size)):
            generated_samples.extend(trainable_model.generate(sess))

        with open(output_file, 'w') as fout:
            for poem in generated_samples:
                buffer = ' '.join([str(x) for x in poem]) + '\n'
                fout.write(buffer)

    def pretrain_generator(self,sess,PRE_GEN_EPOCH,saver):
        self.gen_data_loader.create_batches(self.positive_file)
        for epoch in range(PRE_GEN_EPOCH):
            loss = self.pre_train_epoch(sess, self.generator, self.gen_data_loader)
            if epoch % 5 == 0:
                #generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
                samples = self.generator.generate(sess)
                test_loss = - self.target.ll(samples)
                #likelihood_data_loader.create_batches(eval_file)
                #test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
                print('pre-train epoch ', epoch, 'test_loss ', test_loss)
                buffer = 'epoch:\t'+ str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
                self.writer.add_scalar('Loss/pre_oracle_nll', test_loss, epoch)
                self.log.write(buffer)
                saver.save(sess, self.pretrain_file)

    def pretrain_discrim(self, sess, PRE_DIS_EPOCH,DIS_EPOCHS_PR_BATCH,
        saver,dis_dropout_keep_prob, generated_num):
        print('Start pre-training discriminator...')
        # Train 3 epoch on the generated data and do this for 50 times
        for epoch in range(PRE_DIS_EPOCH):
            self.generate_samples(sess, self.generator, self.BATCH_SIZE, generated_num, self.negative_file)
            self.dis_data_loader.load_train_data(self.positive_file, self.negative_file,generated_num)
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
            saver.save(sess, self.pretrain_file)
            self.writer.add_scalar('Loss/pre_discrim_loss', sess.run(self.discriminator.loss, feed), epoch)

    def pretrain(self,sess, PRE_GEN_EPOCH, PRE_DIS_EPOCH,DIS_EPOCHS_PR_BATCH,
        saver,dis_dropout_keep_prob,generated_num):
        #  pre-train generator
        print('Start pre-training...')
        self.log.write('pre-training...\n')
        self.pretrain_generator(sess,PRE_GEN_EPOCH,saver)
        self.pretrain_discrim(sess, PRE_DIS_EPOCH,DIS_EPOCHS_PR_BATCH,
        saver,dis_dropout_keep_prob,generated_num)
        self.gen_data_loader.create_batches(self.positive_file)
        print('pre-train entropy: ', sess.run(self.generator.pretrain_loss,
            {self.generator.x: self.generator.generate(sess)}))
    def advtrain(self,sess,saver,TOTAL_BATCH,BATCH_SIZE,epochs_generator,epochs_discriminator,
        DIS_EPOCHS_PR_BATCH,rollout_num,generated_num, dis_dropout_keep_prob):
        print('#########################################################################')
        print('Start Adversarial Training...')
        self.log.write('adversarial training...\n')
        for total_batch in range(TOTAL_BATCH):
            # Train the generator for one step
            test_loss , g_loss = self.advtrain_gen(sess,epochs_generator,rollout_num)
            # Test
            if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
                #generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
                #likelihood_data_loader.create_batches(eval_file)
                #test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
                buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
                policy_entropy = sess.run(self.generator.pretrain_loss, {
                    self.generator.x: self.generator.generate(sess)})
                print('total_batch: ', total_batch, 'test_loss: ', test_loss, 
                    'g_loss: ', g_loss ,' policy entropy: ',policy_entropy)
                self.writer.add_scalar('Loss/oracle_nll', test_loss, total_batch)
                self.log.write(buffer)
            disc_loss = self.advtrain_disc(sess,saver,epochs_discriminator,DIS_EPOCHS_PR_BATCH,
                BATCH_SIZE, generated_num, self.positive_file, self.negative_file, dis_dropout_keep_prob)
            class_ = 0
            predictions = np.array([])
            for i in range(10):
                predictions = np.concatenate((predictions,sess.run(self.discriminator.ypred_for_auc, {self.discriminator.input_x: self.generator.generate(sess), self.discriminator.dropout_keep_prob: dis_dropout_keep_prob})[:,class_]))
            self.writer.add_scalar('Loss/discrim_loss', disc_loss, total_batch)
            print("discrim  --  min: {}, max: {}, max mean: {}, loss: {}".format(min(predictions),max(predictions),np.mean(predictions),disc_loss))
            #print("disc_loss: {}".format(disc_loss))
            #print("trained disc --- %s seconds ---" % (time.time() - start_time))



    def advtrain_gen(self,sess,epochs_generator,rollout_num):
        # Train the generator for one step
            
        for it in range(epochs_generator):
            samples = self.generator.generate(sess)
            rewards = self.rollout.get_reward(sess, samples, rollout_num, self.discriminator)
            feed = {self.generator.x: samples, self.generator.rewards: rewards}
            _ = sess.run(self.generator.g_updates, feed_dict=feed)
        

        #print("trained adv --- %s seconds ---" % (time.time() - start_time))
        # TODO: BU NE??? THIS TRAINS GENERATOR/LSTM. WHY???
        # Update roll-out parameters
        self.rollout.update_params()
        if self.target:
            test_loss = - self.target.ll(samples)
        #print("rollout --- %s seconds ---" % (time.time() - start_time))
        # Train the discriminator
        g_loss = sess.run(self.generator.g_loss, feed_dict=feed)
        return test_loss , g_loss

    def advtrain_disc(self,sess,saver,epochs_discriminator,DIS_EPOCHS_PR_BATCH,
        BATCH_SIZE, generated_num,positive_file,negative_file,dis_dropout_keep_prob):
        for _ in range(epochs_discriminator):
            self.generate_samples(sess, self.generator, BATCH_SIZE, generated_num, negative_file)
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
        saver.save(sess, self.advtrain_file)
        disc_loss = sess.run(self.discriminator.loss, feed)
        return disc_loss