from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from ops import *
from utils import *

""" Model Defination"""

class DualNet(object):
    def __init__(
        self,
        sess,
        image_size=256,
        batch_size=1,
        gcn=64,
        dcn=64,
        A_channels=3,
        B_channels=3,
        dataset_name="facades",
        checkpoint_dir=None,
        lambda_A=500.0,
        lambda_B=500.0,
        sample_dir=None,
        dropout_rate=0.0,
        loss_metric="L1",
        flip=False,
        n_critic=5,
        GAN_type="wgan-gp",
        clip_value=0.1,
        log_freq=50,
        disc_type="globalgan",
    ):
        self.dcn = dcn
        self.flip = flip
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B

        self.sess = sess
        self.is_grayscale_A = A_channels == 1
        self.is_grayscale_B = B_channels == 1
        self.batch_size = batch_size
        self.image_size = image_size
        self.gcn = gcn
        self.A_channels = A_channels
        self.B_channels = B_channels
        self.loss_metric = loss_metric

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir

        # directory name for output and logs saving
        self.dir_name = "%s-img_sz_%s-fltr_dim_%d-%s-lambda_AB_%s_%s" % (
            self.dataset_name,
            self.image_size,
            self.gcn,
            self.loss_metric,
            self.lambda_A,
            self.lambda_B,
        )
        self.dropout_rate = dropout_rate
        self.clip_value = clip_value
        self.GAN_type = GAN_type
        self.n_critic = n_critic
        self.log_freq = log_freq
        self.gamma = 10.0
        self.disc_type = disc_type
        self.build_model()

    def build_model(self):
        ### Define placeholders
        self.real_A = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.A_channels], name="real_A")
        self.real_B = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.B_channels], name="real_B")

        ### Define graphs for generators
        self.A2B = self.A_g_net(self.real_A, reuse=False)  # Generator A to B
        self.B2A = self.B_g_net(self.real_B, reuse=False)  # Generator B to A

        # Additional transformations for cycle consistency
        self.A2B2A = self.B_g_net(self.A2B, reuse=True)  # Cycle: A to B to A
        self.B2A2B = self.A_g_net(self.B2A, reuse=True)  # Cycle: B to A to B

        ### Define cycle consistency loss
        if self.loss_metric == "L1":
            self.cycle_A_loss = tf.reduce_mean(tf.abs(self.A2B2A - self.real_A))
            self.cycle_B_loss = tf.reduce_mean(tf.abs(self.B2A2B - self.real_B))
        elif self.loss_metric == "L2":
            self.cycle_A_loss = tf.reduce_mean(tf.square(self.A2B2A - self.real_A))
            self.cycle_B_loss = tf.reduce_mean(tf.square(self.B2A2B - self.real_B))

        ### Define adversarial losses for generators and discriminators
        self.Ad_logits_fake = self.A_d_net(self.A2B, reuse=False)
        self.Ad_logits_real = self.A_d_net(self.real_B, reuse=True)
        self.Bd_logits_fake = self.B_d_net(self.B2A, reuse=False)
        self.Bd_logits_real = self.B_d_net(self.real_A, reuse=True)

        # Define adversarial loss for A to B and B to A translations
        self.AB_loss = celoss(self.Ad_logits_fake, tf.ones_like(self.Ad_logits_fake))
        self.BA_loss = celoss(self.Bd_logits_fake, tf.ones_like(self.Bd_logits_fake))

        # Classic GAN loss or Wasserstein GAN loss (depending on self.GAN_type)
        if self.GAN_type == "classic":
            self.Ad_loss_real = celoss(self.Ad_logits_real, tf.ones_like(self.Ad_logits_real))
            self.Ad_loss_fake = celoss(self.Ad_logits_fake, tf.zeros_like(self.Ad_logits_fake))
            self.Bd_loss_real = celoss(self.Bd_logits_real, tf.ones_like(self.Bd_logits_real))
            self.Bd_loss_fake = celoss(self.Bd_logits_fake, tf.zeros_like(self.Bd_logits_fake))
        else:  # WGAN or WGAN-GP
            self.Ad_loss_real = -tf.reduce_mean(self.Ad_logits_real)
            self.Ad_loss_fake = tf.reduce_mean(self.Ad_logits_fake)
            self.Bd_loss_real = -tf.reduce_mean(self.Bd_logits_real)
            self.Bd_loss_fake = tf.reduce_mean(self.Bd_logits_fake)

        self.Ad_loss = self.Ad_loss_fake + self.Ad_loss_real  # Total discriminator loss for A
        self.Bd_loss = self.Bd_loss_fake + self.Bd_loss_real  # Total discriminator loss for B

        ### Update supervised and total generator losses (include supervised component)
        if self.GAN_type == "classic":
            self.Ag_loss = celoss(self.Ad_logits_fake, tf.ones_like(self.Ad_logits_fake)) + self.lambda_B * self.cycle_B_loss
            self.Bg_loss = celoss(self.Bd_logits_fake, tf.ones_like(self.Bd_logits_fake)) + self.lambda_A * self.cycle_A_loss
        else:
            self.Ag_loss = -tf.reduce_mean(self.Ad_logits_fake) + self.lambda_B * self.cycle_B_loss
            self.Bg_loss = -tf.reduce_mean(self.Bd_logits_fake) + self.lambda_A * self.cycle_A_loss

        ### Combine discriminator and generator losses
        self.d_loss = self.Ad_loss + self.Bd_loss
        self.g_loss = self.Ag_loss + self.Bg_loss

        ### WGAN-GP gradient penalty (if applicable)
        if self.GAN_type == "wgan-gp":
            # Gradient penalty for domain A
            epsilon_A = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
            interpolated_image_A = self.real_A + epsilon_A * (self.B2A - self.real_A)
            d_interpolated_A = self.B_d_net(interpolated_image_A, reuse=True)
            grad_d_interp_A = tf.gradients(d_interpolated_A, [interpolated_image_A])[0]
            slopes_A = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(grad_d_interp_A), axis=[1, 2, 3]))
            gradient_penalty_A = tf.reduce_mean((slopes_A - 1.0) ** 2)

            # Gradient penalty for domain B
            epsilon_B = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
            interpolated_image_B = self.real_B + epsilon_B * (self.A2B - self.real_B)
            d_interpolated_B = self.A_d_net(interpolated_image_B, reuse=True)
            grad_d_interp_B = tf.gradients(d_interpolated_B, [interpolated_image_B])[0]
            slopes_B = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(grad_d_interp_B), axis=[1, 2, 3]))
            gradient_penalty_B = tf.reduce_mean((slopes_B - 1.0) ** 2)

            self.d_loss += self.gamma * (gradient_penalty_A + gradient_penalty_B)  # Add gradient penalties to discriminator loss

        ### Define trainable variables
        t_vars = tf.trainable_variables()
        self.A_d_vars = [var for var in t_vars if "A_d_" in var.name]
        self.B_d_vars = [var for var in t_vars if "B_d_" in var.name]
        self.A_g_vars = [var for var in t_vars if "A_g_" in var.name]
        self.B_g_vars = [var for var in t_vars if "B_g_" in var.name]
        self.d_vars = self.A_d_vars + self.B_d_vars
        self.g_vars = self.A_g_vars + self.B_g_vars

        ### Initialize saver for saving and restoring models
        self.saver = tf.train.Saver()

    def load_random_samples(self):
        # np.random.choice(
        sample_files = np.random.choice(
            glob("./datasets/{}/val/A/*.*[g|G]".format(self.dataset_name)),
            self.batch_size,
        )
        sample_A_imgs = [
            load_data(f, image_size=self.image_size, flip=False) for f in sample_files
        ]

        sample_files = np.random.choice(
            glob("./datasets/{}/val/B/*.*[g|G]".format(self.dataset_name)),
            self.batch_size,
        )
        sample_B_imgs = [
            load_data(f, image_size=self.image_size, flip=False) for f in sample_files
        ]

        sample_A_imgs = np.reshape(
            np.array(sample_A_imgs).astype(np.float32),
            (self.batch_size, self.image_size, self.image_size, -1),
        )
        sample_B_imgs = np.reshape(
            np.array(sample_B_imgs).astype(np.float32),
            (self.batch_size, self.image_size, self.image_size, -1),
        )
        return sample_A_imgs, sample_B_imgs

    def sample_shotcut(self, sample_dir, epoch_idx, batch_idx):
        sample_A_imgs, sample_B_imgs = self.load_random_samples()

        # Run the session to get A to B translations and losses
        A2B_imgs, A2B2A_imgs = self.sess.run(
            [self.A2B, self.A2B2A],
            feed_dict={self.real_A: sample_A_imgs}
        )
        ABloss = self.sess.run(
            self.AB_loss,
            feed_dict={self.real_A: sample_A_imgs, self.real_B: sample_B_imgs},
        )

        # Run the session to get B to A translations and losses
        B2A_imgs, B2A2B_imgs = self.sess.run(
            [self.B2A, self.B2A2B],
            feed_dict={self.real_B: sample_B_imgs}
        )
        BAloss = self.sess.run(
            self.BA_loss,
            feed_dict={self.real_B: sample_B_imgs, self.real_A: sample_A_imgs},
        )

        # Save the images
        save_images(
            A2B_imgs,
            [self.batch_size, 1],
            "./{}/{}/{:06d}_{:04d}_A2B.jpg".format(sample_dir, self.dir_name, epoch_idx, batch_idx)
        )
        save_images(
            B2A_imgs,
            [self.batch_size, 1],
            "./{}/{}/{:06d}_{:04d}_B2A.jpg".format(sample_dir, self.dir_name, epoch_idx, batch_idx)
        )
        save_images(
            A2B2A_imgs,
            [self.batch_size, 1],
            "./{}/{}/{:06d}_{:04d}_A2B2A.jpg".format(sample_dir, self.dir_name, epoch_idx, batch_idx)
        )
        save_images(
            B2A2B_imgs,
            [self.batch_size, 1],
            "./{}/{}/{:06d}_{:04d}_B2A2B.jpg".format(sample_dir, self.dir_name, epoch_idx, batch_idx)
        )

        print("[Sample] AB_loss: {:.8f}, BA_loss: {:.8f}".format(ABloss, BAloss))

    def train(self, args):
        """Train Dual GAN with both paired and unpaired data."""
        decay = 0.9
        self.d_optim = tf.train.RMSPropOptimizer(args.lr, decay=decay).minimize(
            self.d_loss, var_list=self.d_vars
        )
        self.g_optim = tf.train.RMSPropOptimizer(args.lr, decay=decay).minimize(
            self.g_loss, var_list=self.g_vars
        )
        tf.global_variables_initializer().run()
        if self.GAN_type == "wgan":
            self.clip_ops = [
                var.assign(tf.clip_by_value(var, -self.clip_value, self.clip_value))
                for var in self.d_vars
            ]

        self.writer = tf.summary.FileWriter("./logs/" + self.dir_name, self.sess.graph)

        transition_step = 50  # Number of steps to train with paired data

        # Load paired data
        paired_data_A, paired_data_B = self.load_paired_data()
        paired_epoch_size = min(len(paired_data_A), len(paired_data_B)) // self.batch_size
    
        # Load unpaired data
        unpaired_data_A, unpaired_data_B = self.load_unpaired_data()
        unpaired_epoch_size = min(len(unpaired_data_A), len(unpaired_data_B)) // self.batch_size
        print(paired_epoch_size, "Paired Size")
        print(unpaired_epoch_size, "UnPaired Size")
        
        step = 1  # Reset step count
        start_time = time.time()

        for epoch_idx in range(args.epoch):
            print("Epoch: [%2d]" % epoch_idx)

            # Decide whether to use paired or unpaired data based on the step count
            if epoch_idx <= transition_step:
                data_A, data_B, epoch_size = paired_data_A, paired_data_B, paired_epoch_size
                is_paired=True
                print("Inside Paired")
            else:
                data_A, data_B, epoch_size = unpaired_data_A, unpaired_data_B, unpaired_epoch_size
                print("Inside UnPaired")
                is_paired=False

            for batch_idx in range(0, epoch_size):
                imgA_batch, imgB_batch = self.load_training_imgs(data_A, data_B, batch_idx, is_paired)

                # Run optimization step
                self.run_optim(imgA_batch, imgB_batch, step, start_time, batch_idx, is_paired)

                if step % self.log_freq == 0:
                    print("Step: [%4d/%4d]" % (batch_idx, epoch_size))

                if np.mod(step, 100) == 1:
                    self.sample_shotcut(args.sample_dir, epoch_idx, batch_idx)

                if np.mod(step, args.save_freq) == 2:
                    self.save(args.checkpoint_dir, step)

                step += 1

    def load_training_imgs(self, data_A, data_B, idx, is_paired):

        if is_paired:
            # Load images assuming data_A and data_B are paired
            batch_files_A = data_A[idx * self.batch_size : (idx + 1) * self.batch_size]
            batch_files_B = data_B[idx * self.batch_size : (idx + 1) * self.batch_size]
        else:
            # Load images for unpaired training (can be random or sequential)
            batch_files_A = np.random.choice(data_A, self.batch_size, replace=False)
            batch_files_B = np.random.choice(data_B, self.batch_size, replace=False)

        batch_imgs_A = [load_data(f, image_size=self.image_size, flip=self.flip) for f in batch_files_A]
        batch_imgs_B = [load_data(f, image_size=self.image_size, flip=self.flip) for f in batch_files_B]

        batch_imgs_A = np.reshape(np.array(batch_imgs_A).astype(np.float32), (self.batch_size, self.image_size, self.image_size, -1))
        batch_imgs_B = np.reshape(np.array(batch_imgs_B).astype(np.float32), (self.batch_size, self.image_size, self.image_size, -1))

        return batch_imgs_A, batch_imgs_B

    def run_optim(self, batch_A_imgs, batch_B_imgs, counter, start_time, batch_idx, is_paired):
        # Update discriminator
        _, Adfake, Adreal, Bdfake, Bdreal, Ad, Bd = self.sess.run(
            [
                self.d_optim,
                self.Ad_loss_fake,
                self.Ad_loss_real,
                self.Bd_loss_fake,
                self.Bd_loss_real,
                self.Ad_loss,
                self.Bd_loss,
            ],
            feed_dict={self.real_A: batch_A_imgs, self.real_B: batch_B_imgs},
        )

        # For WGAN, clip the discriminator weights
        if "wgan" == self.GAN_type:
            self.sess.run(self.clip_ops)

        # Update generator
        if ("wgan" in self.GAN_type and batch_idx % self.n_critic == 0) or "wgan" not in self.GAN_type:
            if is_paired:
                # Custom logic for paired data
                _, Ag, Bg, cycle_A_loss, cycle_B_loss = self.sess.run(
                    [
                        self.g_optim,
                        self.Ag_loss,
                        self.Bg_loss,
                        self.cycle_A_loss,  # Emphasize cycle consistency for paired data
                        self.cycle_B_loss,
                    ],
                    feed_dict={self.real_A: batch_A_imgs, self.real_B: batch_B_imgs},
                )
            else: 
                # Custom logic for unpaired data
                # Potentially relax cycle consistency or apply different strategies
                _, Ag, Bg, ABloss, BAloss = self.sess.run(
                    [
                        self.g_optim,
                        self.Ag_loss,  # Adjust if needed for unpaired data
                        self.Bg_loss,  # Adjust if needed for unpaired data
                        self.AB_loss,  # Loss for A to B translation
                        self.BA_loss,  # Loss for B to A translation
                    ],
                    feed_dict={self.real_A: batch_A_imgs, self.real_B: batch_B_imgs},
                )

        # Logging training progress
        if batch_idx % self.log_freq == 0:
            elapsed_time = time.time() - start_time
            loss_info = (Ad, Ag, Bd, Bg, cycle_A_loss if is_paired else ABloss, cycle_B_loss if is_paired else BAloss)
            print("time: %4.4f, Ad: %.2f, Ag: %.2f, Bd: %.2f, Bg: %.2f,  A_loss: %.5f, B_loss: %.5f" % (elapsed_time, *loss_info))
            print("Ad_fake: %.2f, Ad_real: %.2f, Bd_fake: %.2f, Bd_real: %.2f" % (Adfake, Adreal, Bdfake, Bdreal))

    def A_d_net(self, imgs, y=None, reuse=False):
        return self.discriminator(imgs, prefix="A_d_", reuse=reuse)

    def B_d_net(self, imgs, y=None, reuse=False):
        return self.discriminator(imgs, prefix="B_d_", reuse=reuse)

    def discriminator(self, image, y=None, prefix="A_d_", reuse=False):
        # image is 256 x 256 x (input_c_dim + output_c_dim)
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            if reuse:
                scope.reuse_variables()
            else:
                assert scope.reuse == False

            h0 = lrelu(conv2d(image, self.dcn, k_h=5, k_w=5, name=prefix + "h0_conv"))
            # h0 is (128 x 128 x self.dcn)
            h1 = lrelu(
                batch_norm(
                    conv2d(h0, self.dcn * 2, name=prefix + "h1_conv"),
                    name=prefix + "bn1",
                )
            )
            # h1 is (64 x 64 x self.dcn*2)
            h2 = lrelu(
                batch_norm(
                    conv2d(h1, self.dcn * 4, name=prefix + "h2_conv"),
                    name=prefix + "bn2",
                )
            )
            # h2 is (32x 32 x self.dcn*4)
            h3 = lrelu(
                batch_norm(
                    conv2d(h2, self.dcn * 8, name=prefix + "h3_conv"),
                    name=prefix + "bn3",
                )
            )
            # h3 is (16 x 16 x self.dcn*8)
            h3 = lrelu(
                batch_norm(
                    conv2d(h3, self.dcn * 8, name=prefix + "h3_1_conv"),
                    name=prefix + "bn3_1",
                )
            )
            # h3 is (8 x 8 x self.dcn*8)

            if self.disc_type == "patchgan":
                h4 = conv2d(h3, 1, name=prefix + "h4")
            else:
                h4 = linear(
                    tf.reshape(h3, [self.batch_size, -1]), 1, prefix + "d_h3_lin"
                )

            return h4

    def A_g_net(self, imgs, reuse=False):
        return self.fcn(imgs, prefix="A_g_", reuse=reuse)

    def B_g_net(self, imgs, reuse=False):
        return self.fcn(imgs, prefix="B_g_", reuse=reuse)

    def calculate_cycle_loss(self, real_image, generated_image):
        if self.loss_metric == "L1":
            return tf.reduce_mean(tf.abs(generated_image - real_image))
        elif self.loss_metric == "L2":
            return tf.reduce_mean(tf.square(generated_image - real_image))

    def load_unpaired_data(self):
        # Load all data
        all_data_A = glob("./datasets/{}/unsupervised/train/A/*.*[g|G]".format(self.dataset_name))
        all_data_B = glob("./datasets/{}/unsupervised/train/B/*.*[g|G]".format(self.dataset_name))

        # Optionally, you could shuffle the data
        np.random.shuffle(all_data_A)
        np.random.shuffle(all_data_B)

        return all_data_A, all_data_B

    def load_paired_data(self):
        # Load paired images from the dataset
        # This assumes you have a way to identify which images are pairs across domains A and B
        data_A = glob("./datasets/{}/supervised/train/A/*.*[g|G]".format(self.dataset_name))
        data_B = glob("./datasets/{}/supervised/train/B/*.*[g|G]".format(self.dataset_name))

        # This part depends on how your dataset is structured.
        # If your paired images have matching filenames in folders A and B, you could pair them like this:
        # paired_data_A = []
        # paired_data_B = []
        # for file_A in data_A:
        #     filename = os.path.basename(file_A)
        #     corresponding_file_B = os.path.join("./datasets/{}/supervised/train/B".format(self.dataset_name), filename)
        #     if os.path.exists(corresponding_file_B):
        #         paired_data_A.append(file_A)
        #         paired_data_B.append(corresponding_file_B)

        # Return the paired data
        return data_A, data_B

    def fcn(self, imgs, prefix=None, reuse=False):
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            if reuse:
                scope.reuse_variables()
            else:
                assert scope.reuse == False

            s = self.image_size
            s2, s4, s8, s16, s32, s64, s128 = (
                int(s / 2),
                int(s / 4),
                int(s / 8),
                int(s / 16),
                int(s / 32),
                int(s / 64),
                int(s / 128),
            )

            # imgs is (256 x 256 x input_c_dim)
            e1 = conv2d(imgs, self.gcn, k_h=5, k_w=5, name=prefix + "e1_conv")
            # e1 is (128 x 128 x self.gcn)
            e2 = batch_norm(
                conv2d(lrelu(e1), self.gcn * 2, name=prefix + "e2_conv"),
                name=prefix + "bn_e2",
            )
            # e2 is (64 x 64 x self.gcn*2)
            e3 = batch_norm(
                conv2d(lrelu(e2), self.gcn * 4, name=prefix + "e3_conv"),
                name=prefix + "bn_e3",
            )
            # e3 is (32 x 32 x self.gcn*4)
            e4 = batch_norm(
                conv2d(lrelu(e3), self.gcn * 8, name=prefix + "e4_conv"),
                name=prefix + "bn_e4",
            )
            # e4 is (16 x 16 x self.gcn*8)
            e5 = batch_norm(
                conv2d(lrelu(e4), self.gcn * 8, name=prefix + "e5_conv"),
                name=prefix + "bn_e5",
            )
            # e5 is (8 x 8 x self.gcn*8)
            e6 = batch_norm(
                conv2d(lrelu(e5), self.gcn * 8, name=prefix + "e6_conv"),
                name=prefix + "bn_e6",
            )
            # e6 is (4 x 4 x self.gcn*8)
            e7 = batch_norm(
                conv2d(lrelu(e6), self.gcn * 8, name=prefix + "e7_conv"),
                name=prefix + "bn_e7",
            )
            # e7 is (2 x 2 x self.gcn*8)
            e8 = batch_norm(
                conv2d(lrelu(e7), self.gcn * 8, name=prefix + "e8_conv"),
                name=prefix + "bn_e8",
            )
            # e8 is (1 x 1 x self.gcn*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(
                tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.gcn * 8],
                name=prefix + "d1",
                with_w=True,
            )
            if self.dropout_rate <= 0.0:
                d1 = batch_norm(self.d1, name=prefix + "bn_d1")
            else:
                d1 = tf.nn.dropout(
                    batch_norm(self.d1, name=prefix + "bn_d1"), self.dropout_rate
                )
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gcn*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(
                tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.gcn * 8],
                name=prefix + "d2",
                with_w=True,
            )
            if self.dropout_rate <= 0.0:
                d2 = batch_norm(self.d2, name=prefix + "bn_d2")
            else:
                d2 = tf.nn.dropout(
                    batch_norm(self.d2, name=prefix + "bn_d2"), self.dropout_rate
                )
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gcn*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(
                tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.gcn * 8],
                name=prefix + "d3",
                with_w=True,
            )
            if self.dropout_rate <= 0.0:
                d3 = batch_norm(self.d3, name=prefix + "bn_d3")
            else:
                d3 = tf.nn.dropout(
                    batch_norm(self.d3, name=prefix + "bn_d3"), self.dropout_rate
                )
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gcn*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(
                tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.gcn * 8],
                name=prefix + "d4",
                with_w=True,
            )
            d4 = batch_norm(self.d4, name=prefix + "bn_d4")

            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gcn*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(
                tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gcn * 4],
                name=prefix + "d5",
                with_w=True,
            )
            d5 = batch_norm(self.d5, name=prefix + "bn_d5")
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gcn*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(
                tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gcn * 2],
                name=prefix + "d6",
                with_w=True,
            )
            d6 = batch_norm(self.d6, name=prefix + "bn_d6")
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gcn*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(
                tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gcn],
                name=prefix + "d7",
                with_w=True,
            )
            d7 = batch_norm(self.d7, name=prefix + "bn_d7")
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gcn*1*2)

            if prefix == "B_g_":
                self.d8, self.d8_w, self.d8_b = deconv2d(
                    tf.nn.relu(d7),
                    [self.batch_size, s, s, self.A_channels],
                    k_h=5,
                    k_w=5,
                    name=prefix + "d8",
                    with_w=True,
                )
            elif prefix == "A_g_":
                self.d8, self.d8_w, self.d8_b = deconv2d(
                    tf.nn.relu(d7),
                    [self.batch_size, s, s, self.B_channels],
                    k_h=5,
                    k_w=5,
                    name=prefix + "d8",
                    with_w=True,
                )
            # d8 is (256 x 256 x output_c_dim)
            return tf.nn.tanh(self.d8)

    def save(self, checkpoint_dir, step):
        model_name = "DualNet.model"
        model_dir = self.dir_name
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(
            self.sess, os.path.join(checkpoint_dir, model_name), global_step=step
        )

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = self.dir_name
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, args):
        """Test DualNet"""
        start_time = time.time()
        tf.global_variables_initializer().run()
        # inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
            test_dir = "./{}/{}".format(args.test_dir, self.dir_name)
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)

            test_log = open(test_dir + "evaluation.txt", "a")
            test_log.write(self.dir_name)

            # fid_value_A = self.calculate_fid_for_domain(args, test_dir, "A", inception_model)
            # fid_value_B = self.calculate_fid_for_domain(args, test_dir, "B", inception_model)

            # print(f"FID for domain A: {fid_value_A}")
            # print(f"FID for domain B: {fid_value_B}")

            # test_log.write(f"FID for domain A: {fid_value_A}\n")
            # test_log.write(f"FID for domain B: {fid_value_B}\n")

            self.test_domain(args, test_log, type="B")
            self.test_domain(args, test_log, type="A")

            # Calculate IoU for domain A and B
            # iou_A = calculate_iou(real_images_A, generated_images_A)
            # iou_B = calculate_iou(real_images_B, generated_images_B)
            
            # print(f"IoU for domain A: {iou_A}")
            # print(f"IoU for domain B: {iou_B}")
            
            # test_log.write(f"IoU for domain A: {iou_A}\n")
            # test_log.write(f"IoU for domain B: {iou_B}\n")

            test_log.close()
            test_log.close()

    def test_domain(self, args, test_log, type="A"):
        test_files = glob(
            "./datasets/{}/val/{}/*.*[g|G]".format(self.dataset_name, type)
        )
        # Load testing input
        print("Loading testing images ...")
        test_imgs = [load_data(f, is_test=True, image_size=self.image_size, flip=args.flip) for f in test_files]
        print("#images loaded: %d" % (len(test_imgs)))
        test_imgs = np.reshape(np.asarray(test_imgs).astype(np.float32),
                            (len(test_files), self.image_size, self.image_size, -1))
        test_imgs = [test_imgs[i * self.batch_size : (i + 1) * self.batch_size]
                    for i in range(0, len(test_imgs) // self.batch_size)]
        test_imgs = np.asarray(test_imgs)
        test_path = "./{}/{}/".format(args.test_dir, self.dir_name)

        psnr_values = []

        # Test input samples
        for i in range(0, len(test_files) // self.batch_size):
            filename_o = test_files[i * self.batch_size].split("/")[-1].split(".")[0]
            print(filename_o)
            idx = i + 1

            if type == "A":
                real_imgs = np.reshape(np.array(test_imgs[i]),
                                    (self.batch_size, self.image_size, self.image_size, -1))
                generated_imgs = self.sess.run(self.A2B, feed_dict={self.real_A: real_imgs})
                save_images(real_imgs, [self.batch_size, 1], test_path + filename_o + "_realA.jpg")
                save_images(generated_imgs, [self.batch_size, 1], test_path + filename_o + "_A2B.jpg")

            elif type == "B":
                real_imgs = np.reshape(np.array(test_imgs[i]),
                                    (self.batch_size, self.image_size, self.image_size, -1))
                generated_imgs = self.sess.run(self.B2A, feed_dict={self.real_B: real_imgs})
                save_images(real_imgs, [self.batch_size, 1], test_path + filename_o + "_realB.jpg")
                save_images(generated_imgs, [self.batch_size, 1], test_path + filename_o + "_B2A.jpg")

            # Calculate PSNR
            for real_img, generated_img in zip(real_imgs, generated_imgs):
                psnr_value = tf.image.psnr(real_img, generated_img, max_val=255)
                psnr_values.append(self.sess.run(psnr_value))

        # Calculate average PSNR
        average_psnr = np.mean(psnr_values)
        print(f"Average PSNR for domain {type}: {average_psnr}")
        test_log.write(f"Average PSNR for domain {type}: {average_psnr}\n")
