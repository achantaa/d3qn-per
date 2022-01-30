import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import InputLayer, Conv2D, BatchNormalization, Dense, Lambda
from utils import PrioritizedReplayBuffer


class QNet(keras.Model):
    """
    Class that defines the Q-Network.
    Structure is as per paper with a slight modification:
    """
    def __init__(self, input_shape, action_space):
        super(QNet, self).__init__()
        self.input_layer = InputLayer(input_shape=input_shape)
        self.conv_layer1 = Conv2D(32, 8, strides=4, padding='valid', activation='relu', use_bias=False,
                                  kernel_initializer=tf.initializers.variance_scaling(2), name='conv1')
        self.batch_norm1 = BatchNormalization(name='bn1')
        self.conv_layer2 = Conv2D(64, 4, strides=2, padding='valid', activation='relu', use_bias=False,
                                  kernel_initializer=tf.initializers.variance_scaling(2), name='conv2')
        self.batch_norm2 = BatchNormalization(name='bn2')
        self.conv_layer3 = Conv2D(64, 3, strides=1, padding='valid', activation='relu', use_bias=False,
                                  kernel_initializer=tf.initializers.variance_scaling(2), name='conv3')
        self.batch_norm3 = BatchNormalization(name='bn3')
        self.conv_flatten = Conv2D(1024, 7, strides=1, padding='valid', activation='relu', use_bias=False,
                                   kernel_initializer=tf.initializers.variance_scaling(2), name='conv_flatten')
        self.splitter = Lambda(lambda w: tf.split(w, 2, 3), name='splitter')
        # self.advantage1 = Dense(512, activation='relu',
        #                         kernel_initializer=tf.initializers.VarianceScaling(2), name='advantage')
        self.advantage = Dense(action_space, name='advantage_out')
        # self.value1 = Dense(512, activation='relu',
        #                     kernel_initializer=tf.initializers.VarianceScaling(2), name='value')
        self.value = Dense(1, name='value_out')

    def call(self, inputs, **kwargs):
        x = self.input_layer(inputs)
        x = self.batch_norm1(self.conv_layer1(x))
        x = self.batch_norm2(self.conv_layer2(x))
        x = self.batch_norm3(self.conv_layer3(x))
        x = self.conv_flatten(x)
        val_stream, adv_stream = self.splitter(x)
        adv = self.advantage(adv_stream)
        val = self.value(val_stream)
        res = val + tf.subtract(adv, tf.reduce_mean(adv, axis=3, keepdims=True))
        return tf.squeeze(res)  # to remove the extra 1-D dimensions from the axes

    def build_graph(self, input_shape):
        """Helper method for generating keras.Model summary"""
        x = keras.Input(shape=input_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))


class D3QNAgent:
    """
    D3QN Agent
    """
    def __init__(self,
                 alpha,
                 gamma,
                 input_shape,
                 action_space,
                 batch_size=64,
                 learning_rate=0.0000625,
                 epsilon_initial=1.0,
                 epsilon_decrement=0.996,
                 epsilon_halfway=0.1,
                 epsilon_final=0.01,
                 epsilon_eval=0.0,
                 epsilon_anneal_frames=1000000,
                 target_update=10000,
                 max_replay_buffer_size=1000000,
                 replay_start_size=50000,
                 max_frames=10000000):
        super(D3QNAgent).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.batch_size = batch_size
        self.max_frames = max_frames
        self.learning_rate = learning_rate
        self.replay_start_size = replay_start_size

        self.epsilon_init = epsilon_initial
        self.epsilon_dec = epsilon_decrement
        self.epsilon_half = epsilon_halfway
        self.epsilon_fin = epsilon_final
        self.epsilon_eval = epsilon_eval
        self.epsilon_anneal_frames = epsilon_anneal_frames
        # Slopes and intercepts for exploration decrease
        # (Credit to Fabio M. Graetz for this and calculating epsilon based on frame number)
        self.slope = -(self.epsilon_init - self.epsilon_half) / self.epsilon_anneal_frames
        self.intercept = self.epsilon_init - self.slope * self.replay_start_size
        self.slope_2 = -(self.epsilon_half - self.epsilon_fin) / (
                self.max_frames - self.epsilon_anneal_frames - self.replay_start_size)
        self.intercept_2 = self.epsilon_fin - self.slope_2 * self.max_frames

        self.action_space = action_space
        self.main_QNet = QNet(input_shape, action_space.n)
        self.target_QNet = QNet(input_shape, action_space.n)
        self.update_target_counter = target_update
        self.memory = PrioritizedReplayBuffer(capacity=max_replay_buffer_size)
        self.huber_loss = keras.losses.Huber()
    
    def compile_models(self):
        """Helper method to compile models after instantiation"""
        self.main_QNet.compile(tf.optimizers.Adam(learning_rate=self.learning_rate))
    
    def update_priorities(self, indices, td_errors):
        """Update Priorities in PER Buffer"""
        for i, e in zip(indices, td_errors):
            self.memory.update(i, e)

    def update_target_net(self):
        """Update Target Network with Main Network weights"""
        self.target_QNet.set_weights(self.main_QNet.get_weights())

    def calc_epsilon(self, frame_number, evaluation=False):
        """Get the appropriate epsilon value from a given frame number"""
        if evaluation:
            return self.epsilon_eval
        elif frame_number < self.replay_start_size:
            return self.epsilon_init
        elif self.replay_start_size <= frame_number < self.replay_start_size + self.epsilon_anneal_frames:
            return self.slope * frame_number + self.intercept
        elif frame_number >= self.replay_start_size + self.epsilon_anneal_frames:
            return self.slope_2 * frame_number + self.intercept_2
    
    def remember(self, state, action, reward, new_state, done):
        """Store an experience in the replay buffer"""
        transition = (state, action, reward, new_state, done)
        max_p = np.max(self.memory.tree.tree[-self.memory.tree.capacity:])
        self.memory.store(max_p, transition)

    def choose_action(self, state, frame_number, evaluation=False):
        """Choose an action based on the current epsilon value"""
        state = state[np.newaxis, :]  # batch = 1 column
        eps = self.calc_epsilon(frame_number, evaluation)
        rnd = np.random.random()
        if rnd < eps:
            return np.random.randint(0, self.action_space.n)
        else:
            action_qs = self.main_QNet.predict(state)
            return tf.argmax(action_qs)

    def learn(self, states, actions, rewards, new_states, dones, is_weights):
        """Learning method for the main Q-Network"""
        main_qs = self.main_QNet(new_states)
        acts = tf.argmax(main_qs, axis=1, output_type=tf.dtypes.int32)

        next_qs = self.target_QNet(new_states)
        idx_flattened = tf.range(0, next_qs.shape[0]) * next_qs.shape[1] + acts
        double_q = tf.gather(tf.reshape(next_qs, [-1]),  # flatten input
                             idx_flattened)  # use flattened indices

        # Bellman
        target_qs = rewards + self.gamma * double_q * (1 - dones)

        with tf.GradientTape() as tape:
            main_q_vals = self.main_QNet(states)
            ohe_actions = tf.one_hot(actions, depth=self.action_space.n, axis=1)
            q_fin = tf.reduce_sum(tf.multiply(main_q_vals, ohe_actions), axis=1)
            td_errors = q_fin - target_qs

            loss = self.huber_loss(target_qs, q_fin)
            loss = tf.reduce_mean(loss * is_weights)
        grads = tape.gradient(loss, self.main_QNet.trainable_weights)
        self.main_QNet.optimizer.apply_gradients(zip(grads, self.main_QNet.trainable_weights))

        return loss, td_errors

    def train(self):
        """Train network on a sampled minibatch, update priorities with the errors"""
        indices, minibatch, is_weights = self.memory.sample(self.batch_size)
        states, actions, rewards, new_states, dones = minibatch
        loss, td_errors = self.learn(states, actions, rewards, new_states, dones, is_weights)
        self.update_priorities(indices, td_errors)
        return loss, td_errors
