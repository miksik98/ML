import gym
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

# environment = gym.make('CartPole-v1')
# model = tf.keras.models.load_model('final.model')
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))
# states = [
#     np.array([0, 0, 0, 0]),
#     np.array([0, 0, 0, -200]),
#     np.array([-4.8, 0, 0, 0])
# ]

# for state in states:
#     m = model(np.reshape(state, (1, state.size)))
#     print('state', state)
#     print('value', float(m[1][0].numpy()))
#     print('action', float(tfp.distributions.Categorical(probs=m[0][0]).experimental_sample_and_log_prob()[0]))
#
# environment.close()


environment = gym.make('LunarLander-v2')
model = tf.keras.models.load_model('final.model_lander')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001))
states = [
    np.array([0, 0.1, 0, 0, 0, 0, 0, 0]),
    np.array([0, 0, 0, 0, -50, 0, 0, 0])
]

for state in states:
    m = model(np.reshape(state, (1, state.size)))
    print('state', state)
    print('value', float(m[1][0].numpy()))
    print('action', float(tfp.distributions.Categorical(probs=m[0][0]).experimental_sample_and_log_prob()[0]))

environment.close()