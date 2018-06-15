# coding:utf-8
import os
import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense

KERAS_BACKEND = 'tensorflow'    # Use keras with tensorflow

# ENV_NAME = 'Breakout-v0' # Environment name
ENV_NAME = 'SpaceInvaders-v0'  # Environment name
FRAME_WIDTH = 84  # Resized frame width
FRAME_HEIGHT = 84  # Resized frame height
NUM_EPISODES = 12000  # Number of episodes the agent plays
STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
GAMMA = 0.99  # Discount factor
EXPLORATION_STEPS = 1000000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
INITIAL_REPLAY_SIZE = 20000  # Number of steps to populate the replay memory before training starts
NUM_REPLAY_MEMORY = 400000  # Number of replay memory the agent uses for training
BATCH_SIZE = 32  # Mini batch size
TARGET_UPDATE_INTERVAL = 10000  # The frequency with which the target network is updated
ACTION_INTERVAL = 4  # The agent sees only every 4th input
TRAIN_INTERVAL = 4  # The agent selects 4 actions between successive updates
LEARNING_RATE = 0.00025  # Learning rate used by RMSProp
MOMENTUM = 0.95  # Momentum used by RMSProp
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
SAVE_INTERVAL = 300000  # The frequency with which the network is saved
NO_OP_STEPS = 30  # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
LOAD_NETWORK = False
TRAIN = True
SAVE_NETWORK_PATH = 'saved_networks/' + ENV_NAME
SAVE_SUMMARY_PATH = 'summary/' + ENV_NAME
NUM_EPISODES_AT_TEST = 30  # Number of episodes the agent plays at test time

# エージェントのクラス設定
class Agent():
    def __init__(self, num_actions):
        '''
        コンストラクタの設定：
        引数として受け取ったアクション数などを用いて，クラス内の各変数群を初期化
        '''
        self.num_actions = num_actions  # Num of Actions
        self.epsilon = INITIAL_EPSILON  # epsilon
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS   # increment step of epsilon
        self.t = 0  #　初期化
        self.repeated_action = 0 #

        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        # Create replay memory
        self.replay_memory = deque() # 処理速度高速化のためキュー・スタックを行うために，dequeを使用

        # Create q network
        self.s, self.q_values, q_network = self.build_network() # Networkの作成，s?,Q値，ネットワーク全体を取得
        q_network_weights = q_network.trainable_weights # Networkの重み

        # Create target network（Same as creation of q network）
        self.st, self.target_q_values, target_network = self.build_network()
        target_network_weights = target_network.trainable_weights

        # Define target network update operation
        # .assign():新しい行の作成，ターゲットネットワークの重みすべてに対して，q-networkの重みを追加する？？
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in range(len(target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.loss, self.grad_update = self.build_training_op(q_network_weights)

        self.sess = tf.InteractiveSession()     # セッションの設定
        self.saver = tf.train.Saver(q_network_weights)      # saverの設定
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(SAVE_SUMMARY_PATH, self.sess.graph)     # ファイル出力

        if not os.path.exists(SAVE_NETWORK_PATH):   # pathが存在しない時に作成
            os.makedirs(SAVE_NETWORK_PATH)

        self.sess.run(tf.initialize_all_variables())    # session内変数群の初期化

        # Load network
        if LOAD_NETWORK:
            self.load_network()

        # Initialize target network
        self.sess.run(self.update_target_network)

    def build_network(self):
        '''
        ネットワークの構築：
        DQNのためのネットワークを作成，基本的にKerasで書いていく．
        Q値はネットワークから出てくる値

        return : プレースホルダ，　Q値，　ネットワークのモデル
        '''
        model = Sequential() # Kerasの中心的なデータ構造はmodel, レイヤーを構成する方法. 今回これを使用
        # Layorの作成
        model.add(Convolution2D(32, 8, 8, activation="relu", input_shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT)))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
        model.add(Flatten())                        # 入力の平滑化，バッチサイズには変化なし
        model.add(Dense(512, activation='relu'))    # 全結合ニューラルネットワークレイヤー，今回は活性化にrelu関数を使用
        model.add(Dense(self.num_actions))          # 全結合ニューラルネットワークレイヤー，活性化なし

        s = tf.placeholder(tf.float32, [None, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT]) # データ収容用フォルダー，データは未定
        q_values = model(s)                         # モデルの出力値？？ modelのs番目？？の値を参照？？

        return s, q_values, model

    def build_training_op(self, q_network_weights):
        '''
        学習環境の構築：
        基本的にtensorflowで設定していく

        return : プレースホルダ，　損失，　更新された勾配の値
        '''
        a = tf.placeholder(tf.int64, [None])                    # Placeholderの設定．int64型での入力が多いため，int64で作成
        y = tf.placeholder(tf.float32, [None])                  # Placeholderの設定

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)   # 出力層として分類したいクラスの数と同じ次元のベクトルを出力
        q_value = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), reduction_indices=1)     #

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)                             # 誤差の計算
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)      # 0~1に正規化
        linear_part = error - quadratic_part                    # 線形部分の計算？？
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)    # 損失を計算

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)       # RMSPropで最小化
        grad_update = optimizer.minimize(loss, var_list=q_network_weights)      # 勾配の更新

        return a, y, loss, grad_update

    def get_initial_state(self, observation, last_observation):
        '''
        初期状態取得関数:
        processed_observationに画像情報入力
        グレースケールに変換，resizeの実行

        return : 状態値
        '''
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
        state = [processed_observation for _ in range(STATE_LENGTH)]
        return np.stack(state, axis=0)  # 行方向（axis=0）に状態を積み重ねていく

    def get_action(self, state):
        '''
        行動選択関数：
        インターバルを保ちながらε-greedy法で行動選択

        return : 行動
        '''
        action = self.repeated_action

        # アクションインターバルで割り切れるときに新しい行動を選択
        if self.t % ACTION_INTERVAL == 0:
            if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE:
                action = random.randrange(self.num_actions)
            else:
                # eval関数：self.sの値をfloat32型でなおかつ255で割った状態値を割り当ててeval関数を実行
                # argmax関数：eval関数で求めたq値に対して，最も値の大きかった行動を取ってくる
                action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
            self.repeated_action = action

        # epsilonを一定間隔で減衰させる
        if self.epsilon > FINAL_EPSILON and self.t >= INITIAL_REPLAY_SIZE:
            self.epsilon -= self.epsilon_step

        return action

    def run(self, state, action, reward, terminal, observation):
        '''
        学習関数:
        main関数から呼び出し後，学習関連を総括して行う関数
        基本的な流れはQ学習と同じ

        return ： 次の状態値
        '''
        next_state = np.append(state[1:, :, :], observation, axis=0)

        # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
        # 得られた報酬をクリップ（離散化して，-1, 0, 1として設定）
        reward = np.sign(reward)

        # replay memoryへのストア
        self.replay_memory.append((state, action, reward, next_state, terminal))
        # メモリが溜まっていったら昔のものを捨てて新しい物を入れる
        if len(self.replay_memory) > NUM_REPLAY_MEMORY:
            self.replay_memory.popleft()

        if self.t >= INITIAL_REPLAY_SIZE:
            # ネットワークの学習
            if self.t % TRAIN_INTERVAL == 0:
                self.train_network()

            # ターゲットネットワークの更新
            if self.t % TARGET_UPDATE_INTERVAL == 0:
                self.sess.run(self.update_target_network)   # 重みなどの設定値を入力して，ネットワークの更新

            # ネットワークの保存
            if self.t % SAVE_INTERVAL == 0:
                save_path = self.saver.save(self.sess, SAVE_NETWORK_PATH + '/' + ENV_NAME, global_step=(self.t))
                print('Successfully saved: ' + save_path)

        self.total_reward += reward
        self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
        self.duration += 1

        if terminal:  # 終了時の処理
            # summaryの作成
            if self.t >= INITIAL_REPLAY_SIZE:
                stats = [self.total_reward, self.total_q_max / float(self.duration),
                        self.duration, self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL))]
                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.episode + 1)

            # Debug
            if self.t < INITIAL_REPLAY_SIZE:
                mode = 'random'
            elif INITIAL_REPLAY_SIZE <= self.t < INITIAL_REPLAY_SIZE + EXPLORATION_STEPS:
                mode = 'explore'
            else:
                mode = 'exploit'
            print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
                self.episode + 1, self.t, self.duration, self.epsilon,
                self.total_reward, self.total_q_max / float(self.duration),
                self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL)), mode))

            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

        self.t += 1

        return next_state

    def train_network(self):
        '''
        ネットワーク学習関数：


        '''
        # バッチの初期化
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        for data in minibatch:
            # append関数でデータを追加していく
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0

        # eval関数：self.sの値をfloat32型でなおかつ255で割った状態値を割り当ててeval関数を実行
        target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_state_batch) / 255.0)})
        y_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(target_q_values_batch, axis=1)

        loss, _ = self.sess.run([self.loss, self.grad_update], feed_dict={
            self.s: np.float32(np.array(state_batch) / 255.0),
            self.a: action_batch,
            self.y: y_batch
        })

        self.total_loss += loss

    def setup_summary(self):
        '''
        セットアップ関数：
        報酬，Q値，総エピソード数，lossをtensorflow summaryに記憶

        return : サマリ
        '''
        episode_total_reward = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Total Reward/Episode', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Average Max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Average Loss/Episode', episode_avg_loss)
        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def load_network(self):
        '''
        ネットワーク読み込み関数：
        チェックポイントに来た時に，学習結果をロード
        '''
        checkpoint = tf.train.get_checkpoint_state(SAVE_NETWORK_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')

    def get_action_at_test(self, state):
        '''
        テストモードにおける行動選択関数：
        ε-greedy法による行動選択

        return : 行動
        '''
        action = self.repeated_action

        if self.t % ACTION_INTERVAL == 0:
            if random.random() <= 0.05:
                action = random.randrange(self.num_actions)
            else:
                action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
            self.repeated_action = action

        self.t += 1

        return action


def preprocess(observation, last_observation):
    '''
    前処理関数：
    processed_observationに画像情報を入力
    グレースケール変換を行い，リサイズを行う
    '''
    processed_observation = np.maximum(observation, last_observation)
    processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
    return np.reshape(processed_observation, (1, FRAME_WIDTH, FRAME_HEIGHT))


def main():
    '''
    main関数：
    学習モード or 学習結果テストモードの2パターンを設定
    総エピソード数を回すfor文
        初期設定
        終了条件が来るまで回るwhile文
            前状態を現状態に
            行動選択
            ゲーム環境への働きかけ
            アニメーション
            次の状態のための環境準備
            学習プロセスで学習を行う
    '''
    # Create gym environment
    env = gym.make(ENV_NAME)
    agent = Agent(num_actions=env.action_space.n)   # クラスの呼び出し，アクションの総数を引き渡し

    # Train mode or Test mode
    if TRAIN:  # Train mode
        for _ in range(NUM_EPISODES):
            terminal = False
            observation = env.reset()   # ゲーム環境のリセット
            for _ in range(random.randint(1, NO_OP_STEPS)):
                last_observation = observation
                observation, _, _, _ = env.step(0)  # Do nothing
            state = agent.get_initial_state(observation, last_observation)
            while not terminal:
                last_observation = observation
                action = agent.get_action(state)
                observation, reward, terminal, _ = env.step(action)
                # env.render()    # Animation
                processed_observation = preprocess(observation, last_observation)   # Preparation of environment
                state = agent.run(state, action, reward, terminal, processed_observation)   # Action
    else:  # Test mode
        # env.monitor.start(ENV_NAME + '-test')
        for _ in range(NUM_EPISODES_AT_TEST):
            terminal = False
            observation = env.reset()
            for _ in range(random.randint(1, NO_OP_STEPS)):
                last_observation = observation
                observation, _, _, _ = env.step(0)  # Do nothing
            state = agent.get_initial_state(observation, last_observation)
            while not terminal:
                last_observation = observation
                action = agent.get_action_at_test(state)
                observation, _, terminal, _ = env.step(action)
                env.render()
                processed_observation = preprocess(observation, last_observation)
                state = np.append(state[1:, :, :], processed_observation, axis=0)
        # env.monitor.close()


if __name__ == '__main__':
    main()
