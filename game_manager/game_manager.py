#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QMainWindow, QFrame, QDesktopWidget, QApplication, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt, QBasicTimer, pyqtSignal
from PyQt5.QtGui import QPainter, QColor

from board_manager import BOARD_DATA, Shape
from block_controller import BLOCK_CONTROLLER

from argparse import ArgumentParser
import time
import json
import pprint

############# add DQN #################

#!/usr/bin/python3
# -*- coding: utf-8 -*-

from datetime import datetime
import pprint
import random

import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from board_manager import BOARD_DATA

BATCH_SIZE = 32
CAPACITY = 10000
GAMMA = 0.99  # 時間割引率
MAX_STEPS = 200  # 1試行のstep数
NUM_EPISODES = 500  # 最大試行回数

from collections import namedtuple

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY  # メモリの最大長さ
        self.memory = []  # 経験を保存する変数
        self.index = 0  # 保存するindexを示す変数

    def push(self, state, action, state_next, reward):
        '''transition = (state, action, state_next, reward)をメモリに保存する'''

        if len(self.memory) < self.capacity:
            self.memory.append(None)  # メモリが満タンでないときは足す

        # namedtupleのTransitionを使用し、値とフィールド名をペアにして保存します
        self.memory[self.index] = Transition(state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity  # 保存するindexを1つずらす

    def sample(self, batch_size):
        '''batch_size分だけ、ランダムに保存内容を取り出す'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        '''関数lenに対して、現在の変数memoryの長さを返す'''
        return len(self.memory)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_out)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        output = self.fc3(h2)
        return output

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  # CartPoleの行動（右に左に押す）の2を取得

        # 経験を記憶するメモリオブジェクトを生成
        self.memory = ReplayMemory(CAPACITY)

        # ニューラルネットワークを構築
        n_in, n_mid, n_out = num_states, 32, num_actions
        self.main_q_network = Net(n_in, n_mid, n_out)  # Netクラスを使用
        self.target_q_network = Net(n_in, n_mid, n_out)  # Netクラスを使用
        print(self.main_q_network)  # ネットワークの形を出力

        # 最適化手法の設定
        self.optimizer = optim.Adam(
            self.main_q_network.parameters(), lr=0.0001)

    def replay(self):
        '''Experience Replayでネットワークの結合パラメータを学習'''

        # 1. メモリサイズの確認
        if len(self.memory) < BATCH_SIZE:
            return

        # 2. ミニバッチの作成
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()

        # 3. 教師信号となるQ(s_t, a_t)値を求める
        self.expected_state_action_values = self.get_expected_state_action_values()

        # 4. 結合パラメータの更新
        self.update_main_q_network()

    def decide_action(self, state, episode):
        '''現在の状態に応じて、行動を決定する'''
        # ε-greedy法で徐々に最適行動のみを採用する
        # epsilon = 0.5 * (1 / (episode + 1))
        epsilon = 1.0 - episode/100000
        if epsilon <= 0:
            epsilon = 0

        print("epsilon : {0}".format(epsilon))

        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()  # ネットワークを推論モードに切り替える
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1, 1)
            # ネットワークの出力の最大値のindexを取り出します = max(1)[1]
            # .view(1,1)は[torch.LongTensor of size 1]　を size 1x1 に変換します

        else:
            # 0,1の行動をランダムに返す
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]])  # 0,1の行動をランダムに返す
            # actionは[torch.LongTensor of size 1x1]の形になります

        return action

    def make_minibatch(self):
        '''2. ミニバッチの作成'''

        # 2.1 メモリからミニバッチ分のデータを取り出す
        transitions = self.memory.sample(BATCH_SIZE)

        # 2.2 各変数をミニバッチに対応する形に変形
        # transitionsは1stepごとの(state, action, state_next, reward)が、BATCH_SIZE分格納されている
        # つまり、(state, action, state_next, reward)×BATCH_SIZE
        # これをミニバッチにしたい。つまり
        # (state×BATCH_SIZE, action×BATCH_SIZE, state_next×BATCH_SIZE, reward×BATCH_SIZE)にする
        batch = Transition(*zip(*transitions))

        # 2.3 各変数の要素をミニバッチに対応する形に変形し、ネットワークで扱えるようVariableにする
        # 例えばstateの場合、[torch.FloatTensor of size 1x4]がBATCH_SIZE分並んでいるのですが、
        # それを torch.FloatTensor of size BATCH_SIZEx4 に変換します
        # 状態、行動、報酬、non_finalの状態のミニバッチのVariableを作成
        # catはConcatenates（結合）のことです。
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        '''3. 教師信号となるQ(s_t, a_t)値を求める'''

        # 3.1 ネットワークを推論モードに切り替える
        self.main_q_network.eval()
        self.target_q_network.eval()

        # 3.2 ネットワークが出力したQ(s_t, a_t)を求める
        # self.model(state_batch)は、右左の両方のQ値を出力しており
        # [torch.FloatTensor of size BATCH_SIZEx2]になっている。
        # ここから実行したアクションa_tに対応するQ値を求めるため、action_batchで行った行動a_tが右か左かのindexを求め
        # それに対応するQ値をgatherでひっぱり出す。
        self.state_action_values = self.main_q_network(
            self.state_batch).gather(1, self.action_batch)

        # 3.3 max{Q(s_t+1, a)}値を求める。ただし次の状態があるかに注意。

        # cartpoleがdoneになっておらず、next_stateがあるかをチェックするインデックスマスクを作成
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                    self.batch.next_state)))
        # まずは全部0にしておく
        next_state_values = torch.zeros(BATCH_SIZE)

        a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor)

        # 次の状態での最大Q値の行動a_mをMain Q-Networkから求める
        # 最後の[1]で行動に対応したindexが返る
        a_m[non_final_mask] = self.main_q_network(
            self.non_final_next_states).detach().max(1)[1]

        # 次の状態があるものだけにフィルターし、size 32を32×1へ
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # 次の状態があるindexの、行動a_mのQ値をtarget Q-Networkから求める
        # detach()で取り出す
        # squeeze()でsize[minibatch×1]を[minibatch]に。
        next_state_values[non_final_mask] = self.target_q_network(
            self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        # 3.4 教師となるQ(s_t, a_t)値を、Q学習の式から求める
        expected_state_action_values = self.reward_batch + GAMMA * next_state_values

        return expected_state_action_values

    def update_main_q_network(self):
        '''4. 結合パラメータの更新'''

        # 4.1 ネットワークを訓練モードに切り替える
        self.main_q_network.train()

        # 4.2 損失関数を計算する（smooth_l1_lossはHuberloss）
        # expected_state_action_valuesは
        # sizeが[minbatch]になっているので、unsqueezeで[minibatch x 1]へ
        loss = F.smooth_l1_loss(self.state_action_values,
                                self.expected_state_action_values.unsqueeze(1))

        print("loss : {0}".format(loss))

        # 4.3 結合パラメータを更新する
        self.optimizer.zero_grad()  # 勾配をリセット
        loss.backward()  # バックプロパゲーションを計算
        self.optimizer.step()  # 結合パラメータを更新

    def update_target_q_network(self):  # DDQNで追加
        '''Target Q-NetworkをMainと同じにする'''
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())

class Agent:
    def __init__(self, num_states, num_actions):
        '''課題の状態と行動の数を設定する'''
        self.brain = Brain(num_states, num_actions)  # エージェントが行動を決定するための頭脳を生成

    def update_q_function(self):
        '''Q関数を更新する'''
        self.brain.replay()

    def get_action(self, state, episode):
        '''行動を決定する'''
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        '''memoryオブジェクトに、state, action, state_next, rewardの内容を保存する'''
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        '''Target Q-NetworkをMain Q-Networkと同じに更新'''
        self.brain.update_target_q_network()

#######################################

def get_option(game_time, manual, use_sample, drop_speed, random_seed, obstacle_height, obstacle_probability, resultlogjson):
    argparser = ArgumentParser()
    argparser.add_argument('--game_time', type=int,
                           default=game_time,
                           help='Specify game time(s)')
    argparser.add_argument('--manual',
                           default=manual,
                           help='Specify if manual control')
    argparser.add_argument('--use_sample',
                           default=use_sample,
                           help='Specify if use sample')
    argparser.add_argument('--drop_speed', type=int,
                           default=drop_speed,
                           help='Specify drop_speed(s)')
    argparser.add_argument('--seed', type=int,
                           default=random_seed,
                           help='Specify random seed')
    argparser.add_argument('--obstacle_height', type=int,
                           default=obstacle_height,
                           help='Specify obstacle height')
    argparser.add_argument('--obstacle_probability', type=int,
                           default=obstacle_probability,
                           help='Specify obstacle probability')
    argparser.add_argument('--resultlogjson', type=str,
                           default=resultlogjson,
                           help='result json log file path')
    return argparser.parse_args()

class Game_Manager(QMainWindow):

    # a[n] = n^2 - n + 1
    LINE_SCORE_1 = 100
    LINE_SCORE_2 = 300
    LINE_SCORE_3 = 700
    LINE_SCORE_4 = 1300
    GAMEOVER_SCORE = -500

    def __init__(self):
        super().__init__()

        self.episode = 0
        self.step = 0

        self.num_states = 7
        self.num_actions = 40

        self.agent = Agent(self.num_states, self.num_actions)

        self.done = False

        self.action = None
        self.observation = None
        self.state = None
        self.observation_next = None
        self.state_next = None
        self.reward = None

        self.state_property = None
        self.state_property_next = None

        self.isStarted = False
        self.isPaused = False
        self.nextMove = None
        self.lastShape = Shape.shapeNone

        self.game_time = -1
        self.block_index = 0
        self.manual = None
        self.use_sample = None
        self.drop_speed = 1000
        self.random_seed = time.time() * 10000000 # 0
        self.obstacle_height = 0
        self.obstacle_probability = 0
        self.resultlogjson = ""
        args = get_option(self.game_time,
                          self.manual,
                          self.use_sample,
                          self.drop_speed,
                          self.random_seed,
                          self.obstacle_height,
                          self.obstacle_probability,
                          self.resultlogjson)
        if args.game_time >= 0:
            self.game_time = args.game_time
        if args.manual in ("y", "g"):
            self.manual = args.manual
        if args.use_sample == "y":
            self.use_sample = args.use_sample
        if args.drop_speed >= 0:
            self.drop_speed = args.drop_speed
        if args.seed >= 0:
            self.random_seed = args.seed
        if args.obstacle_height >= 0:
            self.obstacle_height = args.obstacle_height
        if args.obstacle_probability >= 0:
            self.obstacle_probability = args.obstacle_probability
        if len(args.resultlogjson) != 0:
            self.resultlogjson = args.resultlogjson
        self.initUI()

    def initUI(self):
        self.gridSize = 22
        self.speed = self.drop_speed # block drop speed

        self.timer = QBasicTimer()
        self.setFocusPolicy(Qt.StrongFocus)

        hLayout = QHBoxLayout()

        random_seed_Nextshape = self.random_seed
        self.tboard = Board(self, self.gridSize,
                            self.game_time,
                            random_seed_Nextshape,
                            self.obstacle_height,
                            self.obstacle_probability)
        hLayout.addWidget(self.tboard)

        self.sidePanel = SidePanel(self, self.gridSize)
        hLayout.addWidget(self.sidePanel)

        self.statusbar = self.statusBar()
        self.tboard.msg2Statusbar[str].connect(self.statusbar.showMessage)

        self.start()

        self.center()
        self.setWindowTitle('Tetris')
        self.show()

        self.setFixedSize(self.tboard.width() + self.sidePanel.width(),
                          self.sidePanel.height() + self.statusbar.height())

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)

    def start(self):
        if self.isPaused:
            return

        self.isStarted = True
        self.tboard.score = 0
        BOARD_DATA.clear()
        BOARD_DATA.createNewPiece()
        self.tboard.msg2Statusbar.emit(str(self.tboard.score))
        self.timer.start(self.speed, self)

    def pause(self):
        if not self.isStarted:
            return

        self.isPaused = not self.isPaused

        if self.isPaused:
            self.timer.stop()
            self.tboard.msg2Statusbar.emit("paused")
        else:
            self.timer.start(self.speed, self)

        self.updateWindow()

    def resetfield(self):
        # self.tboard.score = 0
        self.tboard.reset_cnt += 1
        self.tboard.score += Game_Manager.GAMEOVER_SCORE
        BOARD_DATA.clear()
        BOARD_DATA.createNewPiece()

    def reset_episode(self):
        self.tboard.score = 0
        self.tboard.dropdownscore = 0
        self.tboard.linescore = 0
        self.tboard.line = 0
        self.tboard.line_score_stat = [0, 0, 0, 0]
        self.tboard.reset_cnt = 0
        self.tboard.start_time = time.time()
        
        self.episode += 1
        self.step = 0

        BOARD_DATA.clear()
        BOARD_DATA.createNewPiece()

    def updateWindow(self):
        self.tboard.updateData()
        self.sidePanel.updateData()
        self.update()

    def get_holes(self, board):
        board = np.array(board).reshape([22, 10])
        hole_num = 0
        col = 0
        for col in range(0, 10):
            board_col = board[:, col].tolist()
            row = 0
            while row < 22 and board_col[row] == 0:
                row += 1
            hole_num += len([x for x in board_col[row+1:] if x==0])
        
        return hole_num

    def get_bumpiness_and_height(self, board):
        board = np.array(board).reshape(22, 10)
        mask = board != 0
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), 22)
        heights = 22 - invert_heights
        total_height = np.sum(heights)
        currs = heights[:-1]
        nexts = heights[1:]
        diffs = np.abs(currs - nexts)
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height

    def get_row_connection(self, board):
        board = np.array(board).reshape(22, 10)
        mask = board != 0
        row_sum = np.sum(mask, axis=1)
        row_index = np.where(row_sum > 0)[0]
        max_conect_dict = {}
        max_conect_total = 0
        for row in row_index:
            mask_row = mask[row,:]
            conect_cnt = 0
            conect_list = []
            for col in range(mask_row.shape[0]):
                if mask_row[col] == True:
                    conect_cnt += 1
                else:
                    conect_list.append(conect_cnt)
                    conect_cnt = 0
            max_conect_num = max(conect_list)
            if max_conect_num != 1:
                max_conect_total += max_conect_num
        return max_conect_total

    def get_boutyou_piece_mask(self, board):
        board = np.array(board).reshape(22, 10)
        mask = board != 0
        index = np.where(mask > 0)
        index_total = len(index[0])
        for i in range(index_total):
            if(index[1][i]+1 <= 9):
                mask[index[0][i], index[1][i]+1] = True
            if(index[1][i]-1 >= 0):
                mask[index[0][i], index[1][i]-1] = True

        mask = mask.astype(np.int)
        return mask

    
    def timerEvent(self, event):
        # callback function for user control

        if event.timerId() == self.timer.timerId():
            next_x = 0
            next_y_moveblocknum = 0
            y_operation = -1

            if BLOCK_CONTROLLER and not self.nextMove:
                # update CurrentBlockIndex
                if BOARD_DATA.currentY <= 1:
                    self.block_index = self.block_index + 1

                # nextMove data structure
                nextMove = {"strategy":
                                {
                                  "direction": "none",    # next shape direction ( 0 - 3 )
                                  "x": "none",            # next x position (range: 0 - (witdh-1) )
                                  "y_operation": "none",  # movedown or dropdown (0:movedown, 1:dropdown)
                                  "y_moveblocknum": "none", # amount of next y movement
                                  },
                            }
                # get nextMove from GameController
                GameStatus = self.getGameStatus()
                self.observation = GameStatus["field_info"]["backboard"]

                self.done = False

                # self.observation = GameStatus["field_info"]["withblock"]
                # self.state = np.array(self.observation)
                # self.state = torch.from_numpy(self.state).type(torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
                # self.state = torch.unsqueeze(self.state, 0)
                # self.state = self.state / 7.0

                if(self.step == 0):
                    bumpiness, height = self.get_bumpiness_and_height(self.observation)
                    holes = self.get_holes(self.observation)
                    removedlines = 0
                    dropdownlines = 0
                    current_block = GameStatus["block_info"]["currentShape"]["index"]
                    next_block = GameStatus["block_info"]["nextShape"]["index"]
                    self.state_property = [removedlines, dropdownlines, holes, bumpiness, height, current_block, next_block]
                    self.state_property = np.array(self.state_property)
                    self.state_property = torch.from_numpy(self.state_property).type(torch.FloatTensor)
                    self.state_property = torch.unsqueeze(self.state_property, 0)

                # self.action = self.agent.get_action(self.state, self.episode)
                self.action = self.agent.get_action(self.state_property, self.episode)

                self.nextMove = BLOCK_CONTROLLER.GetNextMove(GameStatus, nextMove, self.action.item())

                if self.manual in ("y", "g"):
                    # ignore nextMove, for manual controll
                    self.nextMove["strategy"]["x"] = BOARD_DATA.currentX
                    self.nextMove["strategy"]["y_moveblocknum"] = 1
                    self.nextMove["strategy"]["y_operation"] = 0
                    self.nextMove["strategy"]["direction"] = BOARD_DATA.currentDirection

            if self.nextMove:
                # shape direction operation
                next_x = self.nextMove["strategy"]["x"]
                next_y_moveblocknum = self.nextMove["strategy"]["y_moveblocknum"]
                y_operation = self.nextMove["strategy"]["y_operation"]
                next_direction = self.nextMove["strategy"]["direction"]
                k = 0
                while BOARD_DATA.currentDirection != next_direction and k < 4:
                    ret = BOARD_DATA.rotateRight()
                    if ret == False:
                        print("cannot rotateRight")
                        break
                    k += 1
                # x operation
                k = 0
                while BOARD_DATA.currentX != next_x and k < 5:
                    if BOARD_DATA.currentX > next_x:
                        ret = BOARD_DATA.moveLeft()
                        if ret == False:
                            print("cannot moveLeft")
                            break
                    elif BOARD_DATA.currentX < next_x:
                        ret = BOARD_DATA.moveRight()
                        if ret == False:
                            print("cannot moveRight")
                            break
                    k += 1

            # dropdown/movedown lines
            dropdownlines = 0
            removedlines = 0
            if y_operation == 1: # dropdown
                removedlines, dropdownlines = BOARD_DATA.dropDown()
            else: # movedown, with next_y_moveblocknum lines
                k = 0
                while True:
                    removedlines, movedownlines = BOARD_DATA.moveDown()
                    if movedownlines < 1:
                        # if already dropped
                        break
                    k += 1
                    if k >= next_y_moveblocknum:
                        # if already movedown next_y_moveblocknum block
                        break

            self.UpdateScore(removedlines, dropdownlines)

            GameStatus = self.getGameStatus()
            self.observation_next = GameStatus["field_info"]["backboard"]

            self.reward = 0

            # if BOARD_DATA.currentY < 1:
            #     self.state_next = None
            #     print("reset episode")
            #     self.reset_episode()
            #     self.reward = torch.FloatTensor([Game_Manager.GAMEOVER_SCORE])
            #     self.done = True

            # if removedlines > 0:
            #     self.state_next = np.array(self.observation_next)
            #     self.state_next = torch.from_numpy(self.state_next).type(torch.FloatTensor)
            #     self.state_next = torch.unsqueeze(self.state_next, 0)
            #     self.state_next = self.state_next / 7.0

            #     if removedlines == 1:
            #         linescore = Game_Manager.LINE_SCORE_1
            #     elif removedlines == 2:
            #         linescore = Game_Manager.LINE_SCORE_2
            #     elif removedlines == 3:
            #         linescore = Game_Manager.LINE_SCORE_3
            #     elif removedlines == 4:
            #         linescore = Game_Manager.LINE_SCORE_4

            #     self.reward = torch.FloatTensor([linescore])

            # else:
            #     self.state_next = np.array(self.observation_next)
            #     self.state_next = torch.from_numpy(self.state_next).type(torch.FloatTensor)
            #     self.state_next = torch.unsqueeze(self.state_next, 0)
            #     self.state_next = self.state_next / 7.0
            #     self.reward = torch.FloatTensor([0.0])

            if BOARD_DATA.currentY < 1:
                self.state_property_next = None
                print("reset episode")
                self.reset_episode()
                self.reward = torch.FloatTensor([Game_Manager.GAMEOVER_SCORE])
                self.done = True

            elif removedlines > 0:
                bumpiness, height = self.get_bumpiness_and_height(self.observation_next)
                holes = self.get_holes(self.observation_next)
                current_block = GameStatus["block_info"]["currentShape"]["index"]
                next_block = GameStatus["block_info"]["nextShape"]["index"]
                self.state_property_next = [removedlines, dropdownlines, holes, bumpiness, height, current_block, next_block]
                self.state_property_next = np.array(self.state_property_next)
                self.state_property_next = torch.from_numpy(self.state_property_next).type(torch.FloatTensor)
                self.state_property_next = torch.unsqueeze(self.state_property_next, 0)

                if removedlines == 1:
                    linescore = Game_Manager.LINE_SCORE_1
                elif removedlines == 2:
                    linescore = Game_Manager.LINE_SCORE_2
                elif removedlines == 3:
                    linescore = Game_Manager.LINE_SCORE_3
                elif removedlines == 4:
                    linescore = Game_Manager.LINE_SCORE_4

                self.reward = torch.FloatTensor([linescore])

            else:
                bumpiness, height = self.get_bumpiness_and_height(self.observation_next)
                holes = self.get_holes(self.observation_next)
                current_block = GameStatus["block_info"]["currentShape"]["index"]
                next_block = GameStatus["block_info"]["nextShape"]["index"]
                self.state_property_next = [removedlines, dropdownlines, holes, bumpiness, height, current_block, next_block]
                self.state_property_next = np.array(self.state_property_next)
                self.state_property_next = torch.from_numpy(self.state_property_next).type(torch.FloatTensor)
                self.state_property_next = torch.unsqueeze(self.state_property_next, 0)
                self.reward = torch.FloatTensor([0.0])

                
            ### connect reward ###
            # connect_num = self.get_row_connection(self.observation)
            # connect_next_num = self.get_row_connection(self.observation_next)
            # print("observation : \n {0}".format(np.array(self.observation).reshape(22,10)))
            # print("connect_num : {0}".format(connect_num))
            # print("observation_next : \n {0}".format(np.array(self.observation_next).reshape(22,10)))
            # print("connect_num_next : {0}".format(connect_next_num))


            # only_put_piece = np.array(self.observation_next) - np.array(self.observation)
            # only_put_piece_mask = only_put_piece != 0
            # only_put_piece_mask_boutyou = self.get_boutyou_piece_mask(only_put_piece)
            # only_put_piece_around = only_put_piece_mask_boutyou - only_put_piece_mask.astype(np.int).reshape(22, 10)
            # rinsetu = only_put_piece_around * np.array(self.observation_next).reshape(22, 10)
            # connect_flag = rinsetu.sum()

            # if(connect_flag > 0):
            #     connect_diff = connect_next_num - connect_num
            #     self.reward += torch.FloatTensor([connect_diff])
            #     print("connect score : {0}".format(connect_diff))
            ########################    

            # メモリに経験を追加
            # self.agent.memorize(self.state, self.action, self.state_next, self.reward)
            self.agent.memorize(self.state_property, self.action, self.state_property_next, self.reward)

            print("state_property : {0}".format(self.state_property))
            print("state_property_next : {0}".format(self.state_property_next))
            print("action : {0}".format(self.action))
            print("reward : {0}".format(self.reward))

            # Experience ReplayでQ関数を更新する
            self.agent.update_q_function()

            # init nextMove
            self.nextMove = None

            self.state_property = self.state_property_next

            # update window
            self.updateWindow()

            # 終了時の処理
            if self.done:
                # DDQNで追加、2試行に1度、Target Q-NetworkをMainと同じにコピーする
                if(self.episode % 2 == 0):
                    self.agent.update_target_q_function()    
            else:
                self.step += 1

        else:
            super(Game_Manager, self).timerEvent(event)

    def UpdateScore(self, removedlines, dropdownlines):
        # calculate and update current score
        if removedlines == 1:
            linescore = Game_Manager.LINE_SCORE_1
        elif removedlines == 2:
            linescore = Game_Manager.LINE_SCORE_2
        elif removedlines == 3:
            linescore = Game_Manager.LINE_SCORE_3
        elif removedlines == 4:
            linescore = Game_Manager.LINE_SCORE_4
        else:
            linescore = 0
        dropdownscore = dropdownlines
        self.tboard.dropdownscore += dropdownscore
        self.tboard.linescore += linescore
        self.tboard.score += ( linescore + dropdownscore )
        self.tboard.line += removedlines
        if removedlines > 0:
            self.tboard.line_score_stat[removedlines - 1] += 1

    def getGameStatus(self):
        # return current Board status.
        # define status data.
        status = {"field_info":
                      {
                        "width": "none",
                        "height": "none",
                        "backboard": "none",
                        "withblock": "none", # back board with current block
                      },
                  "block_info":
                      {
                        "currentX":"none",
                        "currentY":"none",
                        "currentDirection":"none",
                        "currentShape":{
                           "class":"none",
                           "index":"none",
                           "direction_range":"none",
                        },
                        "nextShape":{
                           "class":"none",
                           "index":"none",
                           "direction_range":"none",
                        },
                      },
                  "judge_info":
                      {
                        "elapsed_time":"none",
                        "game_time":"none",
                        "gameover_count":"none",
                        "score":"none",
                        "line":"none",
                        "block_index":"none",
                        "episode":"none",
                        "step":"none",
                      },
                  "debug_info":
                      {
                        "dropdownscore":"none",
                        "linescore":"none",
                        "line_score": {
                          "1":"none",
                          "2":"none",
                          "3":"none",
                          "4":"none",
                          "gameover":"none",
                        },
                        "shape_info": {
                          "shapeNone": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeI": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeL": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeJ": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeT": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeO": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeS": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeZ": {
                             "index" : "none",
                             "color" : "none",
                          },
                        },
                        "line_score_stat":"none",
                        "shape_info_stat":"none",
                      },
                  }
        # update status
        ## board
        status["field_info"]["width"] = BOARD_DATA.width
        status["field_info"]["height"] = BOARD_DATA.height
        status["field_info"]["backboard"] = BOARD_DATA.getData()
        status["field_info"]["withblock"] = BOARD_DATA.getDataWithCurrentBlock()
        ## shape
        status["block_info"]["currentX"] = BOARD_DATA.currentX
        status["block_info"]["currentY"] = BOARD_DATA.currentY
        status["block_info"]["currentDirection"] = BOARD_DATA.currentDirection
        status["block_info"]["currentShape"]["class"] = BOARD_DATA.currentShape
        status["block_info"]["currentShape"]["index"] = BOARD_DATA.currentShape.shape
        ### current shape
        if BOARD_DATA.currentShape.shape in (Shape.shapeI, Shape.shapeZ, Shape.shapeS):
            Range = (0, 1)
        elif BOARD_DATA.currentShape.shape == Shape.shapeO:
            Range = (0,)
        else:
            Range = (0, 1, 2, 3)
        status["block_info"]["currentShape"]["direction_range"] = Range
        ### next shape
        status["block_info"]["nextShape"]["class"] = BOARD_DATA.nextShape
        status["block_info"]["nextShape"]["index"] = BOARD_DATA.nextShape.shape
        if BOARD_DATA.nextShape.shape in (Shape.shapeI, Shape.shapeZ, Shape.shapeS):
            Range = (0, 1)
        elif BOARD_DATA.nextShape.shape == Shape.shapeO:
            Range = (0,)
        else:
            Range = (0, 1, 2, 3)
        status["block_info"]["nextShape"]["direction_range"] = Range
        ## judge_info
        status["judge_info"]["elapsed_time"] = round(time.time() - self.tboard.start_time, 3)
        status["judge_info"]["game_time"] = self.game_time
        status["judge_info"]["gameover_count"] = self.tboard.reset_cnt
        status["judge_info"]["score"] = self.tboard.score
        status["judge_info"]["line"] = self.tboard.line
        status["judge_info"]["block_index"] = self.block_index
        status["judge_info"]["episode"] = self.episode
        status["judge_info"]["step"] = self.step

        ## debug_info
        status["debug_info"]["dropdownscore"] = self.tboard.dropdownscore
        status["debug_info"]["linescore"] = self.tboard.linescore
        status["debug_info"]["line_score_stat"] = self.tboard.line_score_stat
        status["debug_info"]["shape_info_stat"] = BOARD_DATA.shape_info_stat
        status["debug_info"]["line_score"]["1"] = Game_Manager.LINE_SCORE_1
        status["debug_info"]["line_score"]["2"] = Game_Manager.LINE_SCORE_2
        status["debug_info"]["line_score"]["3"] = Game_Manager.LINE_SCORE_3
        status["debug_info"]["line_score"]["4"] = Game_Manager.LINE_SCORE_4
        status["debug_info"]["line_score"]["gameover"] = Game_Manager.GAMEOVER_SCORE
        status["debug_info"]["shape_info"]["shapeNone"]["index"] = Shape.shapeNone
        status["debug_info"]["shape_info"]["shapeI"]["index"] = Shape.shapeI
        status["debug_info"]["shape_info"]["shapeI"]["color"] = "red"
        status["debug_info"]["shape_info"]["shapeL"]["index"] = Shape.shapeL
        status["debug_info"]["shape_info"]["shapeL"]["color"] = "green"
        status["debug_info"]["shape_info"]["shapeJ"]["index"] = Shape.shapeJ
        status["debug_info"]["shape_info"]["shapeJ"]["color"] = "purple"
        status["debug_info"]["shape_info"]["shapeT"]["index"] = Shape.shapeT
        status["debug_info"]["shape_info"]["shapeT"]["color"] = "gold"
        status["debug_info"]["shape_info"]["shapeO"]["index"] = Shape.shapeO
        status["debug_info"]["shape_info"]["shapeO"]["color"] = "pink"
        status["debug_info"]["shape_info"]["shapeS"]["index"] = Shape.shapeS
        status["debug_info"]["shape_info"]["shapeS"]["color"] = "blue"
        status["debug_info"]["shape_info"]["shapeZ"]["index"] = Shape.shapeZ
        status["debug_info"]["shape_info"]["shapeZ"]["color"] = "yellow"
        if BOARD_DATA.currentShape == Shape.shapeNone:
            print("warning: current shape is none !!!")

        return status

    def getGameStatusJson(self):
        status = {
                  "debug_info":
                      {
                        "line_score": {
                          "1":"none",
                          "2":"none",
                          "3":"none",
                          "4":"none",
                          "gameover":"none",
                        },
                        "shape_info": {
                          "shapeNone": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeI": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeL": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeJ": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeT": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeO": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeS": {
                             "index" : "none",
                             "color" : "none",
                          },
                          "shapeZ": {
                             "index" : "none",
                             "color" : "none",
                          },
                        },
                        "line_score_stat":"none",
                        "shape_info_stat":"none",
                      },
                  "judge_info":
                      {
                        "elapsed_time":"none",
                        "game_time":"none",
                        "gameover_count":"none",
                        "score":"none",
                        "line":"none",
                        "block_index":"none",
                        "episode":"none",
                        "step":"none",
                      },
                  }
        # update status
        ## debug_info
        status["debug_info"]["line_score_stat"] = self.tboard.line_score_stat
        status["debug_info"]["shape_info_stat"] = BOARD_DATA.shape_info_stat
        status["debug_info"]["line_score"]["1"] = Game_Manager.LINE_SCORE_1
        status["debug_info"]["line_score"]["2"] = Game_Manager.LINE_SCORE_2
        status["debug_info"]["line_score"]["3"] = Game_Manager.LINE_SCORE_3
        status["debug_info"]["line_score"]["4"] = Game_Manager.LINE_SCORE_4
        status["debug_info"]["line_score"]["gameover"] = Game_Manager.GAMEOVER_SCORE
        status["debug_info"]["shape_info"]["shapeNone"]["index"] = Shape.shapeNone
        status["debug_info"]["shape_info"]["shapeI"]["index"] = Shape.shapeI
        status["debug_info"]["shape_info"]["shapeI"]["color"] = "red"
        status["debug_info"]["shape_info"]["shapeL"]["index"] = Shape.shapeL
        status["debug_info"]["shape_info"]["shapeL"]["color"] = "green"
        status["debug_info"]["shape_info"]["shapeJ"]["index"] = Shape.shapeJ
        status["debug_info"]["shape_info"]["shapeJ"]["color"] = "purple"
        status["debug_info"]["shape_info"]["shapeT"]["index"] = Shape.shapeT
        status["debug_info"]["shape_info"]["shapeT"]["color"] = "gold"
        status["debug_info"]["shape_info"]["shapeO"]["index"] = Shape.shapeO
        status["debug_info"]["shape_info"]["shapeO"]["color"] = "pink"
        status["debug_info"]["shape_info"]["shapeS"]["index"] = Shape.shapeS
        status["debug_info"]["shape_info"]["shapeS"]["color"] = "blue"
        status["debug_info"]["shape_info"]["shapeZ"]["index"] = Shape.shapeZ
        status["debug_info"]["shape_info"]["shapeZ"]["color"] = "yellow"
        ## judge_info
        status["judge_info"]["elapsed_time"] = round(time.time() - self.tboard.start_time, 3)
        status["judge_info"]["game_time"] = self.game_time
        status["judge_info"]["gameover_count"] = self.tboard.reset_cnt
        status["judge_info"]["score"] = self.tboard.score
        status["judge_info"]["line"] = self.tboard.line
        status["judge_info"]["block_index"] = self.block_index
        status["judge_info"]["episode"] = self.episode
        status["judge_info"]["step"] = self.step
        return json.dumps(status)

    def keyPressEvent(self, event):
        # for manual control

        if not self.isStarted or BOARD_DATA.currentShape == Shape.shapeNone:
            super(Game_Manager, self).keyPressEvent(event)
            return

        key = event.key()
        
        # key event handle process.
        # depends on self.manual, it's better to make key config file.
        #  "y" : PC keyboard controller
        #  "g" : game controller. KeyUp, space are different from "y"

        if key == Qt.Key_P:
            self.pause()
            return
            
        if self.isPaused:
            return
        elif key == Qt.Key_Left:
            BOARD_DATA.moveLeft()
        elif key == Qt.Key_Right:
            BOARD_DATA.moveRight()
        elif (key == Qt.Key_Up and self.manual == 'y') or (key == Qt.Key_Space and self.manual == 'g'):
            BOARD_DATA.rotateLeft()
        elif key == Qt.Key_M:
            removedlines, movedownlines = BOARD_DATA.moveDown()
            self.UpdateScore(removedlines, 0)
        elif (key == Qt.Key_Space and self.manual == 'y') or (key == Qt.Key_Up and self.manual == 'g'):
            removedlines, dropdownlines = BOARD_DATA.dropDown()
            self.UpdateScore(removedlines, dropdownlines)
        else:
            super(Game_Manager, self).keyPressEvent(event)

        self.updateWindow()


def drawSquare(painter, x, y, val, s):
    colorTable = [0x000000, 0xCC6666, 0x66CC66, 0x6666CC,
                  0xCCCC66, 0xCC66CC, 0x66CCCC, 0xDAAA00]

    if val == 0:
        return

    color = QColor(colorTable[val])
    painter.fillRect(x + 1, y + 1, s - 2, s - 2, color)

    painter.setPen(color.lighter())
    painter.drawLine(x, y + s - 1, x, y)
    painter.drawLine(x, y, x + s - 1, y)

    painter.setPen(color.darker())
    painter.drawLine(x + 1, y + s - 1, x + s - 1, y + s - 1)
    painter.drawLine(x + s - 1, y + s - 1, x + s - 1, y + 1)


class SidePanel(QFrame):
    def __init__(self, parent, gridSize):
        super().__init__(parent)
        self.setFixedSize(gridSize * 5, gridSize * BOARD_DATA.height)
        self.move(gridSize * BOARD_DATA.width, 0)
        self.gridSize = gridSize

    def updateData(self):
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        minX, maxX, minY, maxY = BOARD_DATA.nextShape.getBoundingOffsets(0)

        dy = 3 * self.gridSize
        dx = (self.width() - (maxX - minX) * self.gridSize) / 2

        val = BOARD_DATA.nextShape.shape
        for x, y in BOARD_DATA.nextShape.getCoords(0, 0, -minY):
            drawSquare(painter, x * self.gridSize + dx, y * self.gridSize + dy, val, self.gridSize)


class Board(QFrame):
    msg2Statusbar = pyqtSignal(str)

    def __init__(self, parent, gridSize, game_time, random_seed, obstacle_height, obstacle_probability):
        super().__init__(parent)
        self.setFixedSize(gridSize * BOARD_DATA.width, gridSize * BOARD_DATA.height)
        self.gridSize = gridSize
        self.game_time = game_time
        self.initBoard(random_seed, obstacle_height, obstacle_probability)

    def initBoard(self, random_seed_Nextshape, obstacle_height, obstacle_probability):
        self.score = 0
        self.dropdownscore = 0
        self.linescore = 0
        self.line = 0
        self.line_score_stat = [0, 0, 0, 0]
        self.reset_cnt = 0
        self.start_time = time.time() 
        BOARD_DATA.clear()
        BOARD_DATA.init_randomseed(random_seed_Nextshape)
        BOARD_DATA.init_obstacle_parameter(obstacle_height, obstacle_probability)

    def paintEvent(self, event):
        painter = QPainter(self)

        # Draw backboard
        for x in range(BOARD_DATA.width):
            for y in range(BOARD_DATA.height):
                val = BOARD_DATA.getValue(x, y)
                drawSquare(painter, x * self.gridSize, y * self.gridSize, val, self.gridSize)

        # Draw current shape
        for x, y in BOARD_DATA.getCurrentShapeCoord():
            val = BOARD_DATA.currentShape.shape
            drawSquare(painter, x * self.gridSize, y * self.gridSize, val, self.gridSize)

        # Draw a border
        painter.setPen(QColor(0x777777))
        painter.drawLine(self.width()-1, 0, self.width()-1, self.height())
        painter.setPen(QColor(0xCCCCCC))
        painter.drawLine(self.width(), 0, self.width(), self.height())

    def updateData(self):
        score_str = str(self.score)
        line_str = str(self.line)
        reset_cnt_str = str(self.reset_cnt)
        elapsed_time = round(time.time() - self.start_time, 3)
        elapsed_time_str = str(elapsed_time)
        status_str = "score:" + score_str + ",line:" + line_str + ",gameover:" + reset_cnt_str + ",time[s]:" + elapsed_time_str
        # print string to status bar
        self.msg2Statusbar.emit(status_str)
        self.update()

        #if self.game_time >= 0 and elapsed_time > self.game_time:
        if GAME_MANEGER.step >= 180:
            # finish game.
            print("game finish!! elapsed time: " + elapsed_time_str + "/game_time: " + str(self.game_time) + "/step: " + str(GAME_MANEGER.step))
            print("")
            print("##### YOUR_RESULT #####")
            print(status_str)
            print("")
            print("##### SCORE DETAIL #####")
            GameStatus = GAME_MANEGER.getGameStatus()
            line_score_stat = GameStatus["debug_info"]["line_score_stat"]
            line_Score = GameStatus["debug_info"]["line_score"]
            gameover_count = GameStatus["judge_info"]["gameover_count"]
            score = GameStatus["judge_info"]["score"]
            dropdownscore = GameStatus["debug_info"]["dropdownscore"]
            print("  1 line: " + str(line_Score["1"]) + " * " + str(line_score_stat[0]) + " = " + str(line_Score["1"] * line_score_stat[0]))
            print("  2 line: " + str(line_Score["2"]) + " * " + str(line_score_stat[1]) + " = " + str(line_Score["2"] * line_score_stat[1]))
            print("  3 line: " + str(line_Score["3"]) + " * " + str(line_score_stat[2]) + " = " + str(line_Score["3"] * line_score_stat[2]))
            print("  4 line: " + str(line_Score["4"]) + " * " + str(line_score_stat[3]) + " = " + str(line_Score["4"] * line_score_stat[3]))
            print("  dropdownscore: " + str(dropdownscore))
            print("  gameover: : " + str(line_Score["gameover"]) + " * " + str(gameover_count) + " = " + str(line_Score["gameover"] * gameover_count))

            print("##### ###### #####")
            print("")

            log_file_path = GAME_MANEGER.resultlogjson
            if len(log_file_path) != 0:
                with open(log_file_path, "w") as f:
                    print("##### OUTPUT_RESULT_LOG_FILE #####")
                    print(log_file_path)
                    GameStatusJson = GAME_MANEGER.getGameStatusJson()
                    f.write(GameStatusJson)
                    
            GAME_MANEGER.done = True
            GAME_MANEGER.reset_episode()
            #sys.exit(app.exec_())
            #sys.exit(0)

if __name__ == '__main__':
    app = QApplication([])
    GAME_MANEGER = Game_Manager()
    sys.exit(app.exec_())
