#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QMainWindow, QFrame, QDesktopWidget, QApplication, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt, QBasicTimer, pyqtSignal
from PyQt5.QtGui import QPainter, QColor

from board_manager import BOARD_DATA, Shape
from block_controller import BLOCK_CONTROLLER
from block_controller_sample import BLOCK_CONTROLLER_SAMPLE

from argparse import ArgumentParser
import time
import json
import pprint

import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import namedtuple
import numpy as np

from datetime import datetime
import pprint

BATCH_SIZE = 32
CAPACITY = 10000
num_states = 220 # withblock 22 x 10 +3
num_actions = 40 #x軸10、回転4
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))
GAMMA = 0.95  # 時間割引率
MAX_STEPS = 200  # 1試行のstep数
NUM_EPISODES = 500  # 最大試行回数

EPISODE = 10
episode_final = False
observation = None
state_ai = None

# a[n] = n^2 - n + 1
LINE_SCORE_1 = 100
LINE_SCORE_2 = 300
LINE_SCORE_3 = 700
LINE_SCORE_4 = 1300
GAMEOVER_SCORE = -500

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


class Brain:

    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  # CartPoleの行動（右に左に押す）の2を取得

        self.epsilon = 1.0

        # 経験を記憶するメモリオブジェクトを生成
        self.memory = ReplayMemory(CAPACITY)

        # ニューラルネットワークを構築
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, 32))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(32, 32))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32, num_actions))

        print(self.model)  # ネットワークの形を出力

        # 最適化手法の設定
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def replay(self):
        '''Experience Replayでネットワークの結合パラメータを学習'''

        # -----------------------------------------
        # 1. メモリサイズの確認
        # -----------------------------------------
        # 1.1 メモリサイズがミニバッチより小さい間は何もしない
        if len(self.memory) < BATCH_SIZE:
            return

        # -----------------------------------------
        # 2. ミニバッチの作成
        # -----------------------------------------
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

        # -----------------------------------------
        # 3. 教師信号となるQ(s_t, a_t)値を求める
        # -----------------------------------------
        # 3.1 ネットワークを推論モードに切り替える
        self.model.eval()

        # 3.2 ネットワークが出力したQ(s_t, a_t)を求める
        # self.model(state_batch)は、右左の両方のQ値を出力しており
        # [torch.FloatTensor of size BATCH_SIZEx2]になっている。
        # ここから実行したアクションa_tに対応するQ値を求めるため、action_batchで行った行動a_tが右か左かのindexを求め
        # それに対応するQ値をgatherでひっぱり出す。
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # 3.3 max{Q(s_t+1, a)}値を求める。ただし次の状態があるかに注意
        # cartpoleがdoneになっておらず、next_stateがあるかをチェックするインデックスマスクを作成
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)))
        # まずは全部0にしておく
        next_state_values = torch.zeros(BATCH_SIZE)

        # 次の状態があるindexの最大Q値を求める
        # 出力にアクセスし、max(1)で列方向の最大値の[値、index]を求めます
        # そしてそのQ値（index=0）を出力します
        # detachでその値を取り出します
        next_state_values[non_final_mask] = self.model(
            non_final_next_states).max(1)[0].detach()

        # 3.4 教師となるQ(s_t, a_t)値を、Q学習の式から求める
        expected_state_action_values = reward_batch + GAMMA * next_state_values

        # -----------------------------------------
        # 4. 結合パラメータの更新
        # -----------------------------------------
        # 4.1 ネットワークを訓練モードに切り替える
        self.model.train()

        # 4.2 損失関数を計算する（smooth_l1_lossはHuberloss）
        # expected_state_action_valuesは
        # sizeが[minbatch]になっているので、unsqueezeで[minibatch x 1]へ
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        # 4.3 結合パラメータを更新する
        self.optimizer.zero_grad()  # 勾配をリセット
        loss.backward()  # バックプロパゲーションを計算
        self.optimizer.step()  # 結合パラメータを更新

    def decide_action(self, state, episode):
        '''現在の状態に応じて、行動を決定する'''
        # ε-greedy法で徐々に最適行動のみを採用する
        # epsilon = 0.5 * (1 / (episode + 1))
        epsilon = 1.0 - (episode+1)/1000000
        print("episilon")
        print(epsilon)

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()  # ネットワークを推論モードに切り替える
            with torch.no_grad():
                #action = self.model(state).max(1)[1].view(1, 1)
                print("decide_action state")
                print(state)
                action = self.model(state)
                print("decide_action action")
                print(action)
                action = action.max(1)[1].view(1,1)
                print("decide_action max")
                print(action)
            # ネットワークの出力の最大値のindexを取り出します = max(1)[1]
            # .view(1,1)は[torch.LongTensor of size 1]　を size 1x1 に変換します

        else:
            # 0,1の行動をランダムに返す
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]])  # 0,14の行動をランダムに返す
            # actionは[torch.LongTensor of size 1x1]の形になります

        return action


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

class Block_Controller_AI(object):

    # init parameter
    board_backboard = 0
    board_data_width = 0
    board_data_height = 0
    ShapeNone_index = 0
    CurrentShape_class = 0
    NextShape_class = 0

    agent = Agent(num_states, num_actions)

    # GetNextMove is main function.
    # input
    #    GameStatus : this data include all field status, 
    #                 in detail see the internal GameStatus data.
    # output
    #    nextMove : this data include next shape position and the other,
    #               if return None, do nothing to nextMove.
    def GetNextMove(self, nextMove, GameStatus, state):

        t1 = datetime.now()

        # print GameStatus
        print("=================================================>")
        pprint.pprint(GameStatus, width = 61, compact = True)

        print("getnextmove state")
        print(state)
        print("episode")
        print(GAME_MANEGER.episode)
        action = self.agent.get_action(state, GAME_MANEGER.episode)
        action_item = action.item()

        # search best nextMove -->
        # random sample
        
        nextMove["strategy"]["direction"] = action_item % 4
        nextMove["strategy"]["x"] = action_item % 10
            
        nextMove["strategy"]["y_operation"] = 1
        nextMove["strategy"]["y_moveblocknum"] = 1

        # search best nextMove <--

        # return nextMove
        print("===", datetime.now() - t1)
        print(nextMove)
        return nextMove, action

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

    def __init__(self):
        super().__init__()
        self.agent = Agent(num_states, num_actions)
        self.episode = 0
        self.step = 0


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

        observation = BOARD_DATA.getDataWithCurrentBlock()
        state_back = BOARD_DATA.getData()
        print("observation:{0}".format(observation))
        state_ai = np.array(observation)  # 観測をそのまま状態sとして使用
        state_ai = torch.from_numpy(state_ai).type(torch.FloatTensor)  # NumPy変数をPyTorchのテンソルに変換
        state_ai = torch.unsqueeze(state_ai, 0)  # size 4をsize 1x4に変換
        print("initUI state")
        print(state_ai.size())
        print(state_ai)

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

    def reset_episode(self):
        self.tboard.score = 0
        self.tboard.dropdownscore = 0
        self.tboard.linescore = 0
        self.tboard.line = 0
        self.tboard.line_score_stat = [0, 0, 0, 0]
        self.tboard.reset_cnt = 0
        self.tboard.start_time = time.time()
        self.step = 0
        self.episode += 1
        BOARD_DATA.clear()
        BOARD_DATA.createNewPiece()
        observation = BOARD_DATA.getDataWithCurrentBlock()
        state_back = BOARD_DATA.getData()
        print("observation:{0}".format(observation))
        state_ai = np.array(observation)  # 観測をそのまま状態sとして使用
        state_ai = torch.from_numpy(state_ai).type(torch.FloatTensor)  # NumPy変数をPyTorchのテンソルに変換
        state_ai = torch.unsqueeze(state_ai, 0)  # size 4をsize 1x4に変換
        print("reset episode state")
        print(state_ai)

    def resetfield(self):
        # self.tboard.score = 0
        self.tboard.reset_cnt += 1
        self.tboard.score += GAMEOVER_SCORE
        BOARD_DATA.clear()
        BOARD_DATA.createNewPiece()

    def updateWindow(self):
        self.tboard.updateData()
        self.sidePanel.updateData()
        self.update()
        GAME_MANEGER.step += 1
        print("step:{0}".format(GAME_MANEGER.step))
        print("episode:{0}".format(GAME_MANEGER.episode))

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
        #     max_conect_dict[row] = max_conect_num
        # print(max_conect_dict) 

        #     conect_list.append()
        # print(conect_list)

    def get_piece_mask(self, board):
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

                state_back = BOARD_DATA.getData()

                state_ai = GameStatus["field_info"]["withblock"]
                state_ai = np.array(state_ai)  # 観測をそのまま状態sとして使用
                state_ai = torch.from_numpy(state_ai).type(torch.FloatTensor)  # NumPy変数をPyTorchのテンソルに変換
                state_ai = torch.unsqueeze(state_ai, 0)  # size 4をsize 1x4に変換

                if self.use_sample == "y":
                    self.nextMove, action = BLOCK_CONTROLLER_AI.GetNextMove(nextMove, GameStatus, state_ai)
                else:
                    self.nextMove = BLOCK_CONTROLLER.GetNextMove(nextMove, GameStatus)

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

            observation_next = BOARD_DATA.getData()
            state_back_next = BOARD_DATA.getData()

            self.UpdateScore(removedlines, dropdownlines)

            # elapsed_time = round(time.time() - self.start_time, 3)

            reward = 0
            # check reset field
            if BOARD_DATA.currentY < 1:
                state_next_ai = None
                # if Piece cannot movedown and stack, reset field
                print("reset field.")
                #self.resetfield()
                self.reset_episode()
                reward = torch.FloatTensor([GAMEOVER_SCORE/10])

            # elif self.game_time >= 0 and Board.updateData.elapsed_time > self.game_time and GameStatus["judge_info"]["gameover_count"] == 0:
            #     state_next_ai = None
            #     reward = torch.FloatTensor([1.0])

            elif self.step%10 == 0 :
                state_next_ai = np.array(observation_next)
                state_next_ai = torch.from_numpy(state_next_ai).type(torch.FloatTensor)
                state_next_ai = torch.unsqueeze(state_next_ai, 0)
                reward = torch.FloatTensor([self.step/100])

            elif removedlines > 0:
                state_next_ai = np.array(observation_next)
                state_next_ai = torch.from_numpy(state_next_ai).type(torch.FloatTensor)
                state_next_ai = torch.unsqueeze(state_next_ai, 0)

                if removedlines == 1:
                    linescore = LINE_SCORE_1
                elif removedlines == 2:
                    linescore = LINE_SCORE_2
                elif removedlines == 3:
                    linescore = LINE_SCORE_3
                elif removedlines == 4:
                    linescore = LINE_SCORE_4

                reward = torch.FloatTensor([linescore/10])
            
            else:
                state_next_ai = np.array(observation_next)
                state_next_ai = torch.from_numpy(state_next_ai).type(torch.FloatTensor)
                state_next_ai = torch.unsqueeze(state_next_ai, 0)
                reward = torch.FloatTensor([0.0])

            conect_num = self.get_row_connection(state_back)
            conect_next_num = self.get_row_connection(state_back_next)
            only_put_piece = np.array(state_back_next) - np.array(state_back)
            only_put_piece_mask = only_put_piece != 0
            only_put_piece_around_mask = self.get_piece_mask(only_put_piece)
            
            hoge = only_put_piece_around_mask - only_put_piece_mask.astype(np.int).reshape(22, 10)
            rinsetu = hoge * np.array(state_back_next).reshape(22, 10)
            print(hoge)
            flag = rinsetu.sum()
            print("flag")
            print(flag)

            if(flag > 0):
                conect_diff = conect_next_num - conect_num
                reward += torch.FloatTensor([conect_diff])
                print("conect diff")
                print(conect_diff)

            # else:
            #     conect_put_num = self.get_row_connection(only_put_piece)

            print("conect num")
            print(conect_num)
            print("conect next num")
            print(conect_next_num)
            


            # hole_num = self.get_holes(state_back)
            # print("hole_num")
            # print(hole_num)
            # next_hole_num = self.get_holes(state_back_next)
            # print("next_hole_num")
            # print(next_hole_num)

            # if(hole_num - next_hole_num < 0):
            #     reward += torch.FloatTensor([-10])
            # elif(hole_num - next_hole_num > 0):
            #     reward += torch.FloatTensor([10])

            bump, heig = self.get_bumpiness_and_height(state_back_next)
            # print("bump")
            # print(bump)
            # print("height")
            # print(heig)
            if (state_back_next != None):
                print(np.array(state_back_next).reshape(22, 10))

            self.get_row_connection(state_back_next)
            # print("state_ai")
            # print(np.array(state_ai).reshape(22, 10))
            # print("action")
            # print(action)
            # print("state_next_ai")
            # if(state_next_ai != None):
            #     print(np.array(state_next_ai).reshape(22, 10))
            print("reward")
            print(reward)
            self.agent.memorize(state_ai, action, state_next_ai, reward)

            self.agent.update_q_function()

            state_ai = state_next_ai

            # init nextMove
            self.nextMove = None

            # update window
            self.updateWindow()
        else:
            super(Game_Manager, self).timerEvent(event)

    def UpdateScore(self, removedlines, dropdownlines):
        # calculate and update current score
        if removedlines == 1:
            linescore = LINE_SCORE_1
        elif removedlines == 2:
            linescore = LINE_SCORE_2
        elif removedlines == 3:
            linescore = LINE_SCORE_3
        elif removedlines == 4:
            linescore = LINE_SCORE_4
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
        ## debug_info
        status["debug_info"]["dropdownscore"] = self.tboard.dropdownscore
        status["debug_info"]["linescore"] = self.tboard.linescore
        status["debug_info"]["line_score_stat"] = self.tboard.line_score_stat
        status["debug_info"]["shape_info_stat"] = BOARD_DATA.shape_info_stat
        status["debug_info"]["line_score"]["1"] = LINE_SCORE_1
        status["debug_info"]["line_score"]["2"] = LINE_SCORE_2
        status["debug_info"]["line_score"]["3"] = LINE_SCORE_3
        status["debug_info"]["line_score"]["4"] = LINE_SCORE_4
        status["debug_info"]["line_score"]["gameover"] = GAMEOVER_SCORE
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
                      },
                  }
        # update status
        ## debug_info
        status["debug_info"]["line_score_stat"] = self.tboard.line_score_stat
        status["debug_info"]["shape_info_stat"] = BOARD_DATA.shape_info_stat
        status["debug_info"]["line_score"]["1"] = LINE_SCORE_1
        status["debug_info"]["line_score"]["2"] = LINE_SCORE_2
        status["debug_info"]["line_score"]["3"] = LINE_SCORE_3
        status["debug_info"]["line_score"]["4"] = LINE_SCORE_4
        status["debug_info"]["line_score"]["gameover"] = GAMEOVER_SCORE
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

        if self.game_time >= 0 and elapsed_time > self.game_time:
            # finish game.
            print("game finish!! elapsed time: " + elapsed_time_str + "/game_time: " + str(self.game_time))
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

            #sys.exit(app.exec_())
            #sys.exit(0)
            GAME_MANEGER.reset_episode()
            GAME_MANEGER.episode += 1
            GAME_MANEGER.step = 0
            print("episode:{0}".format(GAME_MANEGER.episode))
            # if(GAME_MANEGER.episode == EPISODE):
            #     sys.exit(0)

if __name__ == '__main__':
    app = QApplication([])
    GAME_MANEGER = Game_Manager()
    BLOCK_CONTROLLER_AI = Block_Controller_AI()
    sys.exit(app.exec_())
