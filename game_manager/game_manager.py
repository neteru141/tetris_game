#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QMainWindow, QFrame, QDesktopWidget, QApplication, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt, QBasicTimer, pyqtSignal
from PyQt5.QtGui import QPainter, QColor

from board_manager import BOARD_DATA, Shape
from block_controller import BLOCK_CONTROLLER
from block_controller_next_steps import BLOCK_CONTROLLER_NEXT_STEP

from argparse import ArgumentParser
import time
import json
import pprint

import torch
import torch.nn as nn
from deep_q_network import DeepQNetwork
from collections import deque
from tensorboardX import SummaryWriter

import os
import shutil
from random import random, randint, sample
import numpy as np

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
    argparser.add_argument("--width", type=int, default=10, help="The common width for all images")
    argparser.add_argument("--height", type=int, default=20, help="The common height for all images")
    argparser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    argparser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    argparser.add_argument("--lr", type=float, default=1e-3)
    argparser.add_argument("--gamma", type=float, default=0.99)
    argparser.add_argument("--initial_epsilon", type=float, default=1)
    argparser.add_argument("--final_epsilon", type=float, default=1e-3)
    argparser.add_argument("--num_decay_epochs", type=float, default=2000)
    argparser.add_argument("--num_epochs", type=int, default=3000)
    argparser.add_argument("--save_interval", type=int, default=1000)
    argparser.add_argument("--replay_memory_size", type=int, default=30000, help="Number of epoches between testing phases")
    argparser.add_argument("--log_path", type=str, default="tensorboard")
    argparser.add_argument("--saved_path", type=str, default="trained_models")
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

        self.width = args.width
        self.heigth = args.height
        self.block_size = args.block_size
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.gamma = args.gamma
        self.initial_epsilon = args.initial_epsilon
        self.final_epsilon = args.final_epsilon
        self.num_decay_epochs = args.num_decay_epochs
        self.num_epochs = args.num_epochs
        self.save_interval = args.save_interval
        self.replay_memory_size = args.replay_memory_size
        self.log_path = args.log_path
        self.saved_path = args.saved_path
        
        self.episode = 0
        self.step = 0
        self.num_states = 4
        self.num_actions = 1
        self.model = DeepQNetwork()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.replay_memory = deque(maxlen=self.replay_memory_size)

        self.init_state_flag = True

        self.state = None
        self.next_state = None
        self.action = None
        self.reward = None

        if os.path.isdir(self.log_path):
            shutil.rmtree(self.log_path)
        os.makedirs(self.log_path)
        self.writer = SummaryWriter(self.log_path)

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

                GameStatus = self.getGameStatus()
                done = False

                print("### step ###")
                print(self.step)
                print("### episode ###")
                print(self.episode)

                if(self.init_state_flag == True):
                    _, self.state = BLOCK_CONTROLLER_NEXT_STEP.GetNextMoveState(GameStatus)
                    self.state = np.array(self.state)
                    self.state = torch.from_numpy(self.state).type(torch.FloatTensor)
                    
                    self.init_state_flag = False

                next_actions, next_states = BLOCK_CONTROLLER_NEXT_STEP.GetNextMoveState(GameStatus)
                next_actions = np.array(next_actions)
                next_actions = torch.from_numpy(next_actions).type(torch.FloatTensor)
                next_states = np.array(next_states)
                next_states = torch.from_numpy(next_states).type(torch.FloatTensor)

                print("### next_actions ###")
                print(next_actions)

                print("### next_states ###")
                print(next_states)

                self.model.eval()
                with torch.no_grad():
                    predictions = self.model(next_states)[:, 0]
                    print("### predictions ###")
                    print(predictions)

                epsilon = self.final_epsilon + (max(self.num_decay_epochs - self.episode, 0) * (self.initial_epsilon - self.final_epsilon) / self.num_decay_epochs)
                print("### epsilon ###")
                print(epsilon)
                u = random()
                random_action = u <= epsilon

                self.model.train()
                if random_action:
                    print("### len(next states) ###")
                    print(len(next_states))
                    index = randint(0, len(next_states) - 1)
                else:
                    index = torch.argmax(predictions).item()

                self.next_state = next_states[index, :]
                print("### self.next_state ###")
                print(self.next_state)
                self.action = next_actions[index]
                print("### self.action ###")
                print(self.action) # (rotation, position)

                self.nextMove = BLOCK_CONTROLLER.GetNextMove(GameStatus, nextMove, self.action)

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

            self.reward = 0

            if BOARD_DATA.currentY < 1:
                self.reward = torch.FloatTensor([Game_Manager.GAMEOVER_SCORE])
                done = True

            elif removedlines > 0:
                if removedlines == 1:
                    linescore = Game_Manager.LINE_SCORE_1
                elif removedlines == 2:
                    linescore = Game_Manager.LINE_SCORE_2
                elif removedlines == 3:
                    linescore = Game_Manager.LINE_SCORE_3
                elif removedlines == 4:
                    linescore = Game_Manager.LINE_SCORE_4

                self.reward = torch.FloatTensor([linescore])

                
            self.replay_memory.append([self.state, self.reward, self.next_state, done])
            
            if done:
                print("reset episode")
                self.reset_episode()

                final_score = GameStatus["judge_info"]["score"]
                final_tetrominoes = self.step
                final_cleared_lines = GameStatus["judge_info"]["line"]

                GameStatus = self.getGameStatus()
                _, self.state = BLOCK_CONTROLLER_NEXT_STEP.GetNextMoveState(GameStatus)
                self.state = np.array(self.state)
                self.state = torch.from_numpy(self.state).type(torch.FloatTensor)

                if len(self.replay_memory) < self.replay_memory_size / 10:
                    pass
                else:

                    batch = sample(self.replay_memory, min(len(self.replay_memory), self.batch_size))
                    state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
                    state_batch = torch.stack(tuple(self.state for self.state in state_batch))
                    reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
                    next_state_batch = torch.stack(tuple(self.state for self.state in next_state_batch))

                    q_values = self.model(state_batch)
                    print("### q_values ###")
                    print(q_values)
                    self.model.eval()
                    with torch.no_grad():
                        next_prediction_batch = self.model(next_state_batch)
                        print("### next prediction batch ###")
                        print(next_prediction_batch)
                    self.model.train()

                    y_batch = torch.cat(
                        tuple(self.reward if done else self.reward + self.gamma * prediction for self.reward, done, prediction in
                            zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

                    print("### y_batch ###")
                    print(y_batch)

                    optimizer.zero_grad()
                    loss = criterion(q_values, y_batch)
                    print("### loss ###")
                    print(loss)
                    loss.backward()
                    optimizer.step()

                    print("Epoch: {}/{}, Action: {}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
                        epoch,
                        self.num_epochs,
                        self.action,
                        final_score,
                        final_tetrominoes,
                        final_cleared_lines))
                    writer.add_scalar('Train/Score', final_score, self.episode - 1)
                    writer.add_scalar('Train/Tetrominoes', final_tetrominoes, self.episode - 1)
                    writer.add_scalar('Train/Cleared lines', final_cleared_lines, self.episode - 1)

                    if self.episode > 0 and self.episode % self.save_interval == 0:
                        torch.save(self.model, "{}/tetris_{}".format(self.saved_path, self.episode))
            else:
                self.state = self.next_state

            # init nextMove
            self.nextMove = None

            # step count up
            self.step += 1

            # update window
            self.updateWindow()

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
        # if GAME_MANEGER.step >= 180:
        #     # finish game.
        #     print("game finish!! elapsed time: " + elapsed_time_str + "/game_time: " + str(self.game_time) + "/step: " + str(GAME_MANEGER.step))
        #     print("")
        #     print("##### YOUR_RESULT #####")
        #     print(status_str)
        #     print("")
        #     print("##### SCORE DETAIL #####")
        #     GameStatus = GAME_MANEGER.getGameStatus()
        #     line_score_stat = GameStatus["debug_info"]["line_score_stat"]
        #     line_Score = GameStatus["debug_info"]["line_score"]
        #     gameover_count = GameStatus["judge_info"]["gameover_count"]
        #     score = GameStatus["judge_info"]["score"]
        #     dropdownscore = GameStatus["debug_info"]["dropdownscore"]
        #     print("  1 line: " + str(line_Score["1"]) + " * " + str(line_score_stat[0]) + " = " + str(line_Score["1"] * line_score_stat[0]))
        #     print("  2 line: " + str(line_Score["2"]) + " * " + str(line_score_stat[1]) + " = " + str(line_Score["2"] * line_score_stat[1]))
        #     print("  3 line: " + str(line_Score["3"]) + " * " + str(line_score_stat[2]) + " = " + str(line_Score["3"] * line_score_stat[2]))
        #     print("  4 line: " + str(line_Score["4"]) + " * " + str(line_score_stat[3]) + " = " + str(line_Score["4"] * line_score_stat[3]))
        #     print("  dropdownscore: " + str(dropdownscore))
        #     print("  gameover: : " + str(line_Score["gameover"]) + " * " + str(gameover_count) + " = " + str(line_Score["gameover"] * gameover_count))

        #     print("##### ###### #####")
        #     print("")

        #     log_file_path = GAME_MANEGER.resultlogjson
        #     if len(log_file_path) != 0:
        #         with open(log_file_path, "w") as f:
        #             print("##### OUTPUT_RESULT_LOG_FILE #####")
        #             print(log_file_path)
        #             GameStatusJson = GAME_MANEGER.getGameStatusJson()
        #             f.write(GameStatusJson)
                    
        #     GAME_MANEGER.done = True
        #     GAME_MANEGER.reset_episode()
            #sys.exit(app.exec_())
            #sys.exit(0)

if __name__ == '__main__':
    app = QApplication([])
    GAME_MANEGER = Game_Manager()
    sys.exit(app.exec_())
