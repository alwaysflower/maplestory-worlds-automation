#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MapleStory Worlds 優化自動化系統
使用 YOLO 模型進行智能物件偵測和自動化操作
版本: 2.0
作者: AI Assistant
"""

import cv2
import mss
import numpy as np
import pyautogui
import pydirectinput
import time
import os
import sys
import logging
import yaml
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from ultralytics import YOLO

# 配置日誌 (輸出到 debug/ 資料夾, 方便 push 分享診斷資訊)
os.makedirs('debug', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug/auto_system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """偵測結果數據類"""
    bbox: List[int]
    confidence: float
    class_id: int
    class_name: str
    center: Tuple[int, int]
    distance_from_center: float = 0.0

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """載入配置文件"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"配置文件 {self.config_path} 不存在，使用默認配置")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"載入配置失敗: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """獲取默認配置"""
        return {
            'model': {
                'default_path': 'weights/best.pt',
                'confidence_threshold': 0.6,
                'iou_threshold': 0.45
            },
            'window': {
                'default': {'left': 100, 'top': 100, 'width': 1200, 'height': 800}
            },
            'controls': {
                'pickup_key': 'z',
                'interact_key': 'space',
                'attack_method': 'click'
            },
            'automation': {
                'action_delay': 0.3,
                'scan_interval': 0.1,
                'max_detection_distance': 200,
                'priority_targets': ['item', 'mob', 'npc']
            },
            'safety': {
                'enable_failsafe': True,
                'max_runtime_hours': 2
            }
        }
    
    def get(self, key_path: str, default=None):
        """獲取配置值，支持點分割路徑如 'model.confidence_threshold'"""
        try:
            keys = key_path.split('.')
            value = self.config
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

class PerformanceMonitor:
    """性能監控器"""
    
    def __init__(self):
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        self.detection_times = []
        
    def update_fps(self):
        """更新 FPS 計數"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def record_detection_time(self, detection_time: float):
        """記錄偵測時間"""
        self.detection_times.append(detection_time)
        if len(self.detection_times) > 100:  # 只保留最近100次
            self.detection_times.pop(0)
    
    def get_avg_detection_time(self) -> float:
        """獲取平均偵測時間"""
        return sum(self.detection_times) / len(self.detection_times) if self.detection_times else 0

class OptimizedMapleBot:
    """優化版 MapleStory 自動化機器人"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.model = None
        self.running = False
        self.paused = False
        self.start_time = None
        self.performance_monitor = PerformanceMonitor()
        
        # 從配置載入設定
        self.monitor = self.config.get('window.default')
        self.confidence_threshold = self.config.get('model.confidence_threshold', 0.6)
        self.action_delay = self.config.get('automation.action_delay', 0.3)
        self.scan_interval = self.config.get('automation.scan_interval', 0.1)
        self.max_runtime = self.config.get('safety.max_runtime_hours', 2) * 3600
        
        # 統計數據
        self.stats = {
            'detections': 0,
            'actions_performed': 0,
            'items_collected': 0,
            'mobs_attacked': 0,
            'npcs_interacted': 0,
            'searches_performed': 0,
            'search_time_total': 0
        }
        
        # 尋找怪物相關變數
        self.last_mob_detection_time = time.time()
        self.is_searching = False
        self.search_start_time = 0
        self.original_position = None
        self.search_direction = 1  # 1 for right, -1 for left
        self.search_moves = 0

        # 追擊/朝向相關變數
        self.facing = 1  # 角色目前朝向: 1=右, -1=左
        self.player_center = None  # 最近一次偵測到的角色中心 (x, y)
        self.attack_range_px = self.config.get('detection_behavior.mob.attack_range_px', 150)
        self.same_layer_px = self.config.get('detection_behavior.mob.same_layer_px', 80)
        self.chase_enable = self.config.get('detection_behavior.mob.chase', True)
        self.chasing_key = None  # 目前持續按住追擊的方向鍵 (None=未在追擊)

        # 撿取物品相關變數
        self.item_action = self.config.get('detection_behavior.item.action', 'ignore')
        self.pickup_range_px = self.config.get('detection_behavior.item.pickup_range_px', 60)
        self.item_same_layer_px = self.config.get('detection_behavior.item.same_layer_px', 80)
        self.item_approach = self.config.get('detection_behavior.item.approach', True)

        # 設定 PyAutoGUI (滑鼠) 與 pydirectinput (鍵盤)
        if self.config.get('safety.enable_failsafe', True):
            pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05  # 減少暫停時間提升性能
        pydirectinput.PAUSE = 0.05
        pydirectinput.FAILSAFE = False  # 由 pyautogui 負責 failsafe
        
        logger.info("OptimizedMapleBot 初始化完成")
        self._load_model()
    
    def _load_model(self):
        """載入 YOLO 模型"""
        model_path = self.config.get('model.default_path')
        if not model_path or not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return False
        
        try:
            logger.info(f"載入模型: {model_path}")
            self.model = YOLO(model_path)
            self.model.conf = self.confidence_threshold
            self.model.iou = self.config.get('model.iou_threshold', 0.45)
            
            logger.info("✅ 模型載入成功!")
            logger.info(f"📊 模型類別: {self.model.names}")
            return True
            
        except Exception as e:
            logger.error(f"模型載入失敗: {e}")
            return False
    
    def capture_screen(self) -> Optional[np.ndarray]:
        """優化的螢幕擷取"""
        try:
            with mss.mss() as sct:
                screenshot = sct.grab(self.monitor)
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                return img
        except Exception as e:
            logger.error(f"螢幕擷取失敗: {e}")
            return None
    
    def detect_objects(self, img: np.ndarray) -> List[Detection]:
        """優化的物件偵測"""
        if self.model is None:
            return []
        
        start_time = time.time()
        
        try:
            results = self.model(img, conf=self.confidence_threshold,
                                  iou=self.config.get('model.iou_threshold', 0.45),
                                  verbose=False)
            detections = []
            
            # 計算畫面中心點
            center_x, center_y = self.monitor['width'] // 2, self.monitor['height'] // 2
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = box.cls[0].cpu().numpy()
                        
                        if conf > self.confidence_threshold:
                            detection_center = (int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2))
                            
                            # 計算距離中心點的距離
                            distance = np.sqrt((detection_center[0] - center_x)**2 + (detection_center[1] - center_y)**2)
                            
                            detection = Detection(
                                bbox=[int(x) for x in xyxy],
                                confidence=float(conf),
                                class_id=int(cls),
                                class_name=self.model.names[int(cls)],
                                center=detection_center,
                                distance_from_center=distance
                            )
                            detections.append(detection)
            
            # 按優先級和距離排序
            detections = self._prioritize_detections(detections)

            # 記錄角色位置 (取信賴度最高的 character), 作為距離/追擊的參考點
            chars = [d for d in detections if d.class_name == 'character']
            if chars:
                best_char = max(chars, key=lambda d: d.confidence)
                self.player_center = best_char.center

            # 記錄統計
            self.stats['detections'] += len(detections)
            detection_time = time.time() - start_time
            self.performance_monitor.record_detection_time(detection_time)
            
            return detections
            
        except Exception as e:
            logger.error(f"物件偵測失敗: {e}")
            return []
    
    def _prioritize_detections(self, detections: List[Detection]) -> List[Detection]:
        """按優先級和距離排序偵測結果"""
        priority_map = {name: i for i, name in enumerate(self.config.get('automation.priority_targets', []))}
        
        def sort_key(detection):
            priority = priority_map.get(detection.class_name, 999)
            return (priority, detection.distance_from_center)
        
        return sorted(detections, key=sort_key)
    
    def perform_action(self, detection: Detection) -> bool:
        """執行優化的遊戲動作"""
        class_name = detection.class_name

        try:
            if class_name == 'mob':
                mob_action = self.config.get('detection_behavior.mob.action', 'attack')
                if mob_action == 'attack':
                    return self._handle_mob(detection)
                else:
                    logger.info(f"👁️ 偵測到怪物 (信賴度: {detection.confidence:.2f}) - 僅記錄")

            elif class_name == 'item':
                if self.item_action == 'pickup':
                    return self._handle_item(detection)
                else:
                    logger.info(f"👁️ 偵測到物品 (信賴度: {detection.confidence:.2f}) - 僅記錄")

            elif class_name == 'npc':
                logger.info(f"👁️ 偵測到 NPC (信賴度: {detection.confidence:.2f}) - 僅記錄")

        except Exception as e:
            logger.error(f"執行動作失敗: {e}")

        return False

    def _turn_to(self, direction: int):
        """轉身面向目標 (1=右, -1=左), 僅在朝向改變時按方向鍵"""
        if self.facing == direction:
            return
        move_key = self.config.get('controls.movement_keys.right', 'right') if direction > 0 \
            else self.config.get('controls.movement_keys.left', 'left')
        pydirectinput.press(move_key)  # 點按一下轉身, 不移動位置
        self.facing = direction

    def _stop_chase(self):
        """鬆開正在按住的追擊方向鍵 (若有)"""
        if self.chasing_key is not None:
            pydirectinput.keyUp(self.chasing_key)
            self.chasing_key = None

    def _handle_mob(self, detection: Detection) -> bool:
        """攻擊/追擊三態: 範圍內攻擊, 範圍外追擊接近"""
        # 參考點: 優先用偵測到的角色中心, 否則用畫面中心
        if self.player_center is not None:
            ref_x, ref_y = self.player_center
        else:
            ref_x = self.monitor['width'] // 2
            ref_y = self.monitor['height'] // 2

        mob_x, mob_y = detection.center
        dx = mob_x - ref_x           # >0 怪在右, <0 怪在左
        horiz_dist = abs(dx)
        direction = 1 if dx > 0 else -1

        # 垂直層判斷: 角色與怪物 y 差過大視為異層, 平砍夠不到, 跳過
        vert_dist = abs(mob_y - ref_y)
        if vert_dist > self.same_layer_px:
            logger.info(f"⏭️ 跳過異層怪物 (垂直距離: {vert_dist:.0f}px > {self.same_layer_px})")
            return False

        # 範圍內: 停止追擊(鬆開方向鍵) + 轉身 + 攻擊
        if horiz_dist <= self.attack_range_px:
            self._stop_chase()
            self._turn_to(direction)
            attack_method = self.config.get('controls.attack_method', 'key')
            if attack_method == 'key':
                attack_key = self.config.get('controls.attack_key', 'a')
                pydirectinput.press(attack_key)
            else:
                abs_x = self.monitor['left'] + mob_x
                abs_y = self.monitor['top'] + mob_y
                pyautogui.moveTo(abs_x, abs_y, duration=0.1)
                pyautogui.click()
            logger.info(f"⚔️ 攻擊怪物 (信賴度: {detection.confidence:.2f}, 水平距離: {horiz_dist:.0f}px)")
            self.stats['mobs_attacked'] += 1
            self.stats['actions_performed'] += 1
            time.sleep(self.config.get('detection_behavior.mob.attack_delay', 0.5))
            return True

        # 範圍外: 持續按住方向鍵朝怪物移動 (跨幀保持, 不每幀點按, 移動連貫)
        if self.chase_enable:
            move_key = self.config.get('controls.movement_keys.right', 'right') if direction > 0 \
                else self.config.get('controls.movement_keys.left', 'left')
            # 方向改變(或尚未按住)時, 先鬆開舊鍵再按新方向; 方向不變則維持按住
            if self.chasing_key != move_key:
                self._stop_chase()
                pydirectinput.keyDown(move_key)
                self.chasing_key = move_key
            self.facing = direction
            logger.info(f"🏃 追擊怪物 ({'右' if direction > 0 else '左'}, 水平距離: {horiz_dist:.0f}px)")
            self.stats['actions_performed'] += 1
            return True

        return False

    def _handle_item(self, detection: Detection) -> bool:
        """撿取物品: 同層且範圍內按撿取鍵, 範圍外走過去接近 (結構同 _handle_mob)"""
        if self.player_center is not None:
            ref_x, ref_y = self.player_center
        else:
            ref_x = self.monitor['width'] // 2
            ref_y = self.monitor['height'] // 2

        item_x, item_y = detection.center
        dx = item_x - ref_x
        horiz_dist = abs(dx)
        direction = 1 if dx > 0 else -1

        # 垂直層判斷: 異層物品撿不到, 跳過
        vert_dist = abs(item_y - ref_y)
        if vert_dist > self.item_same_layer_px:
            logger.info(f"⏭️ 跳過異層物品 (垂直距離: {vert_dist:.0f}px > {self.item_same_layer_px})")
            return False

        # 範圍內: 停止移動 + 按撿取鍵
        if horiz_dist <= self.pickup_range_px:
            self._stop_chase()
            pickup_key = self.config.get('controls.pickup_key', 'z')
            pydirectinput.press(pickup_key)
            logger.info(f"💰 撿取物品 (信賴度: {detection.confidence:.2f}, 水平距離: {horiz_dist:.0f}px)")
            self.stats['items_collected'] += 1
            self.stats['actions_performed'] += 1
            time.sleep(self.config.get('detection_behavior.item.pickup_delay', 0.4))
            return True

        # 範圍外: 持續按住方向鍵走向物品
        if self.item_approach:
            move_key = self.config.get('controls.movement_keys.right', 'right') if direction > 0 \
                else self.config.get('controls.movement_keys.left', 'left')
            if self.chasing_key != move_key:
                self._stop_chase()
                pydirectinput.keyDown(move_key)
                self.chasing_key = move_key
            self.facing = direction
            logger.info(f"🚶 走向物品 ({'右' if direction > 0 else '左'}, 水平距離: {horiz_dist:.0f}px)")
            self.stats['actions_performed'] += 1
            return True

        return False

    def _should_search_for_mobs(self) -> bool:
        """檢查是否應該開始尋找怪物"""
        if not self.config.get('automation.mob_hunting.enable', True):
            return False
        
        # 如果正在搜尋中，不重複開始
        if self.is_searching:
            return False
        
        # 檢查距離上次偵測到怪物的時間
        search_delay = self.config.get('automation.mob_hunting.search_delay', 2.0)
        time_since_last_mob = time.time() - self.last_mob_detection_time
        
        return time_since_last_mob > search_delay
    
    def _start_mob_search(self):
        """開始尋找怪物"""
        if self.is_searching:
            return
        
        self.is_searching = True
        self.search_start_time = time.time()
        self.search_moves = 0
        
        # 記錄當前位置（假設角色在畫面中心）
        self.original_position = (self.monitor['width'] // 2, self.monitor['height'] // 2)
        
        logger.info("🔍 開始尋找怪物...")
    
    def _perform_mob_search(self):
        """執行尋找怪物的移動"""
        if not self.is_searching:
            return
        
        max_search_time = self.config.get('automation.mob_hunting.max_search_time', 10)
        if time.time() - self.search_start_time > max_search_time:
            self._end_mob_search()
            return
        
        search_pattern = self.config.get('automation.mob_hunting.search_pattern', 'horizontal')
        move_distance = self.config.get('automation.mob_hunting.move_distance', 100)
        
        try:
            if search_pattern == 'horizontal':
                self._horizontal_search(move_distance)
            elif search_pattern == 'vertical':
                self._vertical_search(move_distance)
            elif search_pattern == 'random':
                self._random_search(move_distance)
            
            time.sleep(0.5)  # 移動後稍作停頓
            
        except Exception as e:
            logger.error(f"搜尋移動失敗: {e}")
            self._end_mob_search()
    
    def _horizontal_search(self, move_distance: int):
        """水平搜尋移動"""
        move_key = self.config.get('controls.movement_keys.right' if self.search_direction > 0 else 'controls.movement_keys.left', 'right' if self.search_direction > 0 else 'left')
        
        # 按住移動鍵一段時間
        pydirectinput.keyDown(move_key)
        time.sleep(0.3)
        pydirectinput.keyUp(move_key)

        self.search_moves += 1

        # 每移動3次改變方向
        if self.search_moves >= 3:
            self.search_direction *= -1
            self.search_moves = 0
            logger.info(f"🔄 改變搜尋方向: {'右' if self.search_direction > 0 else '左'}")
    
    def _vertical_search(self, move_distance: int):
        """垂直搜尋移動（跳躍和下降）"""
        if self.search_moves % 2 == 0:
            # 跳躍
            jump_key = self.config.get('controls.movement_keys.jump', 'x')
            pydirectinput.press(jump_key)
            logger.info("⬆️ 跳躍搜尋")
        else:
            # 向下移動
            down_key = self.config.get('controls.movement_keys.down', 'down')
            pydirectinput.keyDown(down_key)
            time.sleep(0.2)
            pydirectinput.keyUp(down_key)
            logger.info("⬇️ 向下搜尋")
        
        self.search_moves += 1
    
    def _random_search(self, move_distance: int):
        """隨機搜尋移動"""
        import random
        
        movements = ['left', 'right', 'jump']
        chosen_movement = random.choice(movements)
        
        if chosen_movement == 'jump':
            jump_key = self.config.get('controls.movement_keys.jump', 'x')
            pydirectinput.press(jump_key)
            logger.info("🎲 隨機跳躍")
        else:
            move_key = self.config.get(f'controls.movement_keys.{chosen_movement}', chosen_movement)
            pydirectinput.keyDown(move_key)
            time.sleep(0.3)
            pydirectinput.keyUp(move_key)
            logger.info(f"🎲 隨機移動: {chosen_movement}")
        
        self.search_moves += 1
    
    def _end_mob_search(self):
        """結束尋找怪物"""
        if not self.is_searching:
            return
        
        # 記錄搜尋統計
        search_duration = time.time() - self.search_start_time
        self.stats['searches_performed'] += 1
        self.stats['search_time_total'] += search_duration
        
        self.is_searching = False
        logger.info(f"🏁 結束怪物搜尋 (耗時: {search_duration:.1f}秒)")
        
        # 如果設定要返回中心，執行返回動作
        if self.config.get('automation.mob_hunting.return_to_center', True):
            self._return_to_center()
    
    def _return_to_center(self):
        """返回到搜尋開始的位置"""
        try:
            logger.info("🏠 返回原始位置...")
            # 簡單的返回邏輯：向相反方向移動
            if self.search_direction > 0:
                # 如果最後是向右移動，現在向左移動
                move_key = self.config.get('controls.movement_keys.left', 'left')
            else:
                # 如果最後是向左移動，現在向右移動
                move_key = self.config.get('controls.movement_keys.right', 'right')
            
            pydirectinput.keyDown(move_key)
            time.sleep(0.5)  # 移動時間稍長一些
            pydirectinput.keyUp(move_key)
            
        except Exception as e:
            logger.error(f"返回中心失敗: {e}")
    
    def _check_safety_conditions(self) -> bool:
        """檢查安全條件"""
        if self.start_time and time.time() - self.start_time > self.max_runtime:
            logger.warning("達到最大運行時間限制")
            return False
        return True
    
    def start_automation(self, show_preview: bool = False):
        """開始優化的自動化流程"""
        if self.model is None:
            logger.error("模型未載入，無法開始自動化")
            return
        
        self.running = True
        self.start_time = time.time()
        logger.info("🚀 開始 MapleStory Worlds 優化自動化")

        # 註冊全域熱鍵 (不需視窗焦點): q 暫停/恢復, esc 停止
        self._hotkey_lib = None
        try:
            import keyboard
            self._hotkey_lib = keyboard

            def _toggle_pause():
                self.paused = not self.paused
                logger.info(f"{'⏸️ 暫停' if self.paused else '▶️ 恢復'}自動化")

            def _stop():
                self.running = False
                logger.info("⏹️ 熱鍵停止自動化")

            keyboard.add_hotkey('q', _toggle_pause)
            keyboard.add_hotkey('esc', _stop)
            logger.info("按 'q' 暫停/恢復，'Esc' 停止 (全域熱鍵)")
        except Exception as e:
            logger.warning(f"全域熱鍵不可用 ({e})，請用 Ctrl+C 終止")
        
        last_stats_time = time.time()
        
        try:
            while self.running:
                if not self._check_safety_conditions():
                    break
                
                if self.paused:
                    self._stop_chase()  # 暫停時鬆開追擊鍵, 避免角色卡住往前走
                    time.sleep(0.1)
                    continue
                
                # 擷取和偵測
                img = self.capture_screen()
                if img is None:
                    continue
                
                detections = self.detect_objects(img)

                # 執行動作: 遍歷怪物(已按距離排序), 找到第一隻同層可打/可追的就處理
                # (異層怪物會被 _handle_mob 跳過並回傳 False, 繼續試下一隻)
                mobs = [d for d in detections if d.class_name == 'mob']
                acted = False
                if mobs and not self.paused and self.running:
                    for mob in mobs:
                        if self.perform_action(mob):
                            acted = True
                            break

                # 只有真正處理到同層怪物才算「偵測到可打的怪」:
                # 更新計時、結束搜尋。畫面上只有異層怪不算, 才能觸發搜尋去別的平台
                if acted:
                    self.last_mob_detection_time = time.time()
                    if self.is_searching:
                        self._end_mob_search()

                # 沒怪可打時, 嘗試撿取同層物品 (走過去按 Z)
                picked = False
                if not acted and self.item_action == 'pickup' and not self.paused and self.running:
                    items = [d for d in detections if d.class_name == 'item']
                    for item in items:
                        if self.perform_action(item):
                            picked = True
                            break

                # 既沒打到怪也沒撿到物品這一幀, 鬆開可能還按住的方向鍵, 避免角色一直往前走
                if not acted and not picked:
                    self._stop_chase()

                # 沒有可處理的同層怪物也沒物品可撿, 且不在搜尋中:
                # 檢查是否需要開始搜尋 (走去別的平台找怪)
                if not acted and not picked and self._should_search_for_mobs():
                    self._start_mob_search()
                
                # 如果正在搜尋中，執行搜尋移動
                if self.is_searching:
                    self._perform_mob_search()
                
                # 顯示預覽
                if show_preview and detections:
                    preview_img = self._draw_detections(img.copy(), detections)
                    cv2.imshow('MapleStory Auto Bot - 按 q 暫停/恢復', preview_img)
                
                # 更新性能監控
                self.performance_monitor.update_fps()
                
                # 定期顯示統計
                if time.time() - last_stats_time >= 30:  # 每30秒顯示一次
                    self._log_statistics()
                    last_stats_time = time.time()
                
                # 若顯示預覽, 需 waitKey 刷新視窗 (暫停/停止已由全域熱鍵處理)
                if show_preview:
                    cv2.waitKey(1)

                time.sleep(self.scan_interval)
                
        except KeyboardInterrupt:
            logger.info("⏹️ 使用者中斷自動化")
        except Exception as e:
            logger.error(f"自動化過程中發生錯誤: {e}")
        finally:
            self.running = False
            self._stop_chase()  # 結束時鬆開任何還按住的方向鍵
            if self._hotkey_lib is not None:
                try:
                    self._hotkey_lib.remove_all_hotkeys()
                except Exception:
                    pass
            cv2.destroyAllWindows()
            self._log_final_statistics()
            logger.info("✅ 自動化已停止")
    
    def _draw_detections(self, img: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """繪製偵測結果"""
        for detection in detections:
            bbox = detection.bbox
            class_name = detection.class_name
            confidence = detection.confidence
            
            # 根據類型設定顏色
            color_map = {
                'mob': (0, 0, 255),      # 紅色
                'item': (0, 255, 0),     # 綠色
                'npc': (255, 0, 0),      # 藍色
                'character': (255, 255, 0), # 青色
                'environment': (128, 128, 128), # 灰色
                'ui': (255, 0, 255)      # 洋紅色
            }
            color = color_map.get(class_name, (255, 255, 255))
            
            # 繪製邊界框
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # 繪製標籤
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(img, label, (bbox[0], bbox[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 繪製性能信息
        fps_text = f"FPS: {self.performance_monitor.current_fps}"
        cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return img
    
    def _log_statistics(self):
        """記錄統計信息"""
        runtime = time.time() - self.start_time if self.start_time else 0
        avg_detection_time = self.performance_monitor.get_avg_detection_time()
        
        logger.info("📊 運行統計:")
        logger.info(f"   運行時間: {runtime/60:.1f} 分鐘")
        logger.info(f"   FPS: {self.performance_monitor.current_fps}")
        logger.info(f"   平均偵測時間: {avg_detection_time*1000:.1f}ms")
        logger.info(f"   總偵測次數: {self.stats['detections']}")
        logger.info(f"   執行動作: {self.stats['actions_performed']}")
        logger.info(f"   撿取物品: {self.stats['items_collected']}")
        logger.info(f"   攻擊怪物: {self.stats['mobs_attacked']}")
        logger.info(f"   NPC互動: {self.stats['npcs_interacted']}")
        logger.info(f"   搜尋次數: {self.stats['searches_performed']}")
        if self.stats['searches_performed'] > 0:
            avg_search_time = self.stats['search_time_total'] / self.stats['searches_performed']
            logger.info(f"   平均搜尋時間: {avg_search_time:.1f}秒")
    
    def _log_final_statistics(self):
        """記錄最終統計"""
        logger.info("🎯 最終統計報告:")
        self._log_statistics()
    
    def get_performance_summary(self) -> Dict:
        """獲取性能摘要"""
        runtime = time.time() - self.start_time if self.start_time else 0
        avg_detection_time = self.performance_monitor.get_avg_detection_time()
        
        return {
            'runtime_minutes': runtime / 60,
            'current_fps': self.performance_monitor.current_fps,
            'avg_detection_time_ms': avg_detection_time * 1000,
            'total_detections': self.stats['detections'],
            'actions_performed': self.stats['actions_performed'],
            'items_collected': self.stats['items_collected'],
            'mobs_attacked': self.stats['mobs_attacked'],
            'npcs_interacted': self.stats['npcs_interacted'],
            'searches_performed': self.stats['searches_performed'],
            'avg_search_time': self.stats['search_time_total'] / max(1, self.stats['searches_performed'])
        }
    
    def test_detection(self):
        """測試偵測功能"""
        if self.model is None:
            logger.error("模型未載入")
            return
        
        logger.info("🧪 測試物件偵測功能")
        img = self.capture_screen()
        if img is None:
            logger.error("無法擷取畫面")
            return

        # 診斷: 保存擷取的畫面, 確認截到的是不是遊戲畫面
        cv2.imwrite('debug/debug_capture.png', img)
        logger.info(f"🖼️ 已保存擷取畫面到 debug/debug_capture.png (尺寸: {img.shape[1]}x{img.shape[0]})")
        logger.info(f"   擷取區域: left={self.monitor['left']} top={self.monitor['top']} "
                    f"width={self.monitor['width']} height={self.monitor['height']}")

        # 診斷: 用極低閾值跑一次原始 YOLO, 區分「沒偵測到」還是「被閾值過濾」
        raw_results = self.model(img, conf=0.01, verbose=False)
        raw_count = 0
        for r in raw_results:
            if r.boxes is not None:
                for b in r.boxes:
                    raw_count += 1
                    cls_id = int(b.cls[0].cpu().numpy())
                    conf = float(b.conf[0].cpu().numpy())
                    logger.info(f"   [原始] {self.model.names[cls_id]} 信賴度={conf:.3f}")
        logger.info(f"🔬 極低閾值(0.01)下原始偵測數: {raw_count} "
                    f"(當前閾值 {self.confidence_threshold} 過濾後應更少)")

        detections = self.detect_objects(img)
        logger.info(f"📊 偵測結果: 發現 {len(detections)} 個物件")
        
        for i, detection in enumerate(detections, 1):
            logger.info(f"  {i}. {detection.class_name} (信賴度: {detection.confidence:.2f}, 距離: {detection.distance_from_center:.0f}px)")
        
        if detections:
            result_img = self._draw_detections(img, detections)
            cv2.imshow('Detection Test', result_img)
            logger.info("按任意鍵關閉預覽視窗")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            logger.info("未偵測到任何物件")

def load_available_models() -> Dict[str, str]:
    """載入可用的模型文件"""
    models = {}
    weights_dir = Path("weights")
    
    if weights_dir.exists():
        for i, model_file in enumerate(weights_dir.glob("*.pt"), 1):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            models[str(i)] = str(model_file)
            print(f"  {i}. {model_file} ({size_mb:.1f} MB)")
    
    return models

def main():
    """主程序"""
    print("🍁 MapleStory Worlds 優化自動化系統 v2.0")
    print("=" * 60)
    
    # 檢查配置文件
    if not os.path.exists("config.yaml"):
        logger.warning("配置文件不存在，將使用默認設定")
    
    # 顯示可用模型
    print("可用的模型文件:")
    models = load_available_models()
    
    if not models:
        logger.error("未找到任何模型文件")
        return
    
    # 選擇模型
    choice = input(f"\n請選擇模型文件 (1-{len(models)}, 預設1): ").strip()
    if choice not in models:
        choice = '1'
    
    model_path = models[choice]
    if not os.path.exists(model_path):
        logger.error(f"選擇的模型文件不存在: {model_path}")
        return
    
    # 創建配置並設定模型路徑
    config = ConfigManager()
    config.config['model']['default_path'] = model_path
    
    # 創建機器人
    bot = OptimizedMapleBot()
    
    # 主選單
    while True:
        print("\n🎮 功能選單:")
        print("1. 測試物件偵測")
        print("2. 開始自動化 (有預覽)")
        print("3. 開始自動化 (無預覽)")
        print("4. 調整視窗設定")
        print("5. 查看配置")
        print("6. 查看統計")
        print("7. 退出")
        print("8. 偵測遊戲視窗座標")

        choice = input("\n請選擇功能 (1-8): ").strip()

        if choice == '1':
            bot.test_detection()
        elif choice == '2':
            bot.start_automation(show_preview=True)
        elif choice == '3':
            bot.start_automation(show_preview=False)
        elif choice == '4':
            _adjust_window_settings(bot)
        elif choice == '5':
            _show_config(bot.config)
        elif choice == '6':
            bot._log_statistics()
        elif choice == '8':
            _detect_game_window(bot)
        elif choice == '7':
            break
        else:
            print("❌ 無效選擇")
    
    print("👋 再見！")

def _detect_game_window(bot):
    """列出所有視窗並自動匹配遊戲視窗, 輸出精確座標"""
    try:
        import pygetwindow as gw
    except ImportError:
        logger.error("pygetwindow 未安裝, 請執行: pip install pygetwindow")
        return

    keywords = ['冒险岛', '冒險島', '怀旧服', '懷舊服', '楓之谷', 'maplestory', 'maple', '메이플']
    # 排除瀏覽器/編輯器等含關鍵字但非遊戲的視窗
    exclude = ['chrome', 'edge', 'firefox', 'code', 'powershell', '资源管理器', '資源管理器']
    logger.info("🪟 掃描所有可見視窗:")
    matched = []
    for w in gw.getAllWindows():
        title = (w.title or '').strip()
        if not title:
            continue
        logger.info(f"   標題='{title}' left={w.left} top={w.top} "
                    f"width={w.width} height={w.height}")
        tl = title.lower()
        if any(e in tl for e in exclude):
            continue
        if any(k in tl for k in keywords):
            matched.append(w)

    if matched:
        logger.info("🎯 疑似遊戲視窗:")
        for w in matched:
            logger.info(f"   ★ '{w.title}' left={w.left} top={w.top} "
                        f"width={w.width} height={w.height}")
        w = matched[0]
        bot.monitor = {'left': w.left, 'top': w.top,
                       'width': w.width, 'height': w.height}
        logger.info(f"✅ 已將擷取區域設為: {bot.monitor}")
        logger.info("   (本次有效; 若要永久生效請更新 config.yaml 的 window.default)")
    else:
        logger.info("⚠️ 未自動匹配到遊戲視窗, 請從上方清單找出遊戲標題手動設定")


def _adjust_window_settings(bot):
    """調整視窗設定"""
    print(f"\n當前視窗設定:")
    print(f"  左上角: ({bot.monitor['left']}, {bot.monitor['top']})")
    print(f"  大小: {bot.monitor['width']} x {bot.monitor['height']}")
    
    # 提供預設選項
    print("\n預設選項:")
    print("1. Full HD (1920x1080)")
    print("2. QHD (2560x1440)")
    print("3. 自訂設定")
    
    preset_choice = input("選擇預設或自訂 (1-3): ").strip()
    
    if preset_choice == '1':
        bot.monitor = {'left': 0, 'top': 100, 'width': 1920, 'height': 980}
    elif preset_choice == '2':
        bot.monitor = {'left': 320, 'top': 180, 'width': 1280, 'height': 720}
    elif preset_choice == '3':
        try:
            bot.monitor['left'] = int(input("請輸入左側位置: ") or bot.monitor['left'])
            bot.monitor['top'] = int(input("請輸入頂部位置: ") or bot.monitor['top'])
            bot.monitor['width'] = int(input("請輸入寬度: ") or bot.monitor['width'])
            bot.monitor['height'] = int(input("請輸入高度: ") or bot.monitor['height'])
        except ValueError:
            print("❌ 輸入格式錯誤")
            return
    
    print("✅ 視窗設定已更新")

def _show_config(config: ConfigManager):
    """顯示當前配置"""
    print("\n⚙️ 當前配置:")
    print(f"  模型路徑: {config.get('model.default_path')}")
    print(f"  信賴度閾值: {config.get('model.confidence_threshold')}")
    print(f"  動作延遲: {config.get('automation.action_delay')}秒")
    print(f"  掃描間隔: {config.get('automation.scan_interval')}秒")
    print(f"  最大運行時間: {config.get('safety.max_runtime_hours')}小時")
    print(f"  撿取鍵: {config.get('controls.pickup_key')}")
    print(f"  互動鍵: {config.get('controls.interact_key')}")

if __name__ == "__main__":
    main() 