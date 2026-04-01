#!/usr/bin/env python3
"""
STAMP - Playback Mixin
========================

Video playback control methods for HILAAMainWindow.

Host requirements (attributes expected on self):
- video_loader, current_frame, is_playing
- play_timer, play_btn, timeline_slider, frame_label
- object_list, timeline_widget
- action_logger, sam3_results, object_status
- display_frame(), statusBar()
"""

import logging

from PyQt6.QtCore import Qt

logger = logging.getLogger(__name__)


class PlaybackMixin:
    """影片播放控制邏輯。"""

    def next_frame(self):
        """下一幀。"""
        if self.video_loader is None:
            return
        
        total = self.video_loader.metadata.total_frames
        if self.current_frame < total - 1:
            self.display_frame(self.current_frame + 1)
        else:
            self.stop_play()
    
    def prev_frame(self):
        """上一幀。"""
        if self.current_frame > 0:
            self.display_frame(self.current_frame - 1)
    
    def toggle_play(self):
        """切換播放/暫停。"""
        if self.is_playing:
            self.stop_play()
        else:
            self.start_play()
    
    def start_play(self):
        """開始播放。"""
        if self.video_loader is None:
            return
        
        self.is_playing = True
        self.play_btn.setText("||")
        interval = int(1000 / self.video_loader.metadata.fps)
        self.play_timer.start(interval)
    
    def stop_play(self):
        """停止播放。"""
        self.is_playing = False
        self.play_btn.setText("▶")
        self.play_timer.stop()
    
    def on_slider_moved(self, value: int):
        """滑桿移動。"""
        self.display_frame(value)
    
    def seek_to_frame(self, frame_idx: int):
        """跳轉到指定幀（供 Timeline 使用）。"""
        if self.video_loader is None:
            return
        
        from_frame = self.current_frame
        frame_idx = max(0, min(frame_idx, self.video_loader.metadata.total_frames - 1))
        
        # === ActionLogger: 記錄幀跳轉（只記錄重要跳轉，避免播放時大量 log）===
        if abs(frame_idx - from_frame) > 1:  # 只記錄跳躍超過 1 幀的情況
            self.action_logger.log_frame_navigation(from_frame=from_frame, to_frame=frame_idx)
        
        self.timeline_slider.setValue(frame_idx)
        self.display_frame(frame_idx)
    
    def on_timeline_object_selected(self, obj_id: int):
        """Timeline 上選擇了物件。"""
        # 在物件列表中選擇對應的物件
        for i in range(self.object_list.count()):
            item = self.object_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == obj_id:
                self.object_list.setCurrentItem(item)
                break
        
        logger.info(f"Timeline selected object {obj_id}")
    
    def update_timeline(self):
        """更新 Timeline 顯示。"""
        if not hasattr(self, 'timeline_widget'):
            return
        
        if self.video_loader is None or not self.sam3_results:
            self.timeline_widget.setVisible(False)
            return
        
        # 取得 jitter frames
        jitter_frames = None
        if hasattr(self, 'jitter_analysis') and self.jitter_analysis:
            jitter_frames = self.jitter_analysis.get_all_jitter_frames()
        
        self.timeline_widget.set_data(
            sam3_results=self.sam3_results,
            total_frames=self.video_loader.metadata.total_frames,
            fps=self.video_loader.metadata.fps,
            object_status=getattr(self, 'object_status', None),
            jitter_frames=jitter_frames
        )
        self.timeline_widget.setVisible(True)
