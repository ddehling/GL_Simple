# Standalone tool: audio waveform editor and clip trimmer (not part of the main app).
"""
Modern Sound Editor Utility
Features:
- Load audio files with file picker
- Display waveform with PyQtGraph (ultra-fast plotting)
- Interactive zoom and pan
- Select two points on waveform
- Display selected region in separate view
- Save selected audio segment
"""

import sys
import os
import numpy as np
import soundfile as sf
import sounddevice as sd
import threading
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QLabel,
                             QMessageBox, QSplitter)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QPointF, QSettings
import pyqtgraph as pg


class WaveformWidget(pg.PlotWidget):
    """Custom PyQtGraph widget for displaying waveforms with selection capability"""
    
    def __init__(self, parent=None, allow_selection=True):
        super().__init__(parent)
        
        self.audio_data = None
        self.sample_rate = None
        self.time_array = None
        self.allow_selection = allow_selection
        
        # Disable context menu so right-click can be used for adding keypoints
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        
        # Selection points
        self.selection_start = None
        self.selection_end = None
        self.selection_lines = []
        self.selection_region = None
        
        # Configure plot
        self.setLabel('left', 'Amplitude')
        self.setLabel('bottom', 'Time', units='s')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setBackground('w')
        
        # Enable antialiasing for smoother lines
        self.setAntialiasing(True)
        
        # Disable vertical zoom - only allow horizontal zoom
        self.setMouseEnabled(x=True, y=False)
        
        # Plot items for waveform (left and right channels)
        self.waveform_plots = []
        
        # Store full audio data for dynamic downsampling
        self.full_audio_data = None
        self.full_time_array = None
        
        # Volume envelope for selection view
        self.volume_envelope_keypoints = []  # List of (time, volume) tuples
        self.volume_envelope_line = None
        self.volume_envelope_points = None
        self.show_volume_envelope = False
        self.selected_keypoint_index = None
        
        # Playback position line
        self.playback_line = None
        self.is_playing = False
        self.parent_editor = None  # Will be set by parent
        
        # Connect mouse events for selection
        if self.allow_selection:
            self.scene().sigMouseClicked.connect(self.on_click)
        
        # Connect for volume envelope point dragging
        self.scene().sigMouseClicked.connect(self.on_envelope_click)
        self.dragging_keypoint_index = None
        self.last_mouse_pos = None
        self.mouse_pressed = False
        self.panning_disabled = False  # Track if we disabled panning
        
        # Enable mouse tracking for dragging
        self.setMouseTracking(True)
        self.scene().sigMouseMoved.connect(self.on_mouse_move)
        
        # Connect to view range changes for dynamic downsampling
        self.plotItem.vb.sigRangeChanged.connect(self.on_view_range_changed)
        
        # Disable the PyQtGraph context menu
        self.plotItem.vb.setMenuEnabled(False)
    
    def init_volume_envelope(self, duration):
        """Initialize volume envelope with start and end keypoints at 1.0"""
        self.volume_envelope_keypoints = [
            (0.0, 1.0),           # Start at volume 1.0
            (duration, 1.0)       # End at volume 1.0
        ]
        self.show_volume_envelope = True
        self.draw_volume_envelope()
    
    def draw_volume_envelope(self):
        """Draw the volume envelope on the plot"""
        if not self.show_volume_envelope or not self.volume_envelope_keypoints:
            return
        
        # Clear previous envelope visuals
        if self.volume_envelope_line is not None:
            self.removeItem(self.volume_envelope_line)
        if self.volume_envelope_points is not None:
            self.removeItem(self.volume_envelope_points)
        
        # Keep track of selected keypoint position before sorting
        selected_pos = None
        if self.selected_keypoint_index is not None:
            selected_pos = self.volume_envelope_keypoints[self.selected_keypoint_index]
        
        # Sort keypoints by time and update the list
        self.volume_envelope_keypoints = sorted(self.volume_envelope_keypoints, key=lambda x: x[0])
        
        # Update selected index after sorting
        if selected_pos is not None:
            for idx, kp in enumerate(self.volume_envelope_keypoints):
                if kp == selected_pos:
                    self.selected_keypoint_index = idx
                    break
        
        # Extract times and volumes
        times = [kp[0] for kp in self.volume_envelope_keypoints]
        volumes = [kp[1] for kp in self.volume_envelope_keypoints]
        
        # Draw line connecting keypoints
        self.volume_envelope_line = self.plot(
            times, volumes,
            pen=pg.mkPen(color='m', width=2, style=Qt.PenStyle.DashLine),
            name='Volume Envelope'
        )
        
        # Draw draggable keypoints
        self.volume_envelope_points = pg.ScatterPlotItem(
            times, volumes,
            size=12,
            pen=pg.mkPen(color='m', width=2),
            brush=pg.mkBrush(255, 0, 255, 150),
            symbol='o',
            hoverable=True,
            tip=None
        )
        self.volume_envelope_points.sigClicked.connect(self.on_keypoint_clicked)
        self.addItem(self.volume_envelope_points)
    
    def on_envelope_click(self, event):
        """Handle clicks for adding/deselecting volume envelope keypoints"""
        if not self.show_volume_envelope:
            return
        
        # Left-click on empty space to deselect
        if event.button() == Qt.MouseButton.LeftButton:
            mouse_point = self.plotItem.vb.mapSceneToView(event.scenePos())
            
            # Check if click is near any keypoint
            if not self.is_near_keypoint(mouse_point.x(), mouse_point.y()):
                # Deselect current keypoint
                if self.selected_keypoint_index is not None:
                    self.selected_keypoint_index = None
                    self.draw_volume_envelope()
        
        # Right-click to add keypoint
        elif event.button() == Qt.MouseButton.RightButton:
            mouse_point = self.plotItem.vb.mapSceneToView(event.scenePos())
            click_time = mouse_point.x()
            click_volume = mouse_point.y()
            
            if self.time_array is not None:
                # Clamp time to valid range
                if click_time >= self.time_array[0] and click_time <= self.time_array[-1]:
                    # Clamp volume to 0-2 range
                    click_volume = max(0.0, min(2.0, click_volume))
                    
                    # Add new keypoint
                    self.volume_envelope_keypoints.append((click_time, click_volume))
                    self.draw_volume_envelope()
                    
                    if self.parent_editor:
                        self.parent_editor.update_envelope_info()
    
    def is_near_keypoint(self, x, y, threshold=0.1):
        """Check if coordinates are near any keypoint"""
        view_range = self.viewRange()
        x_range = view_range[0][1] - view_range[0][0]
        y_range = view_range[1][1] - view_range[1][0]
        
        # Scale threshold based on view range
        x_threshold = x_range * 0.02  # 2% of view width
        y_threshold = y_range * 0.05  # 5% of view height
        
        for t, v in self.volume_envelope_keypoints:
            if abs(t - x) < x_threshold and abs(v - y) < y_threshold:
                return True
        return False
    
    def on_keypoint_clicked(self, scatter, points):
        """Handle clicking on a volume envelope keypoint to select it"""
        if len(points) > 0:
            point = points[0]
            # Find which keypoint was clicked
            click_pos = point.pos()
            
            # Find the closest keypoint
            min_dist = float('inf')
            closest_idx = None
            
            for idx, (t, v) in enumerate(self.volume_envelope_keypoints):
                dist = (t - click_pos.x())**2 + (v - click_pos.y())**2
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = idx
            
            if closest_idx is not None:
                # Toggle selection if clicking on already selected point
                if self.selected_keypoint_index == closest_idx:
                    self.selected_keypoint_index = None
                else:
                    self.selected_keypoint_index = closest_idx
                
                self.draw_volume_envelope()
                self.highlight_selected_keypoint()
                
                if self.parent_editor:
                    self.parent_editor.update_envelope_info()
    
    def on_mouse_move(self, pos):
        """Handle mouse movement for dragging keypoints"""
        # Only drag if mouse button is pressed and a keypoint is selected
        if self.mouse_pressed and self.selected_keypoint_index is not None and self.show_volume_envelope:
            self.dragging_keypoint_index = self.selected_keypoint_index
            
            # Disable panning on first drag movement
            if not self.panning_disabled:
                self.plotItem.vb.setMouseEnabled(x=False, y=False)
                self.panning_disabled = True
            
            # Convert to data coordinates
            mouse_point = self.plotItem.vb.mapSceneToView(pos)
            new_time = mouse_point.x()
            new_volume = mouse_point.y()
            
            # Clamp volume between 0-2
            new_volume = max(0.0, min(2.0, new_volume))
            
            # Handle first and last keypoints - they can only move vertically
            if self.dragging_keypoint_index == 0:
                # First keypoint - keep original time
                new_time = self.volume_envelope_keypoints[0][0]
            elif self.dragging_keypoint_index == len(self.volume_envelope_keypoints) - 1:
                # Last keypoint - keep original time
                new_time = self.volume_envelope_keypoints[-1][0]
            else:
                # Middle keypoints - can move horizontally, but constrained between neighbors
                prev_time = self.volume_envelope_keypoints[self.dragging_keypoint_index - 1][0]
                next_time = self.volume_envelope_keypoints[self.dragging_keypoint_index + 1][0]
                new_time = max(prev_time, min(next_time, new_time))
            
            # Update keypoint
            self.volume_envelope_keypoints[self.dragging_keypoint_index] = (new_time, new_volume)
            self.draw_volume_envelope()
            self.highlight_selected_keypoint()
            
            if self.parent_editor:
                self.parent_editor.update_envelope_info()
    
    def mousePressEvent(self, event):
        """Track mouse press for dragging"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_pressed = True
        super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Stop dragging on mouse release"""
        self.mouse_pressed = False
        self.dragging_keypoint_index = None
        
        # Re-enable plot interaction if we disabled it
        if self.panning_disabled:
            self.plotItem.vb.setMouseEnabled(x=True, y=False)
            self.panning_disabled = False
        
        super().mouseReleaseEvent(event)
    
    def highlight_selected_keypoint(self):
        """Highlight the currently selected keypoint"""
        if self.volume_envelope_points is not None:
            times = [kp[0] for kp in self.volume_envelope_keypoints]
            volumes = [kp[1] for kp in self.volume_envelope_keypoints]
            
            brushes = []
            sizes = []
            for i in range(len(self.volume_envelope_keypoints)):
                if i == self.selected_keypoint_index:
                    brushes.append(pg.mkBrush(255, 255, 0, 220))  # Bright yellow for selected
                    sizes.append(16)  # Larger size for selected
                else:
                    brushes.append(pg.mkBrush(255, 0, 255, 150))  # Magenta for others
                    sizes.append(12)  # Normal size
            
            self.volume_envelope_points.setData(times, volumes, size=sizes, brush=brushes)
    
    def update_keypoint_position(self, index, new_time, new_volume):
        """Update a keypoint's position"""
        if 0 <= index < len(self.volume_envelope_keypoints):
            # Clamp values
            new_time = max(self.time_array[0], min(self.time_array[-1], new_time))
            new_volume = max(0.0, min(2.0, new_volume))
            
            self.volume_envelope_keypoints[index] = (new_time, new_volume)
            self.draw_volume_envelope()
    
    def remove_selected_keypoint(self):
        """Remove the currently selected keypoint"""
        if self.selected_keypoint_index is not None:
            # Don't allow removing first or last keypoint
            if self.selected_keypoint_index > 0 and self.selected_keypoint_index < len(self.volume_envelope_keypoints) - 1:
                del self.volume_envelope_keypoints[self.selected_keypoint_index]
                self.selected_keypoint_index = None
                self.dragging_keypoint_index = None
                self.draw_volume_envelope()
                return True
        return False
    
    def get_volume_envelope_array(self, num_samples):
        """Generate volume envelope array for the entire audio"""
        if not self.volume_envelope_keypoints or num_samples == 0:
            return np.ones(num_samples)
        
        # Sort keypoints by time
        keypoints = sorted(self.volume_envelope_keypoints, key=lambda x: x[0])
        
        # Create sample indices array
        sample_times = np.arange(num_samples) / self.sample_rate if self.sample_rate else np.arange(num_samples)
        
        # Interpolate volume values
        times = np.array([kp[0] for kp in keypoints])
        volumes = np.array([kp[1] for kp in keypoints])
        
        # Linear interpolation
        envelope = np.interp(sample_times, times, volumes)
        
        return envelope
    
    def clear_volume_envelope(self):
        """Clear the volume envelope"""
        self.show_volume_envelope = False
        if self.volume_envelope_line is not None:
            self.removeItem(self.volume_envelope_line)
            self.volume_envelope_line = None
        if self.volume_envelope_points is not None:
            self.removeItem(self.volume_envelope_points)
            self.volume_envelope_points = None
        self.volume_envelope_keypoints = []
        self.selected_keypoint_index = None
    
    def plot_waveform(self, audio_data, sample_rate, title="Waveform"):
        """Plot the waveform with dynamic downsampling based on zoom"""
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        
        # Store full data for dynamic downsampling
        self.full_audio_data = audio_data.copy()
        self.full_time_array = np.arange(len(audio_data)) / sample_rate
        
        # Create time array
        self.time_array = self.full_time_array
        
        # Clear previous plot
        self.clear()
        self.waveform_plots = []
        
        # Clear playback line reference since clear() removed it
        self.playback_line = None
        
        # Plot with initial downsampling
        self.update_waveform_display()
        
        self.setTitle(title)
    
    def update_waveform_display(self):
        """Update waveform display with appropriate downsampling based on view"""
        if self.full_audio_data is None:
            return
        
        # Get current view range
        view_range = self.viewRange()
        x_range = view_range[0]
        
        # Calculate visible time range
        time_start = max(x_range[0], self.full_time_array[0])
        time_end = min(x_range[1], self.full_time_array[-1])
        
        # Find sample indices for visible range
        start_idx = int(time_start * self.sample_rate)
        end_idx = int(time_end * self.sample_rate)
        
        # Clamp to valid range
        start_idx = max(0, start_idx)
        end_idx = min(len(self.full_audio_data), end_idx)
        
        # Get visible data
        visible_audio = self.full_audio_data[start_idx:end_idx]
        visible_time = self.full_time_array[start_idx:end_idx]
        
        if len(visible_audio) == 0:
            return
        
        # Calculate target number of points based on widget width
        widget_width = self.width()
        target_points = min(widget_width * 2, len(visible_audio))  # 2 points per pixel
        
        # Downsample if needed
        if len(visible_audio) > target_points:
            downsampled_time, downsampled_audio = self.downsample_minmax(
                visible_time, visible_audio, target_points
            )
        else:
            downsampled_time = visible_time
            downsampled_audio = visible_audio
        
        # Clear old waveform plots
        for plot in self.waveform_plots:
            self.removeItem(plot)
        self.waveform_plots = []
        
        # Check if stereo or mono
        is_stereo = len(downsampled_audio.shape) > 1 and downsampled_audio.shape[1] == 2
        
        if is_stereo:
            # Plot left channel (top)
            left_channel = downsampled_audio[:, 0]
            plot_left = self.plot(downsampled_time, left_channel, 
                                 pen=pg.mkPen(color='b', width=1),
                                 name='Left Channel')
            self.waveform_plots.append(plot_left)
            
            # Plot right channel (bottom)
            right_channel = downsampled_audio[:, 1]
            plot_right = self.plot(downsampled_time, right_channel, 
                                  pen=pg.mkPen(color='r', width=1),
                                  name='Right Channel')
            self.waveform_plots.append(plot_right)
            
            # Add legend if not present
            if self.plotItem.legend is None:
                self.addLegend()
        else:
            # Mono audio - single channel
            if len(downsampled_audio.shape) > 1:
                downsampled_audio = downsampled_audio[:, 0]
            
            plot_mono = self.plot(downsampled_time, downsampled_audio, 
                                 pen=pg.mkPen(color='b', width=1),
                                 name='Mono')
            self.waveform_plots.append(plot_mono)
        
        # Redraw volume envelope if present
        if self.show_volume_envelope:
            self.draw_volume_envelope()
    
    def downsample_minmax(self, time_array, audio_data, target_points):
        """Downsample using min-max method to preserve waveform envelope"""
        n_samples = len(audio_data)
        
        if n_samples <= target_points:
            return time_array, audio_data
        
        # Handle mono vs stereo
        is_stereo = len(audio_data.shape) > 1 and audio_data.shape[1] == 2
        
        downsample_factor = n_samples // (target_points // 2)
        n_bins = n_samples // downsample_factor
        
        downsampled_time = np.zeros(n_bins * 2)
        
        if is_stereo:
            downsampled_audio = np.zeros((n_bins * 2, 2))
            
            for i in range(n_bins):
                start_idx = i * downsample_factor
                end_idx = min(start_idx + downsample_factor, n_samples)
                
                chunk_left = audio_data[start_idx:end_idx, 0]
                chunk_right = audio_data[start_idx:end_idx, 1]
                time_chunk = time_array[start_idx:end_idx]
                
                # Store min and max for this chunk
                downsampled_time[i * 2] = time_chunk[0]
                downsampled_time[i * 2 + 1] = time_chunk[-1]
                
                downsampled_audio[i * 2, 0] = np.min(chunk_left)
                downsampled_audio[i * 2 + 1, 0] = np.max(chunk_left)
                downsampled_audio[i * 2, 1] = np.min(chunk_right)
                downsampled_audio[i * 2 + 1, 1] = np.max(chunk_right)
        else:
            downsampled_audio = np.zeros(n_bins * 2)
            
            # Handle mono audio (could be 1D or 2D with single column)
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
            
            for i in range(n_bins):
                start_idx = i * downsample_factor
                end_idx = min(start_idx + downsample_factor, n_samples)
                
                chunk = audio_data[start_idx:end_idx]
                time_chunk = time_array[start_idx:end_idx]
                
                downsampled_time[i * 2] = time_chunk[0]
                downsampled_time[i * 2 + 1] = time_chunk[-1]
                downsampled_audio[i * 2] = np.min(chunk)
                downsampled_audio[i * 2 + 1] = np.max(chunk)
        
        return downsampled_time, downsampled_audio
    
    def on_view_range_changed(self):
        """Called when the view range changes (zoom/pan)"""
        if self.full_audio_data is not None:
            self.update_waveform_display()
    
    def on_click(self, event):
        """Handle mouse clicks for selection"""
        if not self.allow_selection:
            return
        
        # Get click position in data coordinates
        mouse_point = self.plotItem.vb.mapSceneToView(event.scenePos())
        click_time = mouse_point.x()
        
        if click_time is None or self.time_array is None:
            return
        
        # Ensure click is within bounds
        if click_time < self.time_array[0] or click_time > self.time_array[-1]:
            return
        
        # Set selection points
        if self.selection_start is None:
            self.selection_start = click_time
            self.clear_selection_visual()
            self.draw_selection()
        elif self.selection_end is None:
            self.selection_end = click_time
            # Ensure start < end
            if self.selection_start > self.selection_end:
                self.selection_start, self.selection_end = self.selection_end, self.selection_start
            self.draw_selection()
        else:
            # Reset selection
            self.selection_start = click_time
            self.selection_end = None
            self.clear_selection_visual()
            self.draw_selection()
    
    def clear_selection_visual(self):
        """Clear selection visualization"""
        for line in self.selection_lines:
            self.removeItem(line)
        self.selection_lines = []
        if self.selection_region:
            self.removeItem(self.selection_region)
            self.selection_region = None
    
    def draw_selection(self):
        """Draw selection on the waveform"""
        self.clear_selection_visual()
        
        if self.selection_start is not None:
            line1 = pg.InfiniteLine(pos=self.selection_start, angle=90, 
                                   pen=pg.mkPen(color='r', width=2, style=Qt.PenStyle.DashLine),
                                   label='Start',
                                   movable=True)  # Make it draggable
            line1.sigPositionChanged.connect(self.on_start_line_moved)
            self.addItem(line1)
            self.selection_lines.append(line1)
        
        if self.selection_end is not None:
            line2 = pg.InfiniteLine(pos=self.selection_end, angle=90, 
                                   pen=pg.mkPen(color='g', width=2, style=Qt.PenStyle.DashLine),
                                   label='End',
                                   movable=True)  # Make it draggable
            line2.sigPositionChanged.connect(self.on_end_line_moved)
            self.addItem(line2)
            self.selection_lines.append(line2)
            
            # Highlight selected region
            self.selection_region = pg.LinearRegionItem(
                values=[self.selection_start, self.selection_end],
                brush=pg.mkBrush(255, 255, 0, 50),
                movable=False
            )
            self.addItem(self.selection_region)
    
    def on_start_line_moved(self, line):
        """Handle start line being dragged"""
        new_pos = line.value()
        self.selection_start = new_pos
        
        # Update the region if both points are set
        if self.selection_region and self.selection_end is not None:
            # Ensure start < end
            if self.selection_start > self.selection_end:
                self.selection_start, self.selection_end = self.selection_end, self.selection_start
                self.draw_selection()
            else:
                self.selection_region.setRegion([self.selection_start, self.selection_end])
        
        # Notify parent to update selection view
        self.sigRegionChanged = True
    
    def on_end_line_moved(self, line):
        """Handle end line being dragged"""
        new_pos = line.value()
        self.selection_end = new_pos
        
        # Update the region if both points are set
        if self.selection_region and self.selection_start is not None:
            # Ensure start < end
            if self.selection_start > self.selection_end:
                self.selection_start, self.selection_end = self.selection_end, self.selection_start
                self.draw_selection()
            else:
                self.selection_region.setRegion([self.selection_start, self.selection_end])
        
        # Notify parent to update selection view
        self.sigRegionChanged = True
    
    def get_selection_indices(self):
        """Get the sample indices for the selection"""
        if self.selection_start is None or self.selection_end is None:
            return None, None
        
        start_idx = int(self.selection_start * self.sample_rate)
        end_idx = int(self.selection_end * self.sample_rate)
        
        return start_idx, end_idx
    
    def reset_selection(self):
        """Reset the selection"""
        self.selection_start = None
        self.selection_end = None
        self.clear_selection_visual()
    
    def update_playback_position(self, time_pos):
        """Update the playback position line"""
        if self.playback_line is None:
            self.playback_line = pg.InfiniteLine(
                pos=time_pos,
                angle=90,
                pen=pg.mkPen(color='orange', width=2),
                movable=True  # Make it draggable
            )
            self.playback_line.sigPositionChanged.connect(self.on_playback_line_moved)
            self.addItem(self.playback_line)
        else:
            self.playback_line.setValue(time_pos)
    
    def on_playback_line_moved(self, line):
        """Handle playback line being dragged to seek"""
        if self.parent_editor:
            new_time = line.value()
            self.parent_editor.seek_playback(new_time)
    
    def clear_playback_line(self):
        """Remove the playback position line"""
        if self.playback_line is not None:
            self.removeItem(self.playback_line)
            self.playback_line = None


class SoundEditor(QMainWindow):
    """Main sound editor application"""
    
    # Signal for thread-safe playback finished handling
    playback_finished_signal = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.audio_data = None
        self.sample_rate = None
        self.current_file = None
        
        # Settings for persistent data (using INI file format)
        self.settings = QSettings('sound_editor_settings.ini', QSettings.Format.IniFormat)
        
        # Playback state
        self.playback_stream = None
        self.playback_start_time = 0
        self.playback_offset = 0
        self.is_playing_full = False
        self.is_playing_selection = False
        self.playback_position = 0
        self.playback_lock = threading.Lock()  # Thread-safe position access
        
        # Connect signal for thread-safe playback finished
        self.playback_finished_signal.connect(self.stop_playback)
        
        self.init_ui()
    
    def closeEvent(self, event):
        """Clean up resources when closing the application"""
        # Stop any active playback
        self.stop_playback()
        
        # Accept the close event
        event.accept()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Sound Editor - PyQtGraph')
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.load_btn = QPushButton('Load Audio File')
        self.load_btn.clicked.connect(self.load_audio)
        button_layout.addWidget(self.load_btn)
        
        self.save_selection_btn = QPushButton('Save Selection')
        self.save_selection_btn.clicked.connect(self.save_selection)
        self.save_selection_btn.setEnabled(False)
        button_layout.addWidget(self.save_selection_btn)
        
        self.reset_selection_btn = QPushButton('Reset Selection')
        self.reset_selection_btn.clicked.connect(self.reset_selection)
        self.reset_selection_btn.setEnabled(False)
        button_layout.addWidget(self.reset_selection_btn)
        
        button_layout.addStretch()
        
        self.info_label = QLabel('No file loaded')
        button_layout.addWidget(self.info_label)
        
        main_layout.addLayout(button_layout)
        
        # Create splitter for waveform views
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Main waveform section
        main_waveform_widget = QWidget()
        main_waveform_layout = QVBoxLayout(main_waveform_widget)
        
        # Add play controls for main waveform
        main_controls_layout = QHBoxLayout()
        main_controls_layout.addWidget(QLabel('<b>Full Waveform (Click to select start/end points)</b>'))
        main_controls_layout.addStretch()
        
        self.play_full_btn = QPushButton('▶ Play Full')
        self.play_full_btn.clicked.connect(self.toggle_play_full)
        self.play_full_btn.setEnabled(False)
        main_controls_layout.addWidget(self.play_full_btn)
        
        main_waveform_layout.addLayout(main_controls_layout)
        
        self.main_canvas = WaveformWidget(allow_selection=True)
        self.main_canvas.parent_editor = self  # Set parent reference
        
        main_waveform_layout.addWidget(self.main_canvas)
        
        splitter.addWidget(main_waveform_widget)
        
        # Selection detail section
        selection_widget = QWidget()
        selection_layout = QVBoxLayout(selection_widget)
        
        # Add play controls for selection
        selection_controls_layout = QHBoxLayout()
        selection_controls_layout.addWidget(QLabel('<b>Selected Region Detail</b>'))
        selection_controls_layout.addStretch()
        
        self.play_selection_btn = QPushButton('▶ Play Selection')
        self.play_selection_btn.clicked.connect(self.toggle_play_selection)
        self.play_selection_btn.setEnabled(False)
        selection_controls_layout.addWidget(self.play_selection_btn)
        
        self.remove_keypoint_btn = QPushButton('Remove Keypoint (Del)')
        self.remove_keypoint_btn.clicked.connect(self.remove_keypoint)
        self.remove_keypoint_btn.setEnabled(False)
        selection_controls_layout.addWidget(self.remove_keypoint_btn)
        
        selection_layout.addLayout(selection_controls_layout)
        
        self.selection_canvas = WaveformWidget(allow_selection=False)
        self.selection_canvas.parent_editor = self  # Set parent reference
        
        self.selection_info_label = QLabel('No selection made')
        
        self.envelope_info_label = QLabel('Volume Envelope: Right-click to add | Left-click to select/deselect | Drag selected to move | Del to remove')
        self.envelope_info_label.setStyleSheet('color: purple; font-style: italic;')
        
        selection_layout.addWidget(self.selection_info_label)
        selection_layout.addWidget(self.envelope_info_label)
        selection_layout.addWidget(self.selection_canvas)
        
        splitter.addWidget(selection_widget)
        
        # Set initial sizes (60% main, 40% selection)
        splitter.setSizes([600, 400])
        
        main_layout.addWidget(splitter)
        
        # Connect main canvas selection updates
        self.main_canvas.scene().sigMouseClicked.connect(self.update_selection_view)
        
        # Setup timer for live updates when dragging selection lines
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.check_selection_update)
        self.update_timer.start(100)  # Check every 100ms
        
        # Setup timer for playback position updates
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.update_playback_position)
        self.playback_timer.setInterval(20)  # Update every 20ms for smooth animation
        
        # Setup timer for envelope point dragging
        self.envelope_drag_timer = QTimer()
        self.envelope_drag_timer.timeout.connect(self.update_envelope_dragging)
        self.envelope_drag_timer.setInterval(16)  # ~60 FPS
        self.is_dragging_keypoint = False
        
        # Connect keyboard shortcuts
        from PyQt6.QtGui import QShortcut, QKeySequence
        self.delete_shortcut = QShortcut(QKeySequence('Delete'), self)
        self.delete_shortcut.activated.connect(self.remove_keypoint)
    
    def load_audio(self):
        """Load an audio file"""
        # Get last used directory from settings, default to home directory
        last_dir = self.settings.value('last_directory', '')
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio File",
            last_dir,
            "Audio Files (*.wav *.flac *.ogg *.mp3 *.m4a *.aac *.wma);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # Save the directory for next time
            self.settings.setValue('last_directory', os.path.dirname(file_path))
            
            # Load audio file — try soundfile first, fall back to ffmpeg for
            # formats it can't handle (m4a, aac, wma, etc.)
            try:
                self.audio_data, self.sample_rate = sf.read(file_path)
            except Exception:
                self.audio_data, self.sample_rate = self._read_via_pyav(file_path)
            self.current_file = file_path
            
            # Update UI
            duration = len(self.audio_data) / self.sample_rate
            channels = 1 if len(self.audio_data.shape) == 1 else self.audio_data.shape[1]
            
            self.info_label.setText(
                f'File: {file_path.split("/")[-1]} | '
                f'Duration: {duration:.2f}s | '
                f'Sample Rate: {self.sample_rate}Hz | '
                f'Channels: {channels}'
            )
            
            # Plot waveform
            self.main_canvas.plot_waveform(self.audio_data, self.sample_rate)
            self.main_canvas.reset_selection()
            
            # Initialize playback line at position 0
            self.main_canvas.update_playback_position(0)
            
            # Clear selection view
            self.selection_canvas.clear()
            self.selection_canvas.setTitle('No selection - click on waveform to select')
            self.selection_info_label.setText('No selection made')
            
            self.reset_selection_btn.setEnabled(True)
            self.play_full_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load audio file:\n{str(e)}")
    
    @staticmethod
    def _read_via_pyav(file_path):
        """Decode audio via PyAV for formats soundfile can't handle (m4a, aac, wma, etc.).

        PyAV bundles its own ffmpeg libraries, so no system ffmpeg install is needed.
        """
        import av
        container = av.open(file_path)
        stream = container.streams.audio[0]
        sr = stream.rate
        frames = []
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray()  # shape: (channels, samples), float32
            frames.append(arr)
        container.close()
        if not frames:
            raise RuntimeError("No audio frames decoded")
        audio = np.concatenate(frames, axis=1)  # (channels, total_samples)
        # Transpose to (samples, channels) to match soundfile convention
        if audio.shape[0] == 1:
            audio = audio[0]          # mono → 1-D
        else:
            audio = audio.T           # stereo+ → (samples, channels)
        return audio.astype(np.float64), sr

    @staticmethod
    def _write_via_pyav(file_path, audio_data, sample_rate):
        """Encode audio via PyAV for formats soundfile can't write (mp3, m4a, etc.)."""
        import av

        # Normalise to (samples, channels)
        if audio_data.ndim == 1:
            audio_data = audio_data[:, np.newaxis]
        n_samples, n_channels = audio_data.shape

        # Map extension → codec
        ext = os.path.splitext(file_path)[1].lower()
        codec_map = {'.mp3': 'libmp3lame', '.m4a': 'aac', '.aac': 'aac', '.wma': 'wmav2'}
        codec = codec_map.get(ext, 'libmp3lame')

        container = av.open(file_path, mode='w')
        stream = container.add_stream(codec, rate=sample_rate)
        stream.channels = n_channels
        stream.layout = 'mono' if n_channels == 1 else 'stereo'

        # PyAV expects (channels, samples) in the stream's sample format.
        # Convert float64 → signed 16-bit PCM for broad codec compatibility.
        pcm = np.clip(audio_data.T, -1.0, 1.0)  # (channels, samples)
        pcm_s16 = (pcm * 32767).astype(np.int16)

        # Encode in chunks to keep memory reasonable
        chunk_size = sample_rate  # 1 second at a time
        for start in range(0, n_samples, chunk_size):
            chunk = pcm_s16[:, start:start + chunk_size]
            frame = av.AudioFrame.from_ndarray(chunk, format='s16', layout=stream.layout)
            frame.rate = sample_rate
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush
        for packet in stream.encode(None):
            container.mux(packet)
        container.close()

    def update_selection_view(self, event=None):
        """Update the selection detail view"""
        start_idx, end_idx = self.main_canvas.get_selection_indices()
        
        if start_idx is None or end_idx is None:
            return
        
        # Extract selected audio
        selected_audio = self.audio_data[start_idx:end_idx]
        
        if len(selected_audio) == 0:
            return
        
        # Plot selected region
        self.selection_canvas.plot_waveform(
            selected_audio, 
            self.sample_rate,
            title="Selected Region"
        )
        
        # Initialize volume envelope for the selection
        duration = len(selected_audio) / self.sample_rate
        self.selection_canvas.init_volume_envelope(duration)
        
        # Initialize playback line at position 0 for selection
        self.selection_canvas.update_playback_position(0)
        
        # Update info
        duration = len(selected_audio) / self.sample_rate
        self.selection_info_label.setText(
            f'Selection: {self.main_canvas.selection_start:.3f}s - '
            f'{self.main_canvas.selection_end:.3f}s | '
            f'Duration: {duration:.3f}s | '
            f'Samples: {len(selected_audio)}'
        )
        
        self.save_selection_btn.setEnabled(True)
        self.play_selection_btn.setEnabled(True)
        self.remove_keypoint_btn.setEnabled(True)
    
    def check_selection_update(self):
        """Check if selection has changed and update view"""
        if hasattr(self.main_canvas, 'sigRegionChanged') and self.main_canvas.sigRegionChanged:
            self.main_canvas.sigRegionChanged = False
            self.update_selection_view()
    
    def save_selection(self):
        """Save the selected audio segment with volume envelope applied"""
        start_idx, end_idx = self.main_canvas.get_selection_indices()
        
        if start_idx is None or end_idx is None:
            QMessageBox.warning(self, "Warning", "Please select a region first")
            return
        
        # Get last used directory from settings
        last_dir = self.settings.value('last_directory', '')
        
        # Suggest a filename based on current file
        # soundfile can only write WAV, FLAC, OGG — force extension to .wav
        # if the source format isn't writable
        _writable_exts = {'.wav', '.flac', '.ogg'}
        suggested_name = ""
        if self.current_file:
            base_name = os.path.basename(self.current_file)
            name_without_ext, ext = os.path.splitext(base_name)
            if ext.lower() not in _writable_exts:
                ext = '.wav'
            suggested_name = os.path.join(last_dir, f"{name_without_ext}_Edited{ext}")
        else:
            suggested_name = os.path.join(last_dir, "Edited.wav")

        # Get save file path
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Selected Audio",
            suggested_name,
            "WAV Files (*.wav);;FLAC Files (*.flac);;OGG Files (*.ogg);;MP3 Files (*.mp3);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # Save the directory for next time
            self.settings.setValue('last_directory', os.path.dirname(file_path))
            
            # Extract selected audio
            selected_audio = self.audio_data[start_idx:end_idx].copy()
            
            # Apply volume envelope
            envelope = self.selection_canvas.get_volume_envelope_array(len(selected_audio))
            
            # Apply envelope to audio
            if len(selected_audio.shape) > 1:
                # Stereo - apply to both channels
                for ch in range(selected_audio.shape[1]):
                    selected_audio[:, ch] *= envelope
            else:
                # Mono
                selected_audio *= envelope
            
            # Save — use PyAV for formats soundfile can't write (mp3, m4a, etc.)
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ('.mp3', '.m4a', '.aac', '.wma'):
                self._write_via_pyav(file_path, selected_audio, self.sample_rate)
            else:
                sf.write(file_path, selected_audio, self.sample_rate)
            
            QMessageBox.information(
                self, 
                "Success", 
                f"Selection saved with volume envelope applied to:\n{file_path}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save audio:\n{str(e)}")
    
    def reset_selection(self):
        """Reset the selection"""
        self.main_canvas.reset_selection()
        
        self.selection_canvas.clear()
        self.selection_canvas.setTitle('No selection - click on waveform to select')
        self.selection_info_label.setText('No selection made')
        self.save_selection_btn.setEnabled(False)
        self.play_selection_btn.setEnabled(False)
        self.remove_keypoint_btn.setEnabled(False)
    
    def remove_keypoint(self):
        """Remove the selected volume envelope keypoint"""
        if self.selection_canvas.remove_selected_keypoint():
            self.update_envelope_info()
        else:
            # Try to show a helpful message
            if self.selection_canvas.selected_keypoint_index is not None:
                QMessageBox.information(
                    self,
                    "Cannot Remove",
                    "Cannot remove start or end keypoints.\nYou can only remove middle keypoints."
                )
    
    def update_envelope_info(self):
        """Update the envelope info label"""
        num_keypoints = len(self.selection_canvas.volume_envelope_keypoints)
        selected_text = ""
        if self.selection_canvas.selected_keypoint_index is not None:
            idx = self.selection_canvas.selected_keypoint_index
            t, v = self.selection_canvas.volume_envelope_keypoints[idx]
            selected_text = f' | Selected: #{idx} ({t:.3f}s, {v:.2f}x)'
        
        self.envelope_info_label.setText(
            f'Volume Envelope: {num_keypoints} keypoints{selected_text} | '
            f'Right-click to add | Left-click to select/deselect | Drag selected to move | Del to remove'
        )
    
    def update_envelope_dragging(self):
        """Update keypoint position during dragging"""
        # This would be called during drag operations
        # For now, the ScatterPlotItem handles its own dragging
        pass
    
    def toggle_play_full(self):
        """Toggle playback of full audio"""
        if self.is_playing_full:
            self.stop_playback()
        else:
            self.play_audio(self.audio_data, is_selection=False)
    
    def toggle_play_selection(self):
        """Toggle playback of selected audio"""
        if self.is_playing_selection:
            self.stop_playback()
        else:
            start_idx, end_idx = self.main_canvas.get_selection_indices()
            if start_idx is not None and end_idx is not None:
                selected_audio = self.audio_data[start_idx:end_idx].copy()
                
                # Apply volume envelope to the audio before playback
                envelope = self.selection_canvas.get_volume_envelope_array(len(selected_audio))
                
                if len(selected_audio.shape) > 1:
                    # Stereo - apply to both channels
                    for ch in range(selected_audio.shape[1]):
                        selected_audio[:, ch] *= envelope
                else:
                    # Mono
                    selected_audio *= envelope
                
                self.play_audio(selected_audio, is_selection=True)
    
    def play_audio(self, audio_data, is_selection=False):
        """Play audio data"""
        # Stop any current playback
        self.stop_playback()
        
        # Check if we should resume from current position or start fresh
        if is_selection:
            # For selection, check if there's already a playback line
            if self.selection_canvas.playback_line is not None:
                # Resume from current line position
                current_time = self.selection_canvas.playback_line.value()
                with self.playback_lock:
                    self.playback_position = int(current_time * self.sample_rate)
            else:
                # Start from beginning
                with self.playback_lock:
                    self.playback_position = 0
        else:
            # For full audio, check if there's already a playback line
            if self.main_canvas.playback_line is not None:
                # Resume from current line position
                current_time = self.main_canvas.playback_line.value()
                with self.playback_lock:
                    self.playback_position = int(current_time * self.sample_rate)
            else:
                # Start from beginning
                with self.playback_lock:
                    self.playback_position = 0
        
        # Update state
        if is_selection:
            self.is_playing_selection = True
            self.play_selection_btn.setText('⏸ Stop')
        else:
            self.is_playing_full = True
            self.play_full_btn.setText('⏸ Stop')
        
        # Store playback info
        self.playback_audio = audio_data.copy()  # Make a copy to avoid issues
        self.playback_is_selection = is_selection
        
        # Start playback
        try:
            channels = audio_data.shape[1] if len(audio_data.shape) > 1 else 1
            
            self.playback_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=channels,
                blocksize=8192,  # Even larger block size to prevent underruns
                latency='high',  # Request higher latency for more stable playback
                callback=self.audio_callback,
                finished_callback=self.playback_finished
            )
            self.playback_stream.start()
            self.playback_timer.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Playback Error", f"Failed to play audio:\n{str(e)}")
            self.stop_playback()
    
    def audio_callback(self, outdata, frames, time_info, status):
        """Audio playback callback"""
        # Use thread-safe access to playback position
        with self.playback_lock:
            current_pos = self.playback_position
        
        # Calculate the end position
        end_pos = current_pos + frames
        
        # Check if we have enough data
        if end_pos > len(self.playback_audio):
            # Partial data - fill what we can and pad with zeros
            remaining = len(self.playback_audio) - current_pos
            
            if remaining > 0:
                chunk = self.playback_audio[current_pos:]
                
                # Handle mono vs stereo
                if len(chunk.shape) == 1:
                    outdata[:remaining, 0] = chunk
                else:
                    outdata[:remaining] = chunk
                
                # Pad the rest with silence
                outdata[remaining:] = 0
            else:
                outdata[:] = 0
            
            # Update position and signal end
            with self.playback_lock:
                self.playback_position = end_pos
            raise sd.CallbackStop()
        else:
            # Normal playback - we have enough data
            chunk = self.playback_audio[current_pos:end_pos]
            
            # Handle mono vs stereo
            if len(chunk.shape) == 1:
                outdata[:, 0] = chunk
            else:
                outdata[:] = chunk
            
            # Update position
            with self.playback_lock:
                self.playback_position = end_pos
    
    def update_playback_position(self):
        """Update the playback position line"""
        if not (self.is_playing_full or self.is_playing_selection):
            return
        
        # Calculate current time based on sample position (thread-safe)
        with self.playback_lock:
            current_time = self.playback_position / self.sample_rate
        
        if self.is_playing_full:
            self.main_canvas.update_playback_position(current_time)
        elif self.is_playing_selection:
            self.selection_canvas.update_playback_position(current_time)
    
    def playback_finished(self):
        """Called when playback finishes - runs in audio thread"""
        # Emit signal to stop playback in main thread (thread-safe)
        self.playback_finished_signal.emit()
    
    def stop_playback(self):
        """Stop audio playback"""
        # Stop and close stream safely
        if self.playback_stream is not None:
            try:
                if self.playback_stream.active:
                    self.playback_stream.stop()
                self.playback_stream.close()
            except Exception:
                pass  # Ignore errors during cleanup
            finally:
                self.playback_stream = None
        
        self.playback_timer.stop()
        
        # Don't clear playback lines - keep them visible for seeking
        # self.main_canvas.clear_playback_line()
        # self.selection_canvas.clear_playback_line()
        
        # Update button states
        if self.is_playing_full:
            self.play_full_btn.setText('▶ Play Full')
            self.is_playing_full = False
        
        if self.is_playing_selection:
            self.play_selection_btn.setText('▶ Play Selection')
            self.is_playing_selection = False
    
    def seek_playback(self, new_time):
        """Seek to a specific time position during playback or when stopped"""
        # Allow seeking even when not playing
        if self.is_playing_full or self.is_playing_selection:
            # Calculate new sample position
            new_position = int(new_time * self.sample_rate)
            
            # Clamp to valid range
            new_position = max(0, min(new_position, len(self.playback_audio)))
            
            # Update position (thread-safe)
            with self.playback_lock:
                self.playback_position = new_position
        # If not playing, just allow the line to move for visual reference


def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Configure PyQtGraph for better performance
    pg.setConfigOptions(antialias=True, useOpenGL=True)
    
    editor = SoundEditor()
    editor.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
