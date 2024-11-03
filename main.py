import sys
import configparser
import matplotlib
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QTabWidget, QCheckBox, QGridLayout, QGroupBox,
    QSizePolicy, QStyle, QStyleFactory
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QFont, QPixmap, QImage

# Import OpenCV for camera feed
import cv2

# Import qdarkstyle for a modern dark theme
try:
    import qdarkstyle
except ImportError:
    qdarkstyle = None  # If qdarkstyle is not available, we'll proceed without it

from Simulation.constant import *
from Simulation.models import Hexapod

# Configuration for plotting
conf = configparser.ConfigParser()
conf.read('style.ini', encoding='utf-8')

# Camera settings for Matplotlib (elevation and azimuth)
default_elev = 20
default_azim = 30

# Default colors and sizes
body_color = conf.get("robot plotter", 'body_color', fallback='blue')
leg_color = conf.get("robot plotter", 'leg_color', fallback='red')
joint_size = int(conf.get("robot plotter", 'joint_size', fallback='5'))
head_color = conf.get("robot plotter", 'head_color', fallback='green')
head_size = int(conf.get("robot plotter", 'head_size', fallback='10'))
axis_colors = {
    'x': conf.get("axis", 'color_x', fallback='red'),
    'y': conf.get("axis", 'color_y', fallback='green'),
    'z': conf.get("axis", 'color_z', fallback='blue')
}
origin_size = int(conf.get("axis", 'origin_size', fallback='10'))
ground_color = conf.get('ground', 'color', fallback='gray')
ground_opacity = float(conf.get('ground', 'opacity', fallback='0.2'))

# Functions for drawing and updating the robot
def draw_robot(robot: Hexapod, ax):
    ax.clear()

    # Plot the robot body as a polygon
    body_verts = [(p.x, p.y, p.z) for p in robot.body.vertices]
    body_poly = Poly3DCollection([body_verts], color=body_color, alpha=0.7)
    ax.add_collection3d(body_poly)

    # Plot the body outline
    outline_x = [p.x for p in robot.body.vertices] + [robot.body.vertices[0].x]
    outline_y = [p.y for p in robot.body.vertices] + [robot.body.vertices[0].y]
    outline_z = [p.z for p in robot.body.vertices] + [robot.body.vertices[0].z]
    ax.plot3D(outline_x, outline_y, outline_z, color=leg_color, linewidth=2)

    # Plot the head
    ax.scatter3D([robot.body.head.x], [robot.body.head.y], [robot.body.head.z],
                color=head_color, s=head_size)

    # Plot the legs
    for leg in robot.legs.values():
        leg_x = [p.x for p in leg.points_global]
        leg_y = [p.y for p in leg.points_global]
        leg_z = [p.z for p in leg.points_global]
        ax.plot3D(leg_x, leg_y, leg_z, color=leg_color, linewidth=2)
        ax.scatter3D(leg_x, leg_y, leg_z, color=leg_color, s=joint_size)

    # Plot the support mesh (contact points)
    support_x = [p.x for p in robot.ground_contact_points.values()]
    support_y = [p.y for p in robot.ground_contact_points.values()]
    support_z = [p.z - 0.01 for p in robot.ground_contact_points.values()]
    # if support_x and support_y and support_z:
    #     support_verts = list(zip(support_x, support_y, support_z))
    #     support_poly = Poly3DCollection([support_verts], color=body_color, alpha=0.2)
    #     ax.add_collection3d(support_poly)

    # Plot the ground
    s = float(conf.get('ground', 'size', fallback='20'))
    ground_x = [s / 2, -s / 2, -s / 2, s / 2, s / 2]
    ground_y = [s / 2, s / 2, -s / 2, -s / 2, s / 2]
    ground_z = [0, 0, 0, 0, 0]
    ax.plot3D(ground_x, ground_y, ground_z, color=ground_color, alpha=ground_opacity)

    # Set the axes limits
    ax.set_xlim([-8, 8])
    ax.set_ylim([-8, 8])
    ax.set_zlim([-8, 8])
    ax.axis('off')

    ax.view_init(elev=default_elev, azim=default_azim)

def play_robot_walking(robot: Hexapod, t):
    robot.set_pose_from_walking_sequence(t)

# Thread class for handling camera feed
class CameraThread(QThread):
    frame_updated = pyqtSignal(QImage)

    def __init__(self, camera_index=0, width=640, height=480):
        super().__init__()
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.running = False

    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)  # Set frame width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)  # Set frame height
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Convert the image to RGB format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                # Emit the signal
                self.frame_updated.emit(qt_image)
            else:
                break
        self.cap.release()

    def stop(self):
        self.running = False
        self.wait()

class SimulationTab(QWidget):
    def __init__(self, robot: Hexapod, update_plot_callback):
        super().__init__()
        self.robot = robot
        self.update_plot_callback = update_plot_callback
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        self.setLayout(layout)

        # Controls Layout
        controls_layout = QVBoxLayout()
        controls_layout.setAlignment(Qt.AlignTop)

        # GroupBox for Dimension controls
        self.dim_group = QGroupBox("Dimension Settings")
        self.dim_layout = QVBoxLayout()
        self.dim_group.setLayout(self.dim_layout)

        dim_labels = ['Front', 'Middle', 'Side', 'coxa', 'Femur', 'Tibia']
        default_values = DEFAULT_DIMSIONS + DEFAULT_LEG_LENGTH
        self.dim_sliders = {}

        for label, value in zip(dim_labels, default_values):
            lbl = QLabel(label)
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(1)
            slider.setMaximum(20)
            slider.setValue(value)
            slider.setTickInterval(1)
            slider.setTickPosition(QSlider.TicksBelow)
            lbl.setFont(QFont("Arial", 10))
            self.dim_layout.addWidget(lbl)
            self.dim_layout.addWidget(slider)
            self.dim_sliders[label] = slider

        # Dimension control buttons
        button_layout = QHBoxLayout()
        self.reset_dim_button = QPushButton("Reset Dimensions")
        self.reset_view_button = QPushButton("Reset 3D View")
        self.reset_dim_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.reset_view_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowBack))
        button_layout.addWidget(self.reset_dim_button)
        button_layout.addWidget(self.reset_view_button)
        self.dim_layout.addLayout(button_layout)

        controls_layout.addWidget(self.dim_group)

        # Tabs for simulation controls
        self.simulation_tabs = QTabWidget()
        controls_layout.addWidget(self.simulation_tabs)

        # Leg Pattern Tab
        self.init_leg_pattern_tab()

        # Inverse Kinematics Tab
        self.init_ik_tab()

        # Gait Control Tab
        self.init_gait_tab()

        layout.addLayout(controls_layout)

        # Placeholder for plot - to be added by the main window
        # The main window will add the plot separately

    def init_leg_pattern_tab(self):
        self.leg_tab = QWidget()
        self.leg_tab_layout = QVBoxLayout()
        self.leg_tab.setLayout(self.leg_tab_layout)

        self.leg_tab_layout.addWidget(QLabel("<i>Legs share the same pose</i>"))

        leg_labels = [r'α (coxa-zaxis)', r'β (femur-xaxis)', r'γ (tibia-xaxis)']
        self.leg_sliders = {}

        for label in leg_labels:
            lbl = QLabel(label)
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-180)
            slider.setMaximum(180)
            slider.setValue(0)
            slider.setTickInterval(10)
            slider.setTickPosition(QSlider.TicksBelow)
            lbl.setFont(QFont("Arial", 10))
            self.leg_tab_layout.addWidget(lbl)
            self.leg_tab_layout.addWidget(slider)
            self.leg_sliders[label.split()[0]] = slider  # Using 'α', 'β', 'γ' as keys

        self.simulation_tabs.addTab(self.leg_tab, "Leg Pattern")

    def init_ik_tab(self):
        self.ik_tab = QWidget()
        self.ik_tab_layout = QVBoxLayout()
        self.ik_tab.setLayout(self.ik_tab_layout)

        self.ik_sliders = {}
        axes = ['X', 'Y', 'Z']
        translations = ['TX', 'TY', 'TZ']
        rotations = ['RX', 'RY', 'RZ']

        # Translation sliders
        trans_group = QGroupBox("Translations")
        trans_layout = QGridLayout()
        trans_group.setLayout(trans_layout)
        for i, (axis, label) in enumerate(zip(axes, translations)):
            lbl = QLabel(label)
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-100)  # Use -100 to 100 to represent -1 to 1 in steps of 0.01
            slider.setMaximum(100)
            slider.setValue(0)
            slider.setTickInterval(10)
            slider.setTickPosition(QSlider.TicksBelow)
            lbl.setFont(QFont("Arial", 10))
            trans_layout.addWidget(lbl, i, 0)
            trans_layout.addWidget(slider, i, 1)
            self.ik_sliders[label] = slider

        self.ik_tab_layout.addWidget(trans_group)

        # Rotation sliders
        rot_group = QGroupBox("Rotations")
        rot_layout = QGridLayout()
        rot_group.setLayout(rot_layout)
        for i, (axis, label) in enumerate(zip(axes, rotations)):
            lbl = QLabel(label)
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(-30)
            slider.setMaximum(30)
            slider.setValue(0)
            slider.setTickInterval(1)
            slider.setTickPosition(QSlider.TicksBelow)
            lbl.setFont(QFont("Arial", 10))
            rot_layout.addWidget(lbl, i, 0)
            rot_layout.addWidget(slider, i, 1)
            self.ik_sliders[label] = slider

        self.ik_tab_layout.addWidget(rot_group)

        self.simulation_tabs.addTab(self.ik_tab, "Inverse Kinematics")

    def init_gait_tab(self):
        self.gait_tab = QWidget()
        self.gait_tab_layout = QVBoxLayout()
        self.gait_tab.setLayout(self.gait_tab_layout)

        self.gait_sliders = {}
        gait_params = [
            ('LiftSwing', 10, 40, 20),
            ('HipSwing', 10, 40, 12),
            ('GaitStep', 5, 20, 10),
            ('GaitSpeed', 5, 20, 10)
        ]

        gait_group = QGroupBox("Gait Parameters")
        gait_layout = QGridLayout()
        gait_group.setLayout(gait_layout)

        for i, (label, min_val, max_val, default_val) in enumerate(gait_params):
            lbl = QLabel(label)
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(default_val)
            slider.setTickInterval(1)
            slider.setTickPosition(QSlider.TicksBelow)
            lbl.setFont(QFont("Arial", 10))
            gait_layout.addWidget(lbl, i, 0)
            gait_layout.addWidget(slider, i, 1)
            self.gait_sliders[label] = slider

        self.gait_tab_layout.addWidget(gait_group)

        # Checkboxes
        self.is_tripod_cb = QCheckBox("Tripod Gait")
        self.is_forward_cb = QCheckBox("Forward Direction")
        self.is_rotate_cb = QCheckBox("Rotate Body")
        self.is_tripod_cb.setChecked(True)
        self.is_forward_cb.setChecked(True)

        self.gait_tab_layout.addWidget(self.is_tripod_cb)
        self.gait_tab_layout.addWidget(self.is_forward_cb)
        self.gait_tab_layout.addWidget(self.is_rotate_cb)

        # Buttons
        button_layout = QHBoxLayout()
        self.gait_play_button = QPushButton("Play")
        self.gait_pause_button = QPushButton("Pause")
        self.gait_step_button = QPushButton("Step")

        # Add icons to buttons
        self.gait_play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.gait_pause_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.gait_step_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipForward))

        button_layout.addWidget(self.gait_play_button)
        button_layout.addWidget(self.gait_pause_button)
        button_layout.addWidget(self.gait_step_button)

        self.gait_tab_layout.addLayout(button_layout)

        self.simulation_tabs.addTab(self.gait_tab, "Walking Gaits")

class ControlTab(QWidget):
    def __init__(self, robot: Hexapod, update_plot_callback):
        super().__init__()
        self.robot = robot
        self.update_plot_callback = update_plot_callback
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Camera Feed Display
        self.camera_label = QLabel("Camera Output")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("border: 1px solid black;")
        self.camera_label.setFixedSize(640, 480)  # Set fixed width and height
        layout.addWidget(self.camera_label, alignment=Qt.AlignCenter)

        # Control Buttons Group
        control_group = QGroupBox("Hexapod Controls")
        control_layout = QHBoxLayout()
        control_group.setLayout(control_layout)
        control_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        control_group.setFixedHeight(100)  # Adjusted height for control buttons

        self.forward_button = QPushButton("Forward")
        self.backward_button = QPushButton("Backward")
        self.left_button = QPushButton("Left")
        self.right_button = QPushButton("Right")
        self.stop_button = QPushButton("Stop")

        # Add icons to control buttons
        self.forward_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowUp))
        self.backward_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowDown))
        self.left_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowLeft))
        self.right_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))

        # Set size policies to prevent buttons from expanding
        buttons = [
            self.forward_button,
            self.backward_button,
            self.left_button,
            self.right_button,
            self.stop_button
        ]
        for btn in buttons:
            btn.setFixedSize(100, 40)  # Set fixed size for each button
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        control_layout.addStretch()
        control_layout.addWidget(self.forward_button)
        control_layout.addWidget(self.backward_button)
        control_layout.addWidget(self.left_button)
        control_layout.addWidget(self.right_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addStretch()

        layout.addWidget(control_group)

        # Start and Stop Camera Buttons
        camera_control_layout = QHBoxLayout()
        self.start_camera_button = QPushButton("Start Camera")
        self.stop_camera_button = QPushButton("Stop Camera")
        self.stop_camera_button.setEnabled(False)

        # Add icons to camera control buttons
        self.start_camera_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.stop_camera_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

        # Set fixed sizes for camera control buttons
        self.start_camera_button.setFixedSize(120, 40)
        self.stop_camera_button.setFixedSize(120, 40)

        camera_control_layout.addStretch()
        camera_control_layout.addWidget(self.start_camera_button)
        camera_control_layout.addWidget(self.stop_camera_button)
        camera_control_layout.addStretch()

        layout.addLayout(camera_control_layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hexapod Simulator")
        self.resize(1200, 800)

        self.robot = Hexapod()
        self.gait_step = 0

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')

        self.init_ui()
        self.setup_connections()
        self.update_plot()

    def init_ui(self):
        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout()
        self.main_widget.setLayout(self.main_layout)

        # Main Tab Widget
        self.main_tabs = QTabWidget()
        self.main_layout.addWidget(self.main_tabs)

        # Simulation Tab
        self.simulation_tab = SimulationTab(self.robot, self.update_plot)
        # Add the 3D plot to the Simulation Tab
        simulation_layout = QHBoxLayout()
        simulation_layout.addWidget(self.simulation_tab)
        # Embed the plot in the simulation tab
        simulation_layout.addWidget(self.canvas, stretch=1)
        simulation_container = QWidget()
        simulation_container.setLayout(simulation_layout)
        self.main_tabs.addTab(simulation_container, "Simulation")

        # Control Tab
        self.control_tab = ControlTab(self.robot, self.update_plot)
        self.main_tabs.addTab(self.control_tab, "Control")

        # Apply styles
        self.apply_styles()

    def apply_styles(self):
        font = QFont("Arial", 10)
        self.setFont(font)

        # Dark theme using qdarkstyle if available
        if qdarkstyle:
            self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        else:
            QApplication.setStyle(QStyleFactory.create('Fusion'))
            palette = QApplication.palette()
            palette.setColor(palette.Window, Qt.gray)
            palette.setColor(palette.WindowText, Qt.white)
            QApplication.setPalette(palette)

    def setup_connections(self):
        # Simulation Tab Connections
        # Dimension sliders
        for slider in self.simulation_tab.dim_sliders.values():
            slider.valueChanged.connect(self.on_dimension_changed)

        # Reset buttons
        self.simulation_tab.reset_dim_button.clicked.connect(self.reset_dimensions)
        self.simulation_tab.reset_view_button.clicked.connect(self.reset_view)

        # Leg pattern sliders
        for slider in self.simulation_tab.leg_sliders.values():
            slider.valueChanged.connect(self.on_leg_pattern_changed)

        # Inverse Kinematics sliders
        for slider in self.simulation_tab.ik_sliders.values():
            slider.valueChanged.connect(self.on_ik_changed)

        # Gait sliders and checkboxes
        for slider in self.simulation_tab.gait_sliders.values():
            slider.valueChanged.connect(self.on_gait_parameters_changed)
        self.simulation_tab.is_tripod_cb.stateChanged.connect(self.on_gait_parameters_changed)
        self.simulation_tab.is_forward_cb.stateChanged.connect(self.on_gait_parameters_changed)
        self.simulation_tab.is_rotate_cb.stateChanged.connect(self.on_gait_parameters_changed)

        self.simulation_tab.gait_play_button.clicked.connect(self.on_gait_play)
        self.simulation_tab.gait_pause_button.clicked.connect(self.on_gait_pause)
        self.simulation_tab.gait_step_button.clicked.connect(self.on_gait_step)

        # Timer for gait animation
        self.gait_timer = QTimer()
        self.gait_timer.timeout.connect(self.update_gait)

        # Control Tab Connections
        # Camera controls
        self.control_tab.start_camera_button.clicked.connect(self.start_camera)
        self.control_tab.stop_camera_button.clicked.connect(self.stop_camera)

        # Hexapod control buttons
        self.control_tab.forward_button.clicked.connect(self.move_forward)
        self.control_tab.backward_button.clicked.connect(self.move_backward)
        self.control_tab.left_button.clicked.connect(self.turn_left)
        self.control_tab.right_button.clicked.connect(self.turn_right)
        self.control_tab.stop_button.clicked.connect(self.stop_movement)

    # Simulation Tab Methods
    def on_dimension_changed(self):
        values = [self.simulation_tab.dim_sliders[label].value() for label in ['Front', 'Middle', 'Side', 'coxa', 'Femur', 'Tibia']]
        if self.robot.update_dimensions(values):
            self.update_plot()

    def on_leg_pattern_changed(self):
        values = [self.simulation_tab.leg_sliders[label].value() for label in ['α', 'β', 'γ']]
        if self.robot.update_leg_pattern(values):
            self.update_plot()

    def reset_dimensions(self):
        default_values = DEFAULT_DIMSIONS + DEFAULT_LEG_LENGTH
        for label, value in zip(['Front', 'Middle', 'Side', 'coxa', 'Femur', 'Tibia'], default_values):
            self.simulation_tab.dim_sliders[label].setValue(value)

    def reset_view(self):
        self.ax.view_init(elev=default_elev, azim=default_azim)
        self.update_plot()

    def on_ik_changed(self):
        rx = self.simulation_tab.ik_sliders['RX'].value()
        ry = self.simulation_tab.ik_sliders['RY'].value()
        rz = self.simulation_tab.ik_sliders['RZ'].value()
        tx = self.simulation_tab.ik_sliders['TX'].value() * 0.01 * self.robot.body.f
        ty = self.simulation_tab.ik_sliders['TY'].value() * 0.01 * self.robot.body.s
        tz = self.simulation_tab.ik_sliders['TZ'].value() * 0.01 * self.robot.legs[0].lengths[-1]
        self.robot.solve_ik([rx, ry, rz], [tx, ty, tz])
        self.update_plot()

    def on_gait_parameters_changed(self):
        ls = self.simulation_tab.gait_sliders['LiftSwing'].value()
        hs = self.simulation_tab.gait_sliders['HipSwing'].value()
        st = self.simulation_tab.gait_sliders['GaitStep'].value()
        sp = self.simulation_tab.gait_sliders['GaitSpeed'].value()
        is_tripod = self.simulation_tab.is_tripod_cb.isChecked()
        is_forward = self.simulation_tab.is_forward_cb.isChecked()
        is_rotate = self.simulation_tab.is_rotate_cb.isChecked()

        para = {}
        para['HipSwing'] = hs
        para['LiftSwing'] = ls
        para['StepNum'] = st
        para['Speed'] = sp
        para['Gait'] = 'Tripod' if is_tripod else 'Ripple'
        para['Direction'] = 1 if is_forward else -1
        para['Rotation'] = 1 if is_rotate else 0
        self.robot.generate_walking_sequence(para)
        self.gait_timer.stop()
        self.gait_step = 0

    def on_gait_play(self):
        # Adjust timer interval based on gait speed
        sp = self.simulation_tab.gait_sliders['GaitSpeed'].value()
        if sp == 0:
            return  # Avoid division by zero
        interval = max(10, int(1000 / sp))  # Minimum interval of 10ms
        self.gait_timer.start(interval)

    def on_gait_pause(self):
        self.gait_timer.stop()

    def on_gait_step(self):
        self.gait_step += 1
        self.update_gait()

    def update_gait(self):
        if not self.robot.walking_sequence:
            return
        num_steps = len(next(iter(self.robot.walking_sequence[0].values())))
        t = self.gait_step % num_steps
        play_robot_walking(self.robot, t)
        self.update_plot()
        self.gait_step += 1

    # Control Tab Methods
    def start_camera(self):
        self.camera_thread = CameraThread()
        self.camera_thread.frame_updated.connect(self.update_camera_feed)
        self.camera_thread.start()
        self.control_tab.start_camera_button.setEnabled(False)
        self.control_tab.stop_camera_button.setEnabled(True)

    def stop_camera(self):
        if hasattr(self, 'camera_thread') and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread = None
        self.control_tab.camera_label.clear()
        self.control_tab.start_camera_button.setEnabled(True)
        self.control_tab.stop_camera_button.setEnabled(False)

    def update_camera_feed(self, image):
        self.control_tab.camera_label.setPixmap(QPixmap.fromImage(image).scaled(
            self.control_tab.camera_label.width(),
            self.control_tab.camera_label.height(),
            Qt.KeepAspectRatio
        ))

    # Hexapod Control Methods
    def move_forward(self):
        print("Hexapod moving forward")
        # Implement the logic to move the hexapod forward
        self.robot.move_forward()
        self.update_plot()

    def move_backward(self):
        print("Hexapod moving backward")
        # Implement the logic to move the hexapod backward
        self.robot.move_backward()
        self.update_plot()

    def turn_left(self):
        print("Hexapod turning left")
        # Implement the logic to turn the hexapod left
        self.robot.turn_left()
        self.update_plot()

    def turn_right(self):
        print("Hexapod turning right")
        # Implement the logic to turn the hexapod right
        self.robot.turn_right()
        self.update_plot()

    def stop_movement(self):
        print("Hexapod stopped")
        # Implement the logic to stop the hexapod's movement
        self.robot.stop()
        self.update_plot()

    def update_plot(self):
        draw_robot(self.robot, self.ax)
        self.ax.grid(True)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Adjust margins and padding
        self.figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
