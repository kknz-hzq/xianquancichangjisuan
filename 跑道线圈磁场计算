import sys
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
import csv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QGroupBox, QFormLayout, QTextEdit, QProgressBar,
                             QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
                             QFileDialog, QSplitter, QCheckBox, QComboBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import matplotlib
# 新增的导入，用于云图插值
from scipy.interpolate import griddata

matplotlib.rc("font", family="Microsoft YaHei")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

mu0 = 4.0 * np.pi * 1e-7


# ---------------- utilities ----------------
def kahan_vector_sum(vectors):
    """Kahan compensated summation for vectors (iterable of length-3 arrays)."""
    s = np.zeros(3)
    c = np.zeros(3)
    for v in vectors:
        y = v - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


# ------------------ High-precision Biot-Savart core (Original) ------------------
def integrate_segment(r_func, drdt_func, t0, t1, obs, n_gauss=200, eps=1e-15):
    nodes, weights = leggauss(n_gauss)
    t = 0.5 * (t1 - t0) * nodes + 0.5 * (t1 + t0)
    dt = 0.5 * (t1 - t0)

    pts = np.array([r_func(tt) for tt in t])
    drdt = np.array([drdt_func(tt) for tt in t])

    R = obs - pts
    Rnorm = np.linalg.norm(R, axis=1)
    Rnorm = np.where(Rnorm < eps, eps, Rnorm)

    cross = np.cross(drdt, R)
    integrand = (cross.T / (Rnorm ** 3)).T

    integral = np.sum(integrand * weights[:, None], axis=0) * dt
    return mu0 / (4.0 * np.pi) * integral


# ------------------ Vectorized Biot-Savart core (Fast) ------------------
def integrate_segment_fast(r_vec_func, drdt_vec_func, t0, t1, obs, n_gauss=200, eps=1e-15):
    nodes, weights = leggauss(n_gauss)
    t_nodes = 0.5 * (t1 - t0) * nodes + 0.5 * (t1 + t0)
    dt = 0.5 * (t1 - t0)

    # Vectorized calls
    pts = r_vec_func(t_nodes)
    drdt = drdt_vec_func(t_nodes)

    R = obs - pts
    Rnorm = np.linalg.norm(R, axis=1, keepdims=True)
    Rnorm[Rnorm < eps] = eps

    cross = np.cross(drdt, R)
    integrand = cross / (Rnorm ** 3)

    integral = np.sum(integrand * weights[:, None], axis=0) * dt
    return mu0 / (4.0 * np.pi) * integral


# ---- Arc functions ----
def arc_r(th, center, radius, u_offset, v_offset):
    n_hat = np.array([np.cos(th), np.sin(th), 0.0])
    return np.array(center) + (radius + u_offset) * n_hat + np.array([0.0, 0.0, v_offset])


def arc_drdth(th, radius, u_offset):
    return np.array([-(radius + u_offset) * np.sin(th), (radius + u_offset) * np.cos(th), 0.0])


def arc_r_vec(th_nodes, center, radius, u_offset, v_offset):
    n_hat = np.array([np.cos(th_nodes), np.sin(th_nodes), np.zeros_like(th_nodes)]).T
    return center + (radius + u_offset) * n_hat + np.array([0.0, 0.0, v_offset])


def arc_drdth_vec(th_nodes, radius, u_offset):
    return np.array(
        [-(radius + u_offset) * np.sin(th_nodes), (radius + u_offset) * np.cos(th_nodes), np.zeros_like(th_nodes)]).T


# ---- Straight segment functions ----
def straight_r(s, A, vec, n_hat, u_offset, v_offset, b_hat):
    return A + vec * s + u_offset * n_hat + v_offset * b_hat


def straight_drds(vec):
    return vec


def straight_r_vec(s_nodes, A, vec, n_hat, u_offset, v_offset, b_hat):
    return A + np.outer(s_nodes, vec) + u_offset * n_hat + v_offset * b_hat


def straight_drds_vec(s_nodes, vec):
    return np.tile(vec, (len(s_nodes), 1))


# --- B-field calculation for a single turn (both modes) ---
def racetrack_single_turn_B(I, straight_len, radius, obs, n_line, n_arc, u_offset, v_offset, fast_mode):
    C_top = np.array([0.0, straight_len / 2.0, 0.0])
    C_bot = np.array([0.0, -straight_len / 2.0, 0.0])
    A = np.array([radius, -straight_len / 2.0, 0.0])
    Bp = np.array([radius, straight_len / 2.0, 0.0])
    C = np.array([-radius, straight_len / 2.0, 0.0])
    D = np.array([-radius, -straight_len / 2.0, 0.0])

    vec_right = Bp - A
    vec_left = D - C
    t_hat_right = vec_right / np.linalg.norm(vec_right) if np.linalg.norm(vec_right) > 0 else np.array([0., 1., 0.])
    n_hat_right = np.cross(t_hat_right, np.array([0., 0., 1.]))
    t_hat_left = vec_left / np.linalg.norm(vec_left) if np.linalg.norm(vec_left) > 0 else np.array([0., -1., 0.])
    n_hat_left = np.cross(t_hat_left, np.array([0., 0., 1.]))
    b_hat = np.array([0., 0., 1.])

    if not fast_mode:
        B_right = integrate_segment(lambda s: straight_r(s, A, vec_right, n_hat_right, u_offset, v_offset, b_hat),
                                    lambda s: straight_drds(vec_right), 0.0, 1.0, obs, n_gauss=n_line)
        B_top = integrate_segment(lambda th: arc_r(th, C_top, radius, u_offset, v_offset),
                                  lambda th: arc_drdth(th, radius, u_offset), 0.0, np.pi, obs, n_gauss=n_arc)
        B_left = integrate_segment(lambda s: straight_r(s, C, vec_left, n_hat_left, u_offset, v_offset, b_hat),
                                   lambda s: straight_drds(vec_left), 0.0, 1.0, obs, n_gauss=n_line)
        B_bottom = integrate_segment(lambda th: arc_r(th, C_bot, radius, u_offset, v_offset),
                                     lambda th: arc_drdth(th, radius, u_offset), np.pi, 2.0 * np.pi, obs, n_gauss=n_arc)
    else:  # Fast mode
        B_right = integrate_segment_fast(
            lambda s_nodes: straight_r_vec(s_nodes, A, vec_right, n_hat_right, u_offset, v_offset, b_hat),
            lambda s_nodes: straight_drds_vec(s_nodes, vec_right), 0.0, 1.0, obs, n_gauss=n_line)
        B_top = integrate_segment_fast(lambda th_nodes: arc_r_vec(th_nodes, C_top, radius, u_offset, v_offset),
                                       lambda th_nodes: arc_drdth_vec(th_nodes, radius, u_offset), 0.0, np.pi, obs,
                                       n_gauss=n_arc)
        B_left = integrate_segment_fast(
            lambda s_nodes: straight_r_vec(s_nodes, C, vec_left, n_hat_left, u_offset, v_offset, b_hat),
            lambda s_nodes: straight_drds_vec(s_nodes, vec_left), 0.0, 1.0, obs, n_gauss=n_line)
        B_bottom = integrate_segment_fast(lambda th_nodes: arc_r_vec(th_nodes, C_bot, radius, u_offset, v_offset),
                                          lambda th_nodes: arc_drdth_vec(th_nodes, radius, u_offset), np.pi,
                                          2.0 * np.pi, obs, n_gauss=n_arc)

    return I * kahan_vector_sum([B_right, B_top, B_left, B_bottom])


# ------------------ Background calculation thread ------------------
class FieldCalculationThread(QThread):
    calculation_done = pyqtSignal(list)
    calculation_progress = pyqtSignal(str, int, int)
    error_occurred = pyqtSignal(str)

    def __init__(self, params, points):
        super().__init__()
        self.params = params
        self.points = points

    def run(self):
        try:
            results = []
            total = len(self.points)
            for idx, (x, y, z) in enumerate(self.points):
                self.calculation_progress.emit(f"计算点 {idx + 1}/{total}", idx + 1, total)
                start = time.time()
                B = self.calculate_racetrack_field(self.params, np.array([x, y, z]))
                elapsed = time.time() - start

                magnitude = np.linalg.norm(B)
                signed_magnitude = np.copysign(magnitude, B[2])

                results.append({'x': x, 'y': y, 'z': z,
                                'Bx': B[0], 'By': B[1], 'Bz': B[2],
                                'B_mag': magnitude,
                                'B_signed_z': signed_magnitude,
                                'time': elapsed})
            self.calculation_done.emit(results)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def _calculate_single_racetrack_field(self, params, obs, current_override=None):
        current = current_override if current_override is not None else params['current']
        N_radial = params['N_radial']
        N_turns_axial = params['N_turns_axial']
        r_inner = params['r_inner']
        straight_len = params['straight_len']
        n_line = params['n_line']
        n_arc = params['n_arc']
        fast_mode = params.get('fast_mode', False)
        use_J = params.get('use_J', False)

        physical_width = params['radial_width']
        physical_thickness = params['axial_thickness']
        wire_diameter = params['wire_diameter']

        if N_radial > 1:
            # The distance between centers of the first and last wire in the radial direction
            # is physical_width - wire_diameter.
            # This distance is covered by N_radial - 1 steps.
            dr = (physical_width - wire_diameter) / (N_radial - 1)
        else:
            dr = 0.0

        if N_turns_axial > 1:
            # The distance between centers of the first and last wire in the axial direction
            # is physical_thickness - wire_diameter.
            # This distance is covered by N_turns_axial - 1 steps.
            dz = (physical_thickness - wire_diameter) / (N_turns_axial - 1)
        else:
            dz = 0.0

        # These are offsets from the nominal center of the coil winding pack
        # The total width is (N_radial - 1) * dr
        # The total thickness is (N_turns_axial - 1) * dz
        radial_offset_start = -(N_radial - 1) * dr / 2.0
        axial_offset_start = -(N_turns_axial - 1) * dz / 2.0

        J = params.get('J', 0.0)
        wire_w = params.get('wire_width', 0.003)
        wire_t = params.get('wire_thickness', 0.002)
        n_w = params.get('n_width', 7)
        n_t = params.get('n_thick', 5)

        B_accumulate = []
        if use_J:
            dA = (wire_w / float(n_w)) * (wire_t / float(n_t))
            I_each_fil = np.sign(current) * abs(J) * dA
        else:
            I_each_fil = None

        for j in range(N_radial):
            # The radius of the j-th layer
            radius_offset = radial_offset_start + j * dr
            radius = r_inner + radius_offset

            for i_turn in range(N_turns_axial):
                # The z-offset of the i-th turn
                z_off = axial_offset_start + i_turn * dz
                obs_shifted = obs - np.array([0.0, 0.0, z_off])

                if use_J:
                    xs = np.linspace(-wire_w / 2.0 + (wire_w / (2.0 * n_w)), wire_w / 2.0 - (wire_w / (2.0 * n_w)), n_w)
                    vs = np.linspace(-wire_t / 2.0 + (wire_t / (2.0 * n_t)), wire_t / 2.0 - (wire_t / (2.0 * n_t)), n_t)
                    for u in xs:
                        for v in vs:
                            Bf = racetrack_single_turn_B(I_each_fil, straight_len, radius, obs_shifted,
                                                         n_line, n_arc, u, v, fast_mode)
                            B_accumulate.append(Bf)
                else:
                    Bf = racetrack_single_turn_B(current, straight_len, radius, obs_shifted,
                                                 n_line, n_arc, 0.0, 0.0, fast_mode)
                    B_accumulate.append(Bf)

        B_total = kahan_vector_sum(B_accumulate) if B_accumulate else np.zeros(3)
        return B_total

    def calculate_racetrack_field(self, params, obs):
        use_two_coils = params.get('use_two_coils', False)
        if not use_two_coils:
            return self._calculate_single_racetrack_field(params, obs)
        else:
            distance = params['coil_distance']
            I_magnitude = abs(params['current'])
            current_sign_1 = 1 if params['coil1_dir'] == 'Counter-clockwise' else -1
            current_sign_2 = 1 if params['coil2_dir'] == 'Counter-clockwise' else -1
            current_1 = current_sign_1 * I_magnitude
            current_2 = current_sign_2 * I_magnitude

            obs_1 = obs - np.array([0.0, 0.0, distance / 2.0])
            B1 = self._calculate_single_racetrack_field(params, obs_1, current_override=current_1)

            obs_2 = obs - np.array([0.0, 0.0, -distance / 2.0])
            B2 = self._calculate_single_racetrack_field(params, obs_2, current_override=current_2)

            return B1 + B2


# ------------------ Simple Qt GUI ------------------
class RacetrackFieldApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Racetrack (矩形+半圆) 线圈磁场计算 - 高级版')
        # <--- 修改点 1: 增大主窗口高度
        self.setGeometry(80, 80, 1400, 950)
        self.points = []
        self.results = []
        self.calc_thread = None

        main = QWidget()
        layout = QHBoxLayout()
        main.setLayout(layout)
        self.setCentralWidget(main)

        left = QWidget()
        left_l = QVBoxLayout()
        left.setLayout(left_l)
        right = QWidget()
        right_l = QVBoxLayout()
        right.setLayout(right_l)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([450, 950])
        layout.addWidget(splitter)

        # --- parameters ---
        pg = QGroupBox('线圈参数 (Coil Parameters)')
        pfl = QFormLayout()
        self.I_input = QLineEdit('1.0')
        self.N_radial_input = QLineEdit('10')
        self.N_turns_input = QLineEdit('10')
        self.r_inner_input = QLineEdit('0.025')
        self.radial_width_input = QLineEdit('0.03')
        self.axial_thickness_input = QLineEdit('0.02')
        self.wire_diameter_input = QLineEdit('0.002')
        self.straight_len_input = QLineEdit('0.05')
        self.n_line_input = QLineEdit('100')
        self.n_arc_input = QLineEdit('120')
        self.fast_mode_checkbox = QCheckBox('启用快速计算 (Enable Fast Calculation)')
        self.fast_mode_checkbox.setChecked(True)
        pfl.addRow(self.fast_mode_checkbox)
        pfl.addRow('电流大小 |I| (A):', self.I_input)
        pfl.addRow('径向匝数 N_radial:', self.N_radial_input)
        pfl.addRow('轴向匝数 N_turns_axial:', self.N_turns_input)
        pfl.addRow('内半径 r_inner (m):', self.r_inner_input)
        pfl.addRow('总宽度 W_p (m):', self.radial_width_input)
        pfl.addRow('总厚度 T_p (m):', self.axial_thickness_input)
        pfl.addRow('导线直径 d (m):', self.wire_diameter_input)
        pfl.addRow('直线段长度 L (m):', self.straight_len_input)
        pfl.addRow('线段 Gauss 点 n_line:', self.n_line_input)
        pfl.addRow('弧段 Gauss 点 n_arc:', self.n_arc_input)
        pg.setLayout(pfl)
        left_l.addWidget(pg)

        # --- J-mode controls ---
        pg_j = QGroupBox('使用电流密度J (Use J)')
        pg_j.setCheckable(True)
        pg_j.setChecked(False)
        self.use_J_checkbox = pg_j
        pfl_j = QFormLayout()
        pg_j.setLayout(pfl_j)
        self.J_input = QLineEdit('1.667e5')
        self.wire_w_input = QLineEdit('0.003')
        self.wire_t_input = QLineEdit('0.002')
        self.n_w_input = QLineEdit('7')
        self.n_t_input = QLineEdit('5')
        pfl_j.addRow('J (A/m^2):', self.J_input)
        pfl_j.addRow('wire_width (m):', self.wire_w_input)
        pfl_j.addRow('wire_thickness (m):', self.wire_t_input)
        pfl_j.addRow('filaments n_w:', self.n_w_input)
        pfl_j.addRow('filaments n_t:', self.n_t_input)
        left_l.addWidget(pg_j)

        # --- Dual Coil Settings ---
        pg_dual = QGroupBox('双线圈耦合 (Dual Coil)')
        pg_dual.setCheckable(True)
        pg_dual.setChecked(False)
        self.use_two_coils_checkbox = pg_dual
        pfl_dual = QFormLayout()
        pg_dual.setLayout(pfl_dual)
        self.coil_distance_input = QLineEdit("0.1")
        self.coil1_current_dir_input = QComboBox()
        self.coil1_current_dir_input.addItems(["Counter-clockwise", "Clockwise"])
        self.coil2_current_dir_input = QComboBox()
        self.coil2_current_dir_input.addItems(["Counter-clockwise", "Clockwise"])
        pfl_dual.addRow("z轴间距 (m):", self.coil_distance_input)
        pfl_dual.addRow("线圈1 (+z) 方向:", self.coil1_current_dir_input)
        pfl_dual.addRow("线圈2 (-z) 方向:", self.coil2_current_dir_input)
        left_l.addWidget(pg_dual)

        # --- Multi-Point Generation ---
        pg_gen = QGroupBox('多点生成 (Multi-Point Generation)')
        self.gen_layout = QFormLayout()
        pg_gen.setLayout(self.gen_layout)
        self.gen_mode_combo = QComboBox()
        self.gen_mode_combo.addItems(['Z-axis', 'XY-plane', 'YZ-plane', 'XZ-plane'])
        self.gen_mode_combo.currentIndexChanged.connect(self.update_gen_ui)
        self.gen_layout.addRow('模式 (Mode):', self.gen_mode_combo)
        self.gen_widgets = {
            'x_start': QLineEdit('-0.1'), 'x_end': QLineEdit('0.1'), 'x_num': QLineEdit('21'),
            'y_start': QLineEdit('-0.1'), 'y_end': QLineEdit('0.1'), 'y_num': QLineEdit('21'),
            'z_start': QLineEdit('-0.6'), 'z_end': QLineEdit('0.6'), 'z_num': QLineEdit('101'),
            'x_const': QLineEdit('0.0'), 'y_const': QLineEdit('0.0'), 'z_const': QLineEdit('0.0')
        }
        self.gen_layout.addRow('X-start (m):', self.gen_widgets['x_start'])
        self.gen_layout.addRow('X-end (m):', self.gen_widgets['x_end'])
        self.gen_layout.addRow('X Num Points:', self.gen_widgets['x_num'])
        self.gen_layout.addRow('Y-start (m):', self.gen_widgets['y_start'])
        self.gen_layout.addRow('Y-end (m):', self.gen_widgets['y_end'])
        self.gen_layout.addRow('Y Num Points:', self.gen_widgets['y_num'])
        self.gen_layout.addRow('Z-start (m):', self.gen_widgets['z_start'])
        self.gen_layout.addRow('Z-end (m):', self.gen_widgets['z_end'])
        self.gen_layout.addRow('Z Num Points:', self.gen_widgets['z_num'])
        self.gen_layout.addRow('X-const (m):', self.gen_widgets['x_const'])
        self.gen_layout.addRow('Y-const (m):', self.gen_widgets['y_const'])
        self.gen_layout.addRow('Z-const (m):', self.gen_widgets['z_const'])
        self.btn_gen = QPushButton('生成点 (Generate Points)')
        self.btn_gen.clicked.connect(self.generate_points)
        self.gen_layout.addRow(self.btn_gen)
        self.update_gen_ui()
        left_l.addWidget(pg_gen)

        # --- points ---
        pg2 = QGroupBox('观测点列表 (Observation Points)')
        vbox_p = QVBoxLayout()
        self.points_text = QTextEdit()
        self.points_text.setReadOnly(True)
        self.btn_clearp = QPushButton('清空点 (Clear Points)')
        self.btn_clearp.clicked.connect(self.clear_points)
        vbox_p.addWidget(self.points_text)
        vbox_p.addWidget(self.btn_clearp)
        pg2.setLayout(vbox_p)
        left_l.addWidget(pg2)

        # --- Controls ---
        calc_box = QHBoxLayout()
        self.calc_btn = QPushButton('计算 (Calculate)')
        self.calc_btn.clicked.connect(self.start_calculation)
        self.export_btn = QPushButton('导出CSV (Export CSV)')
        self.export_btn.clicked.connect(self.export_csv)
        self.export_btn.setEnabled(False)
        calc_box.addWidget(self.calc_btn)
        calc_box.addWidget(self.export_btn)
        left_l.addLayout(calc_box)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        left_l.addWidget(self.progress)
        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.status.setFixedHeight(80)
        left_l.addWidget(self.status)

        # --- Right Panel: Plotting, Table, and COMSOL ---

        # === 全新的绘图设置面板 ===
        plot_controls_group = QGroupBox("绘图设置")
        plot_controls_layout = QFormLayout()

        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Z-axis Line Plot", "XY-plane Contour", "YZ-plane Contour", "XZ-plane Contour"])

        self.field_component_combo = QComboBox()
        self.field_component_combo.addItems(['|B| (总磁场)', 'Bx', 'By', 'Bz'])

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["viridis", "plasma", "inferno", "magma", "cividis", "jet", "coolwarm"])

        self.plot_button = QPushButton("更新绘图")
        self.plot_button.clicked.connect(self.plot_results)

        self.autoscale_checkbox = QCheckBox("自动颜色范围")
        self.autoscale_checkbox.setChecked(True)

        self.vmin_input = QLineEdit("-1.0")
        self.vmax_input = QLineEdit("1.0")

        # 联动启用/禁用颜色范围输入框
        self.autoscale_checkbox.toggled.connect(lambda checked: self.vmin_input.setEnabled(not checked))
        self.autoscale_checkbox.toggled.connect(lambda checked: self.vmax_input.setEnabled(not checked))
        self.vmin_input.setEnabled(False)
        self.vmax_input.setEnabled(False)

        color_range_layout = QHBoxLayout()
        color_range_layout.addWidget(self.autoscale_checkbox)
        color_range_layout.addWidget(QLabel("Min:"))
        color_range_layout.addWidget(self.vmin_input)
        color_range_layout.addWidget(QLabel("Max:"))
        color_range_layout.addWidget(self.vmax_input)

        plot_controls_layout.addRow("绘图类型:", self.plot_type_combo)
        plot_controls_layout.addRow("显示分量:", self.field_component_combo)
        plot_controls_layout.addRow("颜色映射:", self.colormap_combo)
        plot_controls_layout.addRow("颜色范围:", color_range_layout)
        plot_controls_layout.addRow(self.plot_button)

        plot_controls_group.setLayout(plot_controls_layout)
        right_l.addWidget(plot_controls_group)

        # --- COMSOL data import and comparison feature ---
        comsol_box = QHBoxLayout()
        self.import_comsol_btn = QPushButton('导入COMSOL数据 (用于Z轴对比)')
        self.import_comsol_btn.clicked.connect(self.import_comsol_data)
        self.clear_comsol_btn = QPushButton('清除COMSOL数据')
        self.clear_comsol_btn.clicked.connect(self.clear_comsol_data)
        self.comsol_file_label = QLabel("未导入COMSOL数据")
        comsol_box.addWidget(self.import_comsol_btn)
        comsol_box.addWidget(self.clear_comsol_btn)
        self.comsol_data = []
        self.comsol_filename = ""
        right_l.addLayout(comsol_box)
        right_l.addWidget(self.comsol_file_label)

        # --- Splitter for Table and Canvas ---
        right_splitter = QSplitter(Qt.Vertical)

        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(
            ['x(m)', 'y(m)', 'z(m)', 'Bx(T)', 'By(T)', 'Bz(T)', '|B|(T)', 'B_signed_z(T)'])

        # <--- 修改点 2: 增大画布 figsize
        self.canvas = FigureCanvas(Figure(figsize=(7, 6)))

        right_splitter.addWidget(self.canvas)
        right_splitter.addWidget(self.table)

        # <--- 修改点 3: 调整分割器初始比例
        right_splitter.setSizes([650, 200])
        right_l.addWidget(right_splitter)

    def update_gen_ui(self):
        mode = self.gen_mode_combo.currentText()
        visible_widgets = {
            'Z-axis': ['z_start', 'z_end', 'z_num'],
            'XY-plane': ['x_start', 'x_end', 'x_num', 'y_start', 'y_end', 'y_num', 'z_const'],
            'YZ-plane': ['y_start', 'y_end', 'y_num', 'z_start', 'z_end', 'z_num', 'x_const'],
            'XZ-plane': ['x_start', 'x_end', 'x_num', 'z_start', 'z_end', 'z_num', 'y_const']
        }.get(mode, [])

        for i in range(1, self.gen_layout.rowCount() - 1):
            field_widget = self.gen_layout.itemAt(i, QFormLayout.FieldRole).widget()
            widget_name = None
            for name, widget in self.gen_widgets.items():
                if widget == field_widget:
                    widget_name = name
                    break
            is_visible = widget_name in visible_widgets
            self.gen_layout.labelForField(field_widget).setVisible(is_visible)
            field_widget.setVisible(is_visible)

    def generate_points(self):
        try:
            mode = self.gen_mode_combo.currentText()
            new_points = []

            if mode == 'Z-axis':
                z_start = float(self.gen_widgets['z_start'].text())
                z_end = float(self.gen_widgets['z_end'].text())
                num = int(self.gen_widgets['z_num'].text())
                if num < 2: raise ValueError("Number of points must be at least 2.")
                z_values = np.linspace(z_start, z_end, num)
                new_points = [(0.0, 0.0, z) for z in z_values]

            elif mode == 'XY-plane':
                x_start = float(self.gen_widgets['x_start'].text())
                x_end = float(self.gen_widgets['x_end'].text())
                x_num = int(self.gen_widgets['x_num'].text())
                y_start = float(self.gen_widgets['y_start'].text())
                y_end = float(self.gen_widgets['y_end'].text())
                y_num = int(self.gen_widgets['y_num'].text())
                z_const = float(self.gen_widgets['z_const'].text())
                x_values = np.linspace(x_start, x_end, x_num)
                y_values = np.linspace(y_start, y_end, y_num)
                for x in x_values:
                    for y in y_values:
                        new_points.append((x, y, z_const))

            elif mode == 'YZ-plane':
                y_start = float(self.gen_widgets['y_start'].text())
                y_end = float(self.gen_widgets['y_end'].text())
                y_num = int(self.gen_widgets['y_num'].text())
                z_start = float(self.gen_widgets['z_start'].text())
                z_end = float(self.gen_widgets['z_end'].text())
                z_num = int(self.gen_widgets['z_num'].text())
                x_const = float(self.gen_widgets['x_const'].text())
                y_values = np.linspace(y_start, y_end, y_num)
                z_values = np.linspace(z_start, z_end, z_num)
                for y in y_values:
                    for z in z_values:
                        new_points.append((x_const, y, z))

            elif mode == 'XZ-plane':
                x_start = float(self.gen_widgets['x_start'].text())
                x_end = float(self.gen_widgets['x_end'].text())
                x_num = int(self.gen_widgets['x_num'].text())
                z_start = float(self.gen_widgets['z_start'].text())
                z_end = float(self.gen_widgets['z_end'].text())
                z_num = int(self.gen_widgets['z_num'].text())
                y_const = float(self.gen_widgets['y_const'].text())
                x_values = np.linspace(x_start, x_end, x_num)
                z_values = np.linspace(z_start, z_end, z_num)
                for x in x_values:
                    for z in z_values:
                        new_points.append((x, y_const, z))

            self.points.extend(new_points)
            self._refresh_points_display()
            self.status.append(f"Generated and added {len(new_points)} points via {mode} mode.")

        except Exception as e:
            QMessageBox.warning(self, '生成错误', '无法生成点: ' + str(e))

    def _refresh_points_display(self):
        s = f"Total points: {len(self.points)}\n"
        display_count = min(len(self.points), 100)
        for i in range(display_count):
            x, y, z = self.points[i]
            s += f"{i + 1}: x={x:.4g}, y={y:.4g}, z={z:.4g}\n"
        if len(self.points) > 100:
            s += f"\n... and {len(self.points) - 100} more points."
        self.points_text.setText(s)

    def clear_points(self):
        self.points = []
        self._refresh_points_display()

    def get_params(self):
        try:
            params = {
                'fast_mode': self.fast_mode_checkbox.isChecked(),
                'current': float(self.I_input.text()),
                'N_radial': int(self.N_radial_input.text()),
                'N_turns_axial': int(self.N_turns_input.text()),
                'r_inner': float(self.r_inner_input.text()),
                'radial_width': float(self.radial_width_input.text()),
                'axial_thickness': float(self.axial_thickness_input.text()),
                'wire_diameter': float(self.wire_diameter_input.text()),
                'straight_len': float(self.straight_len_input.text()),
                'n_line': int(self.n_line_input.text()),
                'n_arc': int(self.n_arc_input.text()),
                'use_J': bool(self.use_J_checkbox.isChecked()),
                'J': float(self.J_input.text()),
                'wire_width': float(self.wire_w_input.text()),
                'wire_thickness': float(self.wire_t_input.text()),
                'n_width': int(self.n_w_input.text()),
                'n_thick': int(self.n_t_input.text()),
                'use_two_coils': bool(self.use_two_coils_checkbox.isChecked()),
                'coil_distance': float(self.coil_distance_input.text()),
                'coil1_dir': self.coil1_current_dir_input.currentText(),
                'coil2_dir': self.coil2_current_dir_input.currentText(),
            }
            if params['N_radial'] < 1 or params['N_turns_axial'] < 1:
                raise ValueError("N_radial and N_turns_axial must be 1 or greater.")
            return params
        except Exception as e:
            QMessageBox.warning(self, '参数错误', '请检查所有输入参数是否为有效数值: ' + str(e))
            raise

    def start_calculation(self):
        if not self.points:
            QMessageBox.warning(self, '没有点', '请先添加观测点')
            return
        try:
            params = self.get_params()
        except:
            return

        self.calc_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status.setText('开始计算...\n')
        self.table.setRowCount(0)

        self.calc_thread = FieldCalculationThread(params, self.points)
        self.calc_thread.calculation_progress.connect(self._on_progress)
        self.calc_thread.calculation_done.connect(self._on_done)
        self.calc_thread.error_occurred.connect(self._on_error)
        self.calc_thread.start()

    def _on_progress(self, msg, cur, tot):
        p = int(cur * 100 / tot)
        self.progress.setValue(p)

    def _on_done(self, results):
        self.calc_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.results = results
        self.table.setRowCount(len(results))
        for i, r in enumerate(results):
            self.table.setItem(i, 0, QTableWidgetItem(f"{r['x']:.6e}"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{r['y']:.6e}"))
            self.table.setItem(i, 2, QTableWidgetItem(f"{r['z']:.6e}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{r['Bx']:.6e}"))
            self.table.setItem(i, 4, QTableWidgetItem(f"{r['By']:.6e}"))
            self.table.setItem(i, 5, QTableWidgetItem(f"{r['Bz']:.6e}"))
            self.table.setItem(i, 6, QTableWidgetItem(f"{r['B_mag']:.6e}"))
            self.table.setItem(i, 7, QTableWidgetItem(f"{r['B_signed_z']:.6e}"))

        total_time = sum(r['time'] for r in results)
        self.status.append(f"计算完成，总耗时 {total_time:.2f} s")
        self.status.append("可以点击 '更新绘图' 按钮来可视化结果。")

    def import_comsol_data(self):
        path, _ = QFileDialog.getOpenFileName(self, '导入COMSOL数据', '', 'CSV文件 (*.csv)')
        if not path:
            return

        try:
            self.comsol_filename = path.split('/')[-1]
            self.comsol_file_label.setText(f"已导入: {self.comsol_filename}")
            with open(path, 'r', encoding='utf-8-sig') as f:
                lines = [line for line in f if not line.strip().startswith('%')]
                if not lines:
                    raise ValueError("CSV file is empty or contains only comments.")

                reader = csv.reader(lines)
                header = next(reader)
                num_cols = len(header)

                self.comsol_data = []
                if num_cols == 2:
                    for row in reader:
                        if len(row) == 2:
                            self.comsol_data.append({'z': float(row[0]), 'bz': float(row[1])})
                elif num_cols == 4:
                    for row in reader:
                        if len(row) == 4:
                            self.comsol_data.append(
                                {'z': float(row[0]), 'bx': float(row[1]), 'by': float(row[2]), 'bz': float(row[3])})
                else:
                    raise ValueError(f"不支持的列数 ({num_cols}). 请提供含 2 (z, Bz) 或 4 (z, Bx, By, Bz) 列的 CSV.")

            self.status.append(f"已导入 {len(self.comsol_data)} 个COMSOL数据点")
            if self.results:
                self.plot_results()  # 自动更新绘图

        except Exception as e:
            QMessageBox.warning(self, '导入错误', f'导入COMSOL数据失败: {str(e)}')
            self.clear_comsol_data()

    def clear_comsol_data(self):
        self.comsol_data = []
        self.comsol_filename = ""
        self.comsol_file_label.setText("未导入COMSOL数据")
        self.status.append("已清除COMSOL数据")
        if self.results:
            self.plot_results()

    def plot_contour_map(self, ax, x_vals, y_vals, data_vals, title, xlabel, ylabel, cmap, vmin, vmax):
        """
        绘制二维磁场分布云图的核心函数 (从圆线圈程序移植而来)
        """
        # 设置插值网格
        try:
            minX, maxX = min(x_vals), max(x_vals)
            minY, maxY = min(y_vals), max(y_vals)

            # 创建插值网格
            xi = np.linspace(minX, maxX, 200)
            yi = np.linspace(minY, maxY, 200)
            Xi, Yi = np.meshgrid(xi, yi)

            # 使用griddata进行插值
            Zi = griddata((x_vals, y_vals), data_vals, (Xi, Yi), method='cubic')

            # 推荐使用pcolormesh
            img = ax.pcolormesh(Xi, Yi, Zi, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)

            # 添加颜色条
            cbar = self.canvas.figure.colorbar(img, ax=ax)
            cbar.set_label('磁场强度 (T)', size=12)

            # 设置坐标轴标签
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)

            # 添加等高线标记
            contour = ax.contour(Xi, Yi, Zi, 10, colors='black', linewidths=0.5)
            ax.clabel(contour, inline=True, fontsize=8)

            ax.set_xlim(minX, maxX)
            ax.set_ylim(minY, maxY)
            ax.set_aspect('equal', adjustable='box')

        except Exception as e:
            ax.text(0.5, 0.5, f'无法绘制云图:\n{str(e)}', horizontalalignment='center', verticalalignment='center')

    def plot_results(self):
        if not self.results:
            QMessageBox.warning(self, "无数据", "没有可供绘制的计算结果。")
            return

        # --- 获取UI设置 ---
        plot_type = self.plot_type_combo.currentText()
        component_key = {'|B| (总磁场)': 'B_mag', 'Bx': 'Bx', 'By': 'By', 'Bz': 'Bz'}[
            self.field_component_combo.currentText()]
        component_title = self.field_component_combo.currentText()
        colormap = self.colormap_combo.currentText()

        vmin, vmax = None, None
        if not self.autoscale_checkbox.isChecked():
            try:
                vmin = float(self.vmin_input.text())
                vmax = float(self.vmax_input.text())
            except ValueError:
                QMessageBox.warning(self, "输入错误", "颜色范围必须是有效的数字。")
                return

        # --- 清空并准备画布 ---
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)

        # --- 根据绘图类型执行操作 ---
        if plot_type == "Z-axis Line Plot":
            # --- MODIFICATION START ---
            z_points, b_mag_points, b_signed_z_points = [], [], []
            is_z_axis_data = True
            for r in self.results:
                if abs(r['x']) > 1e-9 or abs(r['y']) > 1e-9:
                    is_z_axis_data = False
                    break
                z_points.append(r['z'])
                b_mag_points.append(r['B_mag'])
                b_signed_z_points.append(r['B_signed_z'])

            if not is_z_axis_data:
                ax.text(0.5, 0.5, '当前数据点不完全在中轴线上\n无法绘制Z轴一维曲线', horizontalalignment='center')
            else:
                sorted_data = sorted(zip(z_points, b_mag_points, b_signed_z_points))
                z_s, b_mag_s, b_signed_z_s = zip(*sorted_data)

                # Dashed line for absolute magnitude
                ax.plot(z_s, b_mag_s, 'k--', label='|B| (计算)')
                # Solid line for signed B (relative to Z-axis)
                ax.plot(z_s, b_signed_z_s, 'r-', label='B_signed_z (计算)')
                # Add a zero line
                ax.axhline(0, color='gray', linestyle='-.', linewidth=0.8)

                # Plot COMSOL comparison data
                if self.comsol_data:
                    comsol_z = [d['z'] for d in self.comsol_data]
                    if 'bz' in self.comsol_data[0]:
                        comsol_bz = [d['bz'] for d in self.comsol_data]
                        ax.plot(comsol_z, comsol_bz, 'bo', markersize=3, label='Bz (COMSOL)')

                ax.legend()
                ax.set_xlabel("Z-axis position (m)")
                ax.set_ylabel("Magnetic Field (T)")
                ax.set_title("Z轴磁场")
                ax.grid(True)
            # --- MODIFICATION END ---

        else:  # --- 处理所有二维云图 ---
            all_x = np.array([r['x'] for r in self.results])
            all_y = np.array([r['y'] for r in self.results])
            all_z = np.array([r['z'] for r in self.results])
            data_vals = np.array([r[component_key] for r in self.results])

            if plot_type == "XY-plane Contour":
                # 验证Z坐标是否恒定
                if np.std(all_z) > 1e-9:
                    ax.text(0.5, 0.5, f'无法绘制XY平面云图\nZ坐标不唯一 (标准差: {np.std(all_z):.2e})', ha='center')
                else:
                    self.plot_contour_map(ax, all_x, all_y, data_vals, f"{component_title} at Z={all_z[0]:.3f}m",
                                          "X-axis (m)", "Y-axis (m)", colormap, vmin, vmax)

            elif plot_type == "YZ-plane Contour":
                # 验证X坐标是否恒定
                if np.std(all_x) > 1e-9:
                    ax.text(0.5, 0.5, f'无法绘制YZ平面云图\nX坐标不唯一 (标准差: {np.std(all_x):.2e})', ha='center')
                else:
                    self.plot_contour_map(ax, all_y, all_z, data_vals, f"{component_title} at X={all_x[0]:.3f}m",
                                          "Y-axis (m)", "Z-axis (m)", colormap, vmin, vmax)

            elif plot_type == "XZ-plane Contour":
                # 验证Y坐标是否恒定
                if np.std(all_y) > 1e-9:
                    ax.text(0.5, 0.5, f'无法绘制XZ平面云图\nY坐标不唯一 (标准差: {np.std(all_y):.2e})', ha='center')
                else:
                    self.plot_contour_map(ax, all_x, all_z, data_vals, f"{component_title} at Y={all_y[0]:.3f}m",
                                          "X-axis (m)", "Z-axis (m)", colormap, vmin, vmax)

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def _on_error(self, msg):
        self.calc_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.status.append('计算出错: ' + msg)
        QMessageBox.critical(self, '计算错误', msg)

    def export_csv(self):
        if not hasattr(self, 'results') or not self.results:
            QMessageBox.warning(self, '无结果', '没有可导出的计算结果')
            return

        path, _ = QFileDialog.getSaveFileName(self, '保存 CSV', '', 'CSV 文件 (*.csv)')
        if not path:
            return

        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['x(m)', 'y(m)', 'z(m)', 'Bx(T)', 'By(T)', 'Bz(T)', '|B|_absolute(T)', 'B_signed_z(T)'])
                for r in self.results:
                    w.writerow([r['x'], r['y'], r['z'], r['Bx'], r['By'], r['Bz'], r['B_mag'], r['B_signed_z']])
            QMessageBox.information(self, '导出成功', f'已导出到 {path}')
        except Exception as e:
            QMessageBox.warning(self, '导出失败', str(e))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = RacetrackFieldApp()
    win.show()
    sys.exit(app.exec_())
