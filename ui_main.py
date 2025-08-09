import sys
import os
import re
import shutil
import glob
import subprocess
import threading
from pathlib import Path
import psutil

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QSpinBox, QFileDialog,
    QGroupBox, QComboBox, QMessageBox, QProgressBar
)
from PyQt5.QtCore import QObject, pyqtSignal


class WorkerSignals(QObject):
    progress_value = pyqtSignal(int)
    progress_text = pyqtSignal(str)


class SHARPPipelineApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SHARP Signal Processing Pipeline UI")
        self.resize(800, 400)  # 修改窗口宽高，这里是宽1200px，高800px

        # 当前工作目录
        cwd = Path.cwd()

        # 默认路径修改
        self.dataset_folder = cwd / "datasets"
        self.python_code_folder = cwd / "code" / "src"

        # Important: 
        self.phase_output_root = cwd / "results" / "processed_phase"
        self.doppler_output_root = cwd / "results" / "doppler_traces"

        self.subdirs = "S1a"

        self.initUI()
        self.signals = WorkerSignals()
        self.signals.progress_value.connect(self.progress_bar.setValue)
        self.signals.progress_text.connect(self.status_label.setText)

    # ---------------- UI ----------------
    def initUI(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        box = QGroupBox("SHARP Pipeline")
        layout.addWidget(box)
        s = QVBoxLayout(box)

        self.dataset_folder_input = self._folder_row(
            s, "CSI Dataset Folder:", self.dataset_folder)
        self.python_code_folder_input = self._folder_row(
            s, "SHARP Python Code Folder:", self.python_code_folder)

        self.phase_output_input = QLineEdit(str(self.phase_output_root))
        s.addWidget(QLabel("Processed Phase Output Folder:"))
        s.addWidget(self.phase_output_input)

        self.doppler_output_input = QLineEdit(str(self.doppler_output_root))
        s.addWidget(QLabel("Doppler Output Folder:"))
        s.addWidget(self.doppler_output_input)

        self.subdirs_input = QLineEdit(self.subdirs)
        s.addWidget(QLabel("Datasets Subdirectories:"))
        s.addWidget(self.subdirs_input)

        row = QHBoxLayout()
        self.get_files_button = QPushButton("Get Files")
        self.get_files_button.clicked.connect(self.get_files_in_subdir)
        self.file_combo = QComboBox()
        self.visualize_button = QPushButton("Visualize / Show Info")
        self.visualize_button.clicked.connect(self.visualize_selected_file)
        row.addWidget(self.get_files_button)
        row.addWidget(self.file_combo)
        row.addWidget(self.visualize_button)
        s.addLayout(row)

        self.streams_input = QSpinBox()
        self.streams_input.setValue(1)
        s.addWidget(QLabel("Number of Spatial Streams:"))
        s.addWidget(self.streams_input)

        self.cores_input = QSpinBox()
        # 获取物理核心数，获取不到就用1
        physical_cores = psutil.cpu_count(logical=False) or 1
        self.cores_input.setValue(physical_cores)
        s.addWidget(QLabel("Number of CPU Cores:"))
        s.addWidget(self.cores_input)

        self.status_label = QLabel("Status: Ready.")
        s.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat('%p%')
        self.progress_bar.setTextVisible(True)
        s.addWidget(self.progress_bar)

        self._pipeline_button(s, "1. Run Phase Sanitization",
                              self.run_phase_sanitization)
        self._pipeline_button(s, "2. Run Doppler Computation",
                              self.run_doppler_computation)
        self._pipeline_button(
            s, "3. Create Dataset (Train)", self.run_create_datasets)
        self._pipeline_button(s, "4. Train HAR Model", self.run_train_model)

    def _folder_row(self, parent_layout, label, default_path):
        row = QHBoxLayout()
        inp = QLineEdit(str(default_path))
        browse = QPushButton("Browse")
        parent_layout.addWidget(QLabel(label))
        row.addWidget(inp)
        row.addWidget(browse)
        parent_layout.addLayout(row)
        browse.clicked.connect(lambda: self._browse_folder(inp))
        return inp

    def _pipeline_button(self, parent_layout, label, cb):
        b = QPushButton(label)
        b.clicked.connect(cb)
        parent_layout.addWidget(b)

    def _browse_folder(self, field):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder", field.text())
        if folder:
            field.setText(folder)

    # ------------- Helpers -------------
    def _run(self, cmd_list, msg, cwd=None):
        self.status_label.setText(msg + "...")
        subprocess.run(cmd_list, cwd=cwd, check=False)
        self.status_label.setText(msg + " complete.")

    def get_files_in_subdir(self):
        folder = Path(self.dataset_folder_input.text()) / \
            self.subdirs_input.text().split(",")[0].strip()
        self.file_combo.clear()
        if not folder.exists():
            QMessageBox.warning(self, "Missing Folder",
                                f"Folder not found:\n{folder}")
            return
        files = sorted([f.name for f in folder.glob("*.mat")])
        if not files:
            QMessageBox.information(
                self, "No Files", f"No .mat files found in {folder}.")
        self.file_combo.addItems(files)
        self.status_label.setText(f"Found {len(files)} files in {folder.name}")

    def visualize_selected_file(self):
        folder = Path(self.dataset_folder_input.text()) / \
            self.subdirs_input.text().split(",")[0].strip()
        fname = self.file_combo.currentText()
        fpath = folder / fname
        if not fpath.exists():
            QMessageBox.warning(self, "Missing File", f"{fpath} not found")
            return
        try:
            mat = loadmat(fpath)
            msg = f"Variables in {fname}:\n"
            for k in mat:
                if not k.startswith("__"):
                    arr = mat[k]
                    msg += f"- {k}: {type(arr)}, shape: {getattr(arr, 'shape', None)}\n"
            QMessageBox.information(self, "File Info", msg)

            for k, arr in mat.items():
                if not k.startswith("__") and isinstance(arr, np.ndarray):
                    if np.iscomplexobj(arr):
                        arr = np.abs(arr)
                    if arr.ndim == 2:
                        plt.imshow(arr, aspect='auto')
                        plt.title(f"{fname} - {k}")
                        plt.colorbar()
                        plt.show()
                        break
                    elif arr.ndim == 3:
                        plt.imshow(arr[0], aspect='auto')
                        plt.title(f"{fname} - {k} [0]")
                        plt.colorbar()
                        plt.show()
                        break
            else:
                QMessageBox.information(
                    self, "No 2D/3D Arrays", "Nothing to visualize.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # ------------- Pipeline -------------
    def run_phase_sanitization(self):
        """
        Step 1: Run preprocessing + H-estimation ONCE (with correct cwd so relative paths work).
        Step 2: For each activity, filter matching traces into a temp folder, run reconstruction for that subset.
        Output: processed_phase/<subdir>/AR1a_<ACT>_stream_*.mat   (names are finally correct)
        """
        base = Path(self.python_code_folder_input.text())
        data_root = Path(self.dataset_folder_input.text())
        subdir = self.subdirs_input.text().strip()
        streams = self.streams_input.value()
        cores = self.cores_input.value()

        ps_dir = base / "01_phase_sanitization"
        phase_processing = ps_dir / "phase_processing"
        phase_processing.mkdir(parents=True, exist_ok=True)

        # ---- Step 1: preprocessing + H-estimation (once) ----
        # (Run them in their script folder so './phase_processing' resolves correctly.)
        preprocess_cmd = [
            sys.executable,
            str(ps_dir / "CSI_phase_sanitization_signal_preprocessing.py"),
            str(data_root), "1", "-", str(streams), str(cores), "0"
        ]
        hest_cmd = [
            sys.executable,
            str(ps_dir / "CSI_phase_sanitization_H_estimation.py"),
            str(data_root), "1", "-", str(streams), str(cores), "0", "-1"
        ]

        self._run(preprocess_cmd, "Preprocessing (all activities)",
                  cwd=str(ps_dir))
        self._run(hest_cmd, "H-estimation (all activities)", cwd=str(ps_dir))

        # Where final .mat should go (one folder per dataset subdir)
        out_dir = Path(self.phase_output_input.text()) / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- Step 2: per-activity reconstruction on filtered traces ----
        # extend if needed (C, S, J1, J2, etc.)
        activities = ["E", "H", "L", "R", "W"]
        short_subdir = subdir.replace("-", "")  # AR-1a -> AR1a

        for act in activities:
            pat = f"Tr_vector_{short_subdir}_{act}_stream_*.txt"
            matches = sorted((phase_processing).glob(pat))
            if not matches:
                self.status_label.setText(
                    f"No traces found for {subdir} {act} in {phase_processing}. Skipping.")
                continue

            tmp_in = phase_processing / f"_tmp_{subdir}_{act}"
            if tmp_in.exists():
                shutil.rmtree(tmp_in)
            tmp_in.mkdir(parents=True, exist_ok=True)

            # copy only the target activity traces
            for m in matches:
                shutil.copy2(m, tmp_in / m.name)

            recon_cmd = [
                sys.executable,
                str(ps_dir / "CSI_phase_sanitization_signal_reconstruction.py"),
                # <--- filtered input (only this activity)
                str(tmp_in),
                # <--- all .mat live in processed_phase/<subdir>
                str(out_dir),
                str(streams), str(cores), "0", "-1"
            ]
            self._run(recon_cmd, f"Reconstruction ({act})", cwd=str(ps_dir))

            # clean temp folder
            shutil.rmtree(tmp_in, ignore_errors=True)

        self.status_label.setText(
            f"Phase sanitization complete. Output: {out_dir}")

    def run_doppler_computation(self):
        base = Path(self.python_code_folder_input.text())
        out_phase_root = Path(self.phase_output_input.text())
        subdirs = self.subdirs_input.text().strip()    # e.g., "AR-1a"
        doppler_root = Path(self.doppler_output_input.text()).rstrip(
            "/") if hasattr(Path, 'rstrip') else Path(self.doppler_output_input.text())

        # The doppler script expects processed_phase root + subdir name
        cmd = [
            sys.executable,
            str(base / "02_doppler" / "CSI_doppler_computation.py"),
            str(out_phase_root), subdirs, str(
                self.doppler_output_input.text()), "800", "800", "31", "1", "-1.2"
        ]

        def worker():
            self.signals.progress_text.emit("Doppler Computation started...")
            self.signals.progress_value.emit(0)
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, universal_newlines=True)
            for line in iter(p.stdout.readline, ''):
                line = line.strip()
                self.signals.progress_text.emit(line)
                m = re.search(r'\[(\d+)/(\d+)\]', line)
                if m:
                    cur, tot = int(m.group(1)), int(m.group(2))
                    pct = int((cur / tot) * 100)
                    self.signals.progress_value.emit(pct)
            p.stdout.close()
            p.wait()
            self.signals.progress_value.emit(100)
            self.signals.progress_text.emit("Doppler Computation complete.")

        threading.Thread(target=worker).start()

    def run_create_datasets(self):
        base = Path(self.python_code_folder_input.text())
        doppler_root = Path(self.doppler_output_input.text())
        subdirs = self.subdirs_input.text().strip()
        if not doppler_root.exists():
            QMessageBox.critical(
                self, "Error", f"Doppler path does not exist:\n{doppler_root}")
            return
        cmd = [
            sys.executable,
            str(base / "03_dataset" / "CSI_doppler_create_dataset_train.py"),
            str(doppler_root), subdirs, "31", "1", "340", "30", "E,L,W,R,J", "4"
        ]
        self._run(cmd, "Creating Dataset")

    def run_train_model(self):
        base = Path(self.python_code_folder_input.text())
        doppler_root = Path(self.doppler_output_input.text())
        subdirs = self.subdirs_input.text().strip()
        if not doppler_root.exists():
            QMessageBox.critical(
                self, "Error", f"Doppler path does not exist:\n{doppler_root}")
            return
        cmd = [
            sys.executable,
            str(base / "04_network" / "CSI_network.py"),
            str(doppler_root), subdirs, "100", "340", "1", "32", "4",
            "single_ant", "E,L,W,R,J", "--bandwidth", "80", "--sub_band", "1"
        ]
        self._run(cmd, "Training Model")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SHARPPipelineApp()
    w.show()
    sys.exit(app.exec_())
