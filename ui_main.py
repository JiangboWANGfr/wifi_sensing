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
    QGroupBox, QComboBox, QMessageBox, QProgressBar, QTextEdit, QSplitter
)
from PyQt5.QtCore import QObject, pyqtSignal


class WorkerSignals(QObject):
    progress_value = pyqtSignal(int)
    progress_text = pyqtSignal(str)


class SHARPPipelineApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SHARP Signal Processing Pipeline UI")
        self.resize(900, 520)

        # ---- Stable project root based on this file's location ----
        self.project_root = Path(__file__).resolve().parent

        # Default folders (no matter where you launch Python)
        self.dataset_folder = self.project_root / "datasets"
        self.python_code_folder = self.project_root / "code" / "src"

        self.results_output_root = self.project_root / "results"
        self.phase_output_root = self.project_root / "results" / "processed_phase"
        self.doppler_output_root = self.project_root / "results" / "doppler_traces"

        # UI state
        self.subdir_current = None  # will be set after scanning

        self.initUI()
        self.signals = WorkerSignals()
        self.signals.progress_value.connect(self.progress_bar.setValue)
        # show brief status AND full log
        self.signals.progress_text.connect(self.status_label.setText)
        self.signals.progress_text.connect(self.append_log)

        # Initial population of subdir dropdown + files
        self.refresh_subdirs()

        # Initial population of subdir dropdown + files
        self.refresh_subdirs()

    # ---------------- UI ----------------
    def initUI(self):
        central = QWidget()
        self.setCentralWidget(central)

        # --- Splitter: left controls | right live output ---
        outer = QHBoxLayout(central)
        splitter = QSplitter()
        outer.addWidget(splitter)

        # Left panel: controls
        left = QWidget()
        left_layout = QVBoxLayout(left)
        splitter.addWidget(left)

        box = QGroupBox("SHARP Pipeline")
        left_layout.addWidget(box)
        s = QVBoxLayout(box)

        # Folder rows
        self.dataset_folder_input = self._folder_row(
            s, "CSI Dataset Folder:", self.dataset_folder, on_changed=self.on_dataset_folder_changed)
        self.python_code_folder_input = self._folder_row(
            s, "SHARP Python Code Folder:", self.python_code_folder)

        # Output folders (read-only)
        self.phase_output_input = QLineEdit(str(self.phase_output_root))
        self.phase_output_input.setReadOnly(True)
        s.addWidget(QLabel("Processed Phase Output Folder:"))
        s.addWidget(self.phase_output_input)

        self.doppler_output_input = QLineEdit(str(self.doppler_output_root))
        self.doppler_output_input.setReadOnly(True)
        s.addWidget(QLabel("Doppler Output Folder:"))
        s.addWidget(self.doppler_output_input)

        # Subdir dropdown
        subrow = QHBoxLayout()
        subrow.addWidget(QLabel("Dataset Subdirectory:"))
        self.subdirs_combo = QComboBox()
        self.subdirs_combo.currentTextChanged.connect(self.on_subdir_changed)
        subrow.addWidget(self.subdirs_combo)
        s.addLayout(subrow)

        # Files row
        filerow = QHBoxLayout()
        self.get_files_button = QPushButton("Refresh Files")
        self.get_files_button.clicked.connect(self.get_files_in_subdir)
        self.file_combo = QComboBox()
        self.visualize_button = QPushButton("Visualize / Show Info")
        self.visualize_button.clicked.connect(self.visualize_selected_file)
        filerow.addWidget(self.get_files_button)
        filerow.addWidget(self.file_combo)
        filerow.addWidget(self.visualize_button)
        s.addLayout(filerow)

        # Params
        self.streams_input = QSpinBox()
        self.streams_input.setMinimum(1)
        self.streams_input.setValue(1)
        s.addWidget(QLabel("Number of Spatial Streams:"))
        s.addWidget(self.streams_input)

        self.cores_input = QSpinBox()
        physical_cores = psutil.cpu_count(logical=False) or 1
        self.cores_input.setMinimum(1)
        self.cores_input.setValue(physical_cores)
        s.addWidget(QLabel("Number of CPU Cores:"))
        s.addWidget(self.cores_input)

        # Status + Progress
        self.status_label = QLabel("Status: Ready.")
        s.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat('%p%')
        self.progress_bar.setTextVisible(True)
        s.addWidget(self.progress_bar)

        # Pipeline buttons
        self._pipeline_button(s, "1. Run Phase Sanitization",
                              self.run_phase_sanitization)
        self._pipeline_button(s, "2. Run Doppler Computation",
                              self.run_doppler_computation)
        self._pipeline_button(
            s, "3. Create Dataset (Train)", self.run_create_datasets)
        self._pipeline_button(s, "4. Train HAR Model", self.run_train_model)

        # Right panel: live output log
        right = QWidget()
        rlayout = QVBoxLayout(right)
        rlayout.addWidget(QLabel("Run Output"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.NoWrap)
        rlayout.addWidget(self.log_text)
        splitter.addWidget(right)

        splitter.setSizes([600, 300])

    def _folder_row(self, parent_layout, label, default_path, on_changed=None):
        row = QHBoxLayout()
        inp = QLineEdit(str(default_path))
        browse = QPushButton("Browse")
        parent_layout.addWidget(QLabel(label))
        row.addWidget(inp)
        row.addWidget(browse)
        parent_layout.addLayout(row)

        def _browse():
            folder = QFileDialog.getExistingDirectory(
                self, "Select Folder", inp.text())
            if folder:
                inp.setText(folder)
                if on_changed:
                    on_changed()
        browse.clicked.connect(_browse)
        if on_changed:
            inp.editingFinished.connect(on_changed)
        return inp

    def _pipeline_button(self, parent_layout, label, cb):
        b = QPushButton(label)
        b.clicked.connect(cb)
        parent_layout.addWidget(b)

    def append_log(self, text: str):
        if not hasattr(self, 'log_text') or self.log_text is None:
            return
        self.log_text.append(text)
        self.log_text.moveCursor(self.log_text.textCursor().End)

    # ------------- Data discovery -------------
    def refresh_subdirs(self):
        """Scan dataset folder and populate subdir dropdown."""
        root = Path(self.dataset_folder_input.text())
        if not root.exists():
            self.subdirs_combo.clear()
            self.status_label.setText(f"Dataset folder not found: {root}")
            return
        subdirs = [p.name for p in sorted(
            root.iterdir()) if p.is_dir() and not p.name.startswith('.')]
        self.subdirs_combo.blockSignals(True)
        self.subdirs_combo.clear()
        self.subdirs_combo.addItems(subdirs)
        self.subdirs_combo.blockSignals(False)
        if subdirs:
            self.subdir_current = subdirs[0]
            self.subdirs_combo.setCurrentText(self.subdir_current)
            self.get_files_in_subdir()
        else:
            self.subdir_current = None
            self.file_combo.clear()
            self.status_label.setText("No subdirectories inside datasets.")

    def on_dataset_folder_changed(self):
        self.refresh_subdirs()

    def on_subdir_changed(self, text):
        self.subdir_current = text
        self.get_files_in_subdir()

    # ------------- Helpers -------------
    def _run(self, cmd_list, msg, cwd=None):
        self.status_label.setText(msg + "...")
        try:
            subprocess.run(cmd_list, cwd=cwd, check=True)
            self.status_label.setText(msg + " complete.")
        except subprocess.CalledProcessError as e:
            self.status_label.setText(msg + f" FAILED ({e.returncode})")
            QMessageBox.critical(
                self, "Error", f"Command failed:\n{' '.join(map(str, cmd_list))}\n\n{e}")

    def get_files_in_subdir(self):
        root = Path(self.dataset_folder_input.text())
        if not self.subdir_current:
            self.file_combo.clear()
            return
        folder = root / self.subdir_current
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
        root = Path(self.dataset_folder_input.text())
        if not self.subdir_current:
            QMessageBox.information(
                self, "No Subdir", "Please choose a dataset subdirectory.")
            return
        folder = root / self.subdir_current
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
        Updated to run scripts from code/src (no nested folders). We set cwd to code/src
        so any relative paths like './phase_processing' resolve inside that folder.
        Outputs are copied/moved to results/processed_phase/<subdir> by the reconstruction step.
        """
        base = Path(self.python_code_folder_input.text())  # code/src
        data_root = Path(self.dataset_folder_input.text())
        subdir = self.subdir_current
        if not subdir:
            QMessageBox.information(
                self, "No Subdir", "Please choose a dataset subdirectory.")
            return
        streams = self.streams_input.value()
        cores = self.cores_input.value()

        phase_processing = self.results_output_root / "phase_processing"
        phase_processing.mkdir(parents=True, exist_ok=True)

        data_path = data_root / subdir 

        # ---- Step 1: preprocessing + H-estimation (once for all activities) ----
        preprocess_cmd = [
            sys.executable,
            str(base / "CSI_phase_sanitization_signal_preprocessing.py"),
            str(phase_processing), str(data_path)+ "/", "1", "-", str(streams), str(cores), "0"
        ]
        hest_cmd = [
            sys.executable,
            str(base / "CSI_phase_sanitization_H_estimation.py"),
            str(phase_processing), str(data_path), "1", "-", str(streams), str(cores), "0", "-1"
        ]

        self._run(preprocess_cmd, "Preprocessing (all activities)", cwd=str(base))
        self._run(hest_cmd, "H-estimation (all activities)", cwd=str(base))

        # Where final .mat should go (one folder per dataset subdir)
        out_dir = Path(self.phase_output_input.text()) / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- Step 2: per-activity reconstruction on filtered traces ----
        activities = ["E", "H", "L", "R", "W"]
        short_subdir = subdir.replace("-", "")

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

            for m in matches:
                shutil.copy2(m, tmp_in / m.name)

            recon_cmd = [
                sys.executable,
                str(base / "CSI_phase_sanitization_signal_reconstruction.py"),
                str(tmp_in),
                str(out_dir),
                str(streams), str(cores), "0", "-1"
            ]
            self._run(recon_cmd, f"Reconstruction ({act})", cwd=str(base))

            shutil.rmtree(tmp_in, ignore_errors=True)

        self.status_label.setText(
            f"Phase sanitization complete. Output: {out_dir}")

    def run_doppler_computation(self):
        base = Path(self.python_code_folder_input.text())  # code/src
        out_phase_root = Path(self.phase_output_input.text())
        subdir = self.subdir_current or ""
        doppler_root = Path(self.doppler_output_input.text())

        cmd = [
            sys.executable,
            str(base / "CSI_doppler_computation.py"),
            str(out_phase_root), subdir, str(doppler_root),
            "800", "800", "31", "1", "-1.2"
        ]

        def worker():
            self.signals.progress_text.emit("Doppler Computation started...")
            self.signals.progress_value.emit(0)
            p = subprocess.Popen(cmd, cwd=str(base), stdout=subprocess.PIPE,
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

        threading.Thread(target=worker, daemon=True).start()

    def run_create_datasets(self):
        base = Path(self.python_code_folder_input.text())
        doppler_root = Path(self.doppler_output_input.text())
        subdir = self.subdir_current or ""
        if not doppler_root.exists():
            QMessageBox.critical(
                self, "Error", f"Doppler path does not exist:\n{doppler_root}")
            return
        cmd = [
            sys.executable,
            str(base / "CSI_doppler_create_dataset_train.py"),
            str(doppler_root), subdir, "31", "1", "340", "30", "E,L,W,R,J", "4"
        ]
        self._run(cmd, "Creating Dataset", cwd=str(base))

    def run_train_model(self):
        base = Path(self.python_code_folder_input.text())
        doppler_root = Path(self.doppler_output_input.text())
        subdir = self.subdir_current or ""
        if not doppler_root.exists():
            QMessageBox.critical(
                self, "Error", f"Doppler path does not exist:\n{doppler_root}")
            return
        cmd = [
            sys.executable,
            str(base / "CSI_network.py"),
            str(doppler_root), subdir, "100", "340", "1", "32", "4",
            "single_ant", "E,L,W,R,J", "--bandwidth", "80", "--sub_band", "1"
        ]
        self._run(cmd, "Training Model", cwd=str(base))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SHARPPipelineApp()
    w.show()
    sys.exit(app.exec_())
