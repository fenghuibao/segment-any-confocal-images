from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from aicsimageio import AICSImage
from qtpy.QtCore import QLocale, QObject, QThread, QTimer, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from skimage.exposure import rescale_intensity
from superqt import QRangeSlider
from scipy.signal import medfilt

# External modules you already use
from frangi_filter.frangi_filter import *  # noqa: F401,F403
from SegmentAnyConfocal.SegmentAnyConfocal import *  # noqa: F401,F403

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else ([x] if x is not None else [])


def _get_source_from_layer(layer) -> Optional[str]:
    md = getattr(layer, "metadata", {}) or {}
    for key in ("source", "original_path", "path", "file"):
        if key in md:
            return md[key]
    return getattr(getattr(layer, "source", None), "path", None)


def _get_units_from_layer(layer) -> str:
    md = getattr(layer, "metadata", {}) or {}
    return md.get("PhysicalSizeXUnit") or md.get("unit") or md.get("Units") or "um"


def _ensure_unit_in_metadata(md: Dict[str, Any]) -> str:
    unit = md.get("PhysicalSizeXUnit") or md.get("unit") or md.get("Units") or "um"
    md["unit"] = unit
    return unit


def _ensure_group_id(md: Dict[str, Any], fallback: Optional[str] = None) -> str:
    gid = md.get("group_id") or fallback or uuid4().hex
    md["group_id"] = gid
    return gid


def _get_group_id(layer) -> Optional[str]:
    md = getattr(layer, "metadata", {}) or {}
    gid = md.get("group_id")
    if gid:
        return str(gid)
    return None


def _safe_axis_labels(viewer, layer):
    # Respect dimensionality, avoid forcing a bogus T axis on 2D/3D layers
    nd = layer.data.ndim
    if nd >= 4:
        labels = ("T", "Z", "Y", "X")
    elif nd == 3:
        labels = ("Z", "Y", "X")
    elif nd == 2:
        labels = ("Y", "X")
    else:
        labels = tuple(str(i) for i in range(nd))
    try:
        viewer.dims.axis_labels = labels
    except Exception:
        pass


def _describe_size_and_resolution(layer) -> Dict[str, Any]:
    data = np.asarray(layer.data)
    result: Dict[str, Any] = {}
    sc = tuple(getattr(layer, "scale", (1,) * data.ndim))
    unit = _get_units_from_layer(layer)

    if data.ndim >= 3:
        nz, ny, nx = data.shape[-3], data.shape[-2], data.shape[-1]
        vz, vy, vx = float(sc[-3]), float(sc[-2]), float(sc[-1])
        phys = (nz * vz, ny * vy, nx * vx)
        result.update(
            dict(
                ndim=3,
                shape=(nz, ny, nx),
                voxel=(vz, vy, vx),
                phys=(float(phys[0]), float(phys[1]), float(phys[2])),
                unit=unit,
            )
        )
    elif data.ndim == 2:
        ny, nx = data.shape[-2], data.shape[-1]
        vy, vx = float(sc[-2]), float(sc[-1])
        phys = (ny * vy, nx * vx)
        result.update(
            dict(
                ndim=2,
                shape=(ny, nx),
                voxel=(vy, vx),
                phys=(float(phys[0]), float(phys[1])),
                unit=unit,
            )
        )
    else:
        result.update(dict(ndim=data.ndim, shape=tuple(data.shape)))
    return result


def _try_autofill_scale_from_source(layer) -> bool:
    src = _get_source_from_layer(layer)
    if not src:
        return False
    try:
        img = AICSImage(src)
        px = img.physical_pixel_sizes
        sc = list(getattr(layer, "scale", (1,) * layer.data.ndim))
        if layer.data.ndim >= 3 and px.Z and px.Y and px.X:
            sc[-3:] = [float(px.Z), float(px.Y), float(px.X)]
        elif layer.data.ndim == 2 and px.Y and px.X:
            sc[-2:] = [float(px.Y), float(px.X)]
        else:
            return False
        layer.scale = tuple(sc)
        md = dict(getattr(layer, "metadata", {}) or {})
        _ensure_unit_in_metadata(md)
        _ensure_group_id(md)
        layer.metadata = md
        return True
    except Exception:
        return False


def _rescale_0_255_tczyx(arr: np.ndarray) -> np.ndarray:
    """Linearly rescale to uint8 [0..255] per (T,C) block across Z,Y,X."""
    a = np.asarray(arr)
    if a.ndim == 5:
        mins = a.min(axis=(-3, -2, -1), keepdims=True).astype(np.float32)
        maxs = a.max(axis=(-3, -2, -1), keepdims=True).astype(np.float32)
        rng = (maxs - mins)
        rng[rng == 0] = 1.0
        out = (a.astype(np.float32) - mins) / rng * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)
    if a.ndim == 4:
        a5 = a[None, ...]
        return _rescale_0_255_tczyx(a5)[0]
    if a.ndim == 3:
        mn, mx = float(a.min()), float(a.max())
        rng = (mx - mn) if (mx > mn) else 1.0
        out = (a.astype(np.float32) - mn) / rng * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)
    return np.clip(a, 0, 255).astype(np.uint8)


def _load_image_tc_zyx(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load to TCZYX and pack metadata including Z/Y/X scales."""
    img = AICSImage(path)
    data = img.get_image_data("TCZYX")
    pps = img.physical_pixel_sizes
    zyx_scale = [
        float(pps.Z) if pps.Z else 1.0,
        float(pps.Y) if pps.Y else 1.0,
        float(pps.X) if pps.X else 1.0,
    ]
    ch_names = None
    try:
        ch_names = [
            (c.name if getattr(c, "name", None) else f"Channel {i+1}")
            for i, c in enumerate(img.metadata.images[0].pixels.channels)
        ]
    except Exception:
        pass
    meta: Dict[str, Any] = dict(
        dims="TCZYX",
        channel_names=ch_names,
        unit="um",
        scale_per_axis=(1.0, 1.0, *zyx_scale),
        source=path,
    )
    _ensure_group_id(meta)
    return data, meta


def _get_pixel_size_tuple(layer, ndim: int):
    sc = getattr(layer, "scale", (1,) * layer.data.ndim)
    if ndim == 3:
        return (float(sc[-3]), float(sc[-2]), float(sc[-1]))
    if ndim == 2:
        return (float(sc[-2]), float(sc[-1]))
    raise ValueError(f"Unsupported ndim for pixel size: {ndim}")


# ---------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------

class SegWorker(QObject):
    finished = Signal(object, object)  # (payload, error)
    started = Signal()
    progress = Signal(int, float)  # (iteration, delta)

    def __init__(self, kwargs: Dict[str, Any]):
        super().__init__()
        self.kwargs = kwargs

    def run(self):
        self.started.emit()
        try:
            def _progress_cb(it, delta):
                try:
                    self.progress.emit(int(it), float(delta))
                except Exception:
                    pass

            kwargs = dict(self.kwargs)
            kwargs["progress"] = _progress_cb
            seg, info = segmentation(**kwargs)
            seg = seg[0]
            self.finished.emit((seg, info), None)
        except Exception as e:
            self.finished.emit(None, e)


# ---------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------

class SegmentConfocalWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._group_layers: List = []
        self._last_frangi_ctx: Dict[str, Any] = {}
        self._frangi_layer = None
        self._denoise_layer = None
        self._denoise_ctx: Dict[str, Any] = {}
        self._low_percent = 0
        self._view_initialized = False  # only on first load

        # ---- Top-level: scroll area with a single content widget -------
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget()
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        self.setMinimumWidth(460)
        self.setMaximumWidth(560)

        layout.addWidget(self._build_top_box())
        layout.addWidget(self._build_info_box())
        layout.addWidget(self._build_frangi_box())
        layout.addWidget(self._build_seg_box())
        layout.addStretch(1)

        outer.addWidget(scroll)

        # ---- listeners -------------------------------------------------
        self.viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self.viewer.layers.events.removed.connect(self._update_info)
        self.viewer.layers.selection.events.active.connect(self._on_active_changed)
        try:
            self.viewer.dims.events.current_step.connect(self._on_dims_step_changed)
        except Exception:
            pass
        self.range_slider.valueChanged.connect(self._on_rescale_params_changed)

        # initial state
        self._update_info()
        self._fit_view_and_scalebar()
        self._patch_open_and_drop()

    # -----------------------------------------------------------------
    # UI builders
    # -----------------------------------------------------------------

    def _build_top_box(self) -> QGroupBox:
        box = QGroupBox("Data / Axes / Device")
        g = QGridLayout()
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(6)

        self.open_btn = QPushButton("Select File")
        self.open_btn.clicked.connect(self._open_image_tc_zyx)
        self.device_combo = QComboBox()
        self._populate_devices()
        self.device_combo.currentTextChanged.connect(self._on_device_changed)

        g.addWidget(self.open_btn, 0, 0)
        g.addWidget(QLabel("Device:"), 0, 1, alignment=Qt.AlignRight)
        g.addWidget(self.device_combo, 0, 2)

        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        g.addWidget(QLabel("Path:"), 1, 0, alignment=Qt.AlignRight)
        g.addWidget(self.path_edit, 1, 1, 1, 2)

        self.c_spin = QSpinBox()
        self.c_spin.setRange(0, 0)  # 0-based
        self.c_spin.setEnabled(False)
        self.c_spin.setValue(0)
        self.c_spin.valueChanged.connect(self._on_c_changed)

        self.t_spin = QSpinBox()
        self.t_spin.setRange(0, 0)  # 0-based
        self.t_spin.setEnabled(False)
        self.t_spin.setValue(0)
        self.t_spin.valueChanged.connect(self._on_t_changed)

        row_ct = QWidget()
        row_ct_layout = QHBoxLayout()
        row_ct_layout.setContentsMargins(0, 0, 0, 0)
        row_ct_layout.addWidget(QLabel("Channel:"))
        row_ct_layout.addWidget(self.c_spin)
        row_ct_layout.addSpacing(12)
        row_ct_layout.addWidget(QLabel("Time:"))
        row_ct_layout.addWidget(self.t_spin)
        row_ct.setLayout(row_ct_layout)
        g.addWidget(row_ct, 2, 0, 1, 3)

        box.setLayout(g)
        return box

    def _build_info_box(self) -> QGroupBox:
        box = QGroupBox("Image Info")
        g = QGridLayout()
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(6)

        self.lbl_shape = QLabel("–")
        self.lbl_phys = QLabel("–")
        self.lbl_voxpix = QLabel("–")

        g.addWidget(QLabel("Shape:"), 0, 0, alignment=Qt.AlignRight)
        g.addWidget(self.lbl_shape, 0, 1)
        g.addWidget(QLabel("Physical size:"), 1, 0, alignment=Qt.AlignRight)
        g.addWidget(self.lbl_phys, 1, 1)
        g.addWidget(QLabel("Voxel size:"), 2, 0, alignment=Qt.AlignRight)
        g.addWidget(self.lbl_voxpix, 2, 1)

        # Editable voxel/pixel sizes
        self.edit_vz = QDoubleSpinBox()
        self.edit_vz.setDecimals(6)
        self.edit_vz.setRange(0, 1e3)
        self.edit_vz.setLocale(QLocale.c())

        self.edit_vyx = QDoubleSpinBox()
        self.edit_vyx.setDecimals(6)
        self.edit_vyx.setRange(0, 1e3)
        self.edit_vyx.setLocale(QLocale.c())

        self.btn_apply_voxel = QPushButton("Update Voxel/Pixel size")
        self.btn_apply_voxel.clicked.connect(self._on_apply_voxel_clicked)

        vox_row = 3
        g.addWidget(QLabel("Edit size:"), vox_row, 0, alignment=Qt.AlignRight)

        col_widget = QWidget()
        col_layout = QGridLayout()
        col_layout.setContentsMargins(0, 0, 0, 0)
        col_layout.setHorizontalSpacing(8)

        self.lblZ = QLabel("Z:")
        self.lblYX = QLabel("Y/X:")
        col_layout.addWidget(self.lblZ, 0, 0, alignment=Qt.AlignRight)
        col_layout.addWidget(self.edit_vz, 0, 1)
        col_layout.addWidget(self.lblYX, 0, 2, alignment=Qt.AlignRight)
        col_layout.addWidget(self.edit_vyx, 0, 3)
        col_widget.setLayout(col_layout)

        g.addWidget(col_widget, vox_row, 1)
        g.addWidget(self.btn_apply_voxel, vox_row + 1, 1)

        # --- Denoise controls (checkbox + size + apply button) ----------
        self.chk_denoise = QCheckBox("Denoise")
        self.cmb_kernel = QComboBox()
        for k in (3, 5, 7, 11):
            self.cmb_kernel.addItem(str(k))
        self.cmb_kernel.setCurrentText("3")

        self.btn_apply_denoise = QPushButton("Apply median filter")
        self.btn_apply_denoise.clicked.connect(self._on_apply_denoise_clicked)

        row_med = QWidget()
        row_med_l = QHBoxLayout()
        row_med_l.setContentsMargins(0, 0, 0, 0)
        row_med_l.addWidget(self.chk_denoise)
        row_med_l.addStretch(1)
        row_med_l.addWidget(QLabel("filter size"))
        row_med_l.addWidget(self.cmb_kernel)
        row_med_l.addWidget(self.btn_apply_denoise)
        row_med.setLayout(row_med_l)
        g.addWidget(row_med, vox_row + 2, 0, 1, 2)

        # default scale bar
        try:
            self.viewer.scale_bar.visible = True
            self.viewer.scale_bar.unit = "um"
        except Exception:
            pass

        box.setLayout(g)
        return box

    def _build_frangi_box(self) -> QGroupBox:
        box = QGroupBox("Frangi Filter")
        g = QGridLayout()
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(6)

        self.kernel_spin = QSpinBox()
        self.kernel_spin.setRange(1, 5)
        self.kernel_spin.setValue(4)

        self.sigma_count = QSpinBox()
        self.sigma_count.setRange(1, 99)
        self.sigma_count.setValue(5)

        g.addWidget(QLabel("Kernel radius:"), 0, 0, alignment=Qt.AlignRight)
        g.addWidget(self.kernel_spin, 0, 1)
        g.addWidget(QLabel("Sigma count:"), 0, 2, alignment=Qt.AlignRight)
        g.addWidget(self.sigma_count, 0, 3)

        self.sigma_min = QDoubleSpinBox()
        self.sigma_min.setDecimals(2)
        self.sigma_min.setRange(0.1, 5.0)
        self.sigma_min.setSingleStep(0.1)
        self.sigma_min.setValue(0.1)
        self.sigma_min.setLocale(QLocale.c())

        self.sigma_max = QDoubleSpinBox()
        self.sigma_max.setDecimals(2)
        self.sigma_max.setRange(0.1, 5.0)
        self.sigma_max.setSingleStep(0.1)
        self.sigma_max.setValue(1.0)
        self.sigma_max.setLocale(QLocale.c())

        g.addWidget(QLabel("Sigma min:"), 1, 0, alignment=Qt.AlignRight)
        g.addWidget(self.sigma_min, 1, 1)
        g.addWidget(QLabel("Sigma max:"), 1, 2, alignment=Qt.AlignRight)
        g.addWidget(self.sigma_max, 1, 3)

        self.chk_use_2d = QCheckBox("Run on 2d slice")
        self.z_index_spin = QSpinBox()
        self.z_index_spin.setRange(0, 0)  # 0-based
        self.z_index_spin.setValue(0)
        self.z_index_spin.setEnabled(False)

        def _toggle_slice(v):
            self.z_index_spin.setEnabled(bool(v))
            self._sync_z_spin_to_view()

        self.chk_use_2d.toggled.connect(_toggle_slice)

        row2d = QWidget()
        row2d_layout = QHBoxLayout()
        row2d_layout.setContentsMargins(0, 0, 0, 0)
        row2d_layout.addWidget(self.chk_use_2d)
        row2d_layout.addWidget(QLabel("Frame Index:"))
        row2d_layout.addWidget(self.z_index_spin)
        row2d.setLayout(row2d_layout)
        g.addWidget(row2d, 2, 0, 1, 4)

        # Optional intensity rescale (percentiles)
        self.chk_rescale = QCheckBox("Enable intensity rescale (percentiles)")
        self.chk_rescale.setChecked(False)
        self.chk_rescale.setEnabled(False)
        self.chk_rescale.toggled.connect(self._on_toggle_rescale)

        self.range_slider = QRangeSlider(Qt.Horizontal)
        self.range_slider.setRange(0, 100)
        self.range_slider.setValue((0, 100))
        self.range_slider.setEnabled(False)

        self.range_lbl = QLabel("percentiles: 0%–100%")

        g.addWidget(self.chk_rescale, 3, 0, 1, 4)

        row_range = QWidget()
        row_range_l = QHBoxLayout()
        row_range_l.setContentsMargins(0, 0, 0, 0)
        row_range_l.addWidget(QLabel("range"))
        row_range_l.addWidget(self.range_slider)
        row_range_l.addWidget(self.range_lbl)
        row_range.setLayout(row_range_l)
        g.addWidget(row_range, 4, 0, 1, 4)

        self.apply_btn = QPushButton("Run Frangi Filter")
        self.apply_btn.setMinimumHeight(32)
        self.apply_btn.clicked.connect(self._on_apply_frangi_clicked)
        g.addWidget(self.apply_btn, 5, 0, 1, 4)

        box.setLayout(g)
        return box

    def _build_seg_box(self) -> QGroupBox:
        box = QGroupBox("Segmentation")
        g = QGridLayout()
        g.setHorizontalSpacing(16)
        g.setVerticalSpacing(8)

        self.beta1_spin = QDoubleSpinBox()
        self.beta1_spin.setDecimals(2)
        self.beta1_spin.setRange(0.0, 10)
        self.beta1_spin.setSingleStep(0.01)
        self.beta1_spin.setValue(1.0)
        self.beta1_spin.setLocale(QLocale.c())

        self.beta2_spin = QDoubleSpinBox()
        self.beta2_spin.setDecimals(2)
        self.beta2_spin.setRange(0.0, 10)
        self.beta2_spin.setSingleStep(0.01)
        self.beta2_spin.setValue(2.0)
        self.beta2_spin.setLocale(QLocale.c())

        self.cutoff_spin = QDoubleSpinBox()
        self.cutoff_spin.setDecimals(2)
        self.cutoff_spin.setRange(0, 10)
        self.cutoff_spin.setSingleStep(0.01)
        self.cutoff_spin.setValue(5.0)
        self.cutoff_spin.setLocale(QLocale.c())

        self.nfore_spin = QSpinBox()
        self.nfore_spin.setRange(0, 16)
        self.nfore_spin.setValue(8)

        self.nback_spin = QSpinBox()
        self.nback_spin.setRange(0, 16)
        self.nback_spin.setValue(3)

        self.maxiter_spin = QSpinBox()
        self.maxiter_spin.setRange(1, 200)
        self.maxiter_spin.setValue(50)

        g.addWidget(QLabel("beta1:"), 0, 0, alignment=Qt.AlignRight)
        g.addWidget(self.beta1_spin, 0, 1)
        g.addWidget(QLabel("beta2:"), 0, 2, alignment=Qt.AlignRight)
        g.addWidget(self.beta2_spin, 0, 3)
        g.addWidget(QLabel("cutoff:"), 1, 0, alignment=Qt.AlignRight)
        g.addWidget(self.cutoff_spin, 1, 1)
        g.addWidget(QLabel("maxiter:"), 1, 2, alignment=Qt.AlignRight)
        g.addWidget(self.maxiter_spin, 1, 3)
        g.addWidget(QLabel("nforeground:"), 2, 0, alignment=Qt.AlignRight)
        g.addWidget(self.nfore_spin, 2, 1)
        g.addWidget(QLabel("nbackground:"), 2, 2, alignment=Qt.AlignRight)
        g.addWidget(self.nback_spin, 2, 3)

        self.seg_progress = QProgressBar()
        self.seg_progress.setValue(0)
        self.seg_progress.setTextVisible(True)
        g.addWidget(self.seg_progress, 3, 0, 1, 4)

        self.kw_preview = QTextEdit()
        self.kw_preview.setReadOnly(True)
        self.kw_preview.setMinimumHeight(90)
        g.addWidget(self.kw_preview, 4, 0, 1, 4)

        self.seg_btn = QPushButton("Run segmentation")
        self.seg_btn.setMinimumHeight(32)
        self.seg_btn.clicked.connect(self._on_run_segmentation_clicked)
        g.addWidget(self.seg_btn, 5, 0, 1, 4)

        box.setLayout(g)
        return box

    # -----------------------------------------------------------------
    # Open / drop interception & unified loader with 0–255 display rescale
    # -----------------------------------------------------------------

    def _patch_open_and_drop(self):
        """Intercept drop/open; use AICSImage for supported formats."""
        try:
            qt = self.viewer.window._qt_viewer
        except Exception:
            return

        # Wrap _qt_open
        if hasattr(qt, "_qt_open") and not hasattr(self, "_orig__qt_open"):
            self._orig__qt_open = qt._qt_open

            def _wrapped__qt_open(filenames, stack=False, choose_plugin=False, plugin=None, layer_type=None, **kwargs):
                try:
                    one_file = (
                        (isinstance(filenames, (list, tuple)) and len(filenames) == 1)
                        or isinstance(filenames, str)
                    )
                    if one_file and self._try_open_with_aics(filenames, stack=stack, choose_plugin=choose_plugin, **kwargs):
                        return []
                except Exception:
                    pass
                return self._orig__qt_open(
                    filenames,
                    stack=stack,
                    choose_plugin=choose_plugin,
                    plugin=plugin,
                    layer_type=layer_type,
                    **kwargs,
                )

            qt._qt_open = _wrapped__qt_open

        # Wrap dropEvent
        if hasattr(qt, "dropEvent") and not hasattr(self, "_orig_dropEvent"):
            self._orig_dropEvent = qt.dropEvent
            SUPP = (".czi", ".nd2", ".lif", ".lsm", ".tif", ".tiff", ".ome.tif", ".ome.tiff", ".png", ".jpg", ".jpeg")

            def _custom_drop(ev):
                try:
                    urls = ev.mimeData().urls()
                    if urls:
                        paths = []
                        for u in urls:
                            if u.isLocalFile():
                                paths.append(u.toLocalFile())
                            else:
                                s = u.toString()
                                if s.startswith("file://"):
                                    paths.append(s[7:])
                        if len(paths) == 1 and any(paths[0].lower().endswith(ext) for ext in SUPP):
                            if self._try_open_with_aics(paths[0]):
                                ev.acceptProposedAction()
                                return
                except Exception:
                    pass
                return self._orig_dropEvent(ev)

            qt.dropEvent = _custom_drop

    def _try_open_with_aics(self, paths, stack=False, choose_plugin=False, **kwargs) -> bool:
        if isinstance(paths, (list, tuple)):
            if len(paths) != 1:
                return False
            path = paths[0]
        else:
            path = paths

        try:
            _ = AICSImage(path)
        except Exception:
            return False

        self._load_and_add_path(path)
        return True

    def _load_and_add_path(self, path: str):
        data_tc_zyx, meta = _load_image_tc_zyx(path)
        data_uint8 = _rescale_0_255_tczyx(data_tc_zyx)
        self.path_edit.setText(path)

        added = self.viewer.add_image(
            data_uint8,
            name=os.path.basename(path),
            channel_axis=1,
            rgb=False,
            blending="additive",
            visible=True,
            metadata=dict(meta),
        )
        layers = _as_list(added)
        self._group_layers = layers

        sc5 = tuple(meta.get("scale_per_axis", (1, 1, 1, 1, 1)))
        unit = meta.get("unit", "um")
        ch_names = meta.get("channel_names") or [f"Channel {i+1}" for i in range(len(layers))]
        gid = meta.get("group_id", uuid4().hex)

        for i, lyr in enumerate(layers):
            sc = (1.0, sc5[-3], sc5[-2], sc5[-1]) if lyr.data.ndim == 4 else (sc5[-2], sc5[-1])
            try:
                lyr.scale = sc
            except Exception:
                lyr.scale = (1.0,) * (lyr.data.ndim - 3) + (sc5[-3], sc5[-2], sc5[-1])

            md = dict(getattr(lyr, "metadata", {}) or {})
            md.setdefault("unit", unit)
            md.setdefault("dims", "TCZYX")
            md.setdefault("source", path)
            md["group_id"] = gid
            md["channel_index"] = i
            md["channel_name"] = ch_names[i] if i < len(ch_names) else f"Channel {i+1}"
            lyr.metadata = md

            try:
                lyr.contrast_limits = (0, 255)
            except Exception:
                pass

        self._set_ct_controls(T=int(data_uint8.shape[0]), C=int(data_uint8.shape[1]))
        if layers:
            self.viewer.layers.selection.active = layers[0]
            _safe_axis_labels(self.viewer, layers[0])

        try:
            self.viewer.scale_bar.visible = True
            self.viewer.scale_bar.unit = unit
            self.viewer.scale_bar.position = "bottom_right"
            self.viewer.scale_bar.font_size = 8
        except Exception:
            pass

        try:
            if data_uint8.shape[2] > 0:
                self.z_index_spin.setRange(0, int(data_uint8.shape[2]) - 1)  # 0-based
        except Exception:
            pass

        self._view_initialized = False
        self._fit_view_and_scalebar()
        self._update_info()
        try:
            self._update_segmentation_preview()
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Device helpers
    # -----------------------------------------------------------------

    def _populate_devices(self):
        self.device_combo.clear()
        choices = ["cpu"]
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    choices.append("cuda")
            except Exception:
                pass
            try:
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    choices.append("mps")
            except Exception:
                pass
        for c in choices:
            self.device_combo.addItem(c)
        if "cuda" in choices:
            self.device_combo.setCurrentText("cuda")
        elif "mps" in choices:
            self.device_combo.setCurrentText("mps")
        else:
            self.device_combo.setCurrentText("cpu")
        if hasattr(self, "kw_preview"):
            self._update_segmentation_preview()

    def _resolve_device(self) -> str:
        if torch is None:
            return "cpu"
        try:
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    # -----------------------------------------------------------------
    # File dialog open & normalization
    # -----------------------------------------------------------------

    def _open_image_tc_zyx(self):
        from qtpy.QtWidgets import QFileDialog
        dlg = QFileDialog(self, "Open image")
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter(
            "Images (*.tif *.tiff *.czi *.nd2 *.lif *.lsm *.ome.tif *.ome.tiff *.png *.jpg *.jpeg);;All files (*)"
        )
        if not dlg.exec_():
            return
        path = dlg.selectedFiles()[0]
        try:
            self._load_and_add_path(path)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", f"Failed to load: {e!r}")

    def _on_layer_inserted(self, event=None):
        layer = getattr(event, "value", None)
        if layer is None:
            return

        def _deferred():
            try:
                self._maybe_normalize_dragdrop_layer(layer)
            except Exception:
                pass
            self._view_initialized = False
            self._fit_view_and_scalebar()
            self._update_info()

        QTimer.singleShot(0, _deferred)

    def _maybe_normalize_dragdrop_layer(self, layer):
        md0 = getattr(layer, "metadata", {}) or {}
        if md0.get("is_frangi") or md0.get("is_segmentation") or md0.get("is_denoise"):
            return

        try:
            from napari.layers import Image as NapariImage
            if not isinstance(layer, NapariImage):
                return
        except Exception:
            return

        src = _get_source_from_layer(layer)
        if src:
            self.path_edit.setText(src)
        if not src:
            return

        if (getattr(layer, "metadata", {}) or {}).get("dims") == "TCZYX":
            _safe_axis_labels(self.viewer, layer)
            T = int(layer.data.shape[0]) if layer.data.ndim >= 4 else 1
            same = [l for l in self.viewer.layers if getattr(l, "metadata", {}).get("source") == src]
            gid = None
            for l in same:
                gid = _get_group_id(l) or gid
            gid = gid or uuid4().hex
            for l in same:
                md_same = dict(getattr(l, "metadata", {}) or {})
                md_same["group_id"] = gid
                l.metadata = md_same
            self._group_layers = same
            self._set_ct_controls(T=T, C=len(same))
            try:
                if layer.data.ndim >= 3:
                    self.z_index_spin.setRange(0, int(layer.data.shape[-3]) - 1)
                else:
                    self.z_index_spin.setRange(0, 0)
            except Exception:
                pass
            self._view_initialized = False
            self._fit_view_and_scalebar()
            return

        try:
            data, meta = _load_image_tc_zyx(src)
            data = _rescale_0_255_tczyx(data)
            base = os.path.basename(src)
            name = base
            existing = {l.name for l in self.viewer.layers}
            k = 1
            while name in existing:
                k += 1
                name = f"{base} ({k})"
            added = self.viewer.add_image(
                data,
                name=name,
                channel_axis=1,
                rgb=False,
                blending="additive",
                visible=True,
                metadata=dict(meta),
            )
            new_layers = _as_list(added)
            self._group_layers = new_layers

            sc5 = tuple(meta.get("scale_per_axis", (1, 1, 1, 1, 1)))
            unit = meta.get("unit", "um")
            ch_names = meta.get("channel_names") or [f"Channel {i+1}" for i in range(len(new_layers))]
            gid = meta.get("group_id", uuid4().hex)

            for i, lyr in enumerate(new_layers):
                sc = (1.0, sc5[-3], sc5[-2], sc5[-1]) if lyr.data.ndim == 4 else (sc5[-2], sc5[-1])
                try:
                    lyr.scale = sc
                except Exception:
                    lyr.scale = (1.0,) * (lyr.data.ndim - 3) + (sc5[-3], sc5[-2], sc5[-1])
                md = dict(getattr(lyr, "metadata", {}) or {})
                md.setdefault("unit", unit)
                md.setdefault("dims", "TCZYX")
                md.setdefault("source", src)
                md["group_id"] = gid
                md["channel_index"] = i
                md["channel_name"] = ch_names[i] if i < len(ch_names) else f"Channel {i+1}"
                lyr.metadata = md
                try:
                    lyr.contrast_limits = (0, 255)
                except Exception:
                    pass

            if new_layers:
                self.viewer.layers.selection.active = new_layers[0]
                _safe_axis_labels(self.viewer, new_layers[0])

            try:
                self.viewer.layers.remove(layer)
            except Exception:
                pass

            self._set_ct_controls(T=int(data.shape[0]), C=int(data.shape[1]))
            try:
                self.z_index_spin.setRange(0, int(data.shape[2]) - 1)
            except Exception:
                pass

            self._view_initialized = False
            self._fit_view_and_scalebar()
            self._update_info()
            return
        except Exception:
            _try_autofill_scale_from_source(layer)
            try:
                md = layer.metadata if layer.metadata else {}
                md["unit"] = _ensure_unit_in_metadata(md)
                _ensure_group_id(md)
                layer.metadata = md
            except Exception:
                pass
            try:
                self.viewer.scale_bar.visible = True
                self.viewer.scale_bar.unit = _get_units_from_layer(layer)
            except Exception:
                pass
            _safe_axis_labels(self.viewer, layer)
            T = int(layer.data.shape[0]) if layer.data.ndim >= 4 else 1
            self._group_layers = [layer]
            self._set_ct_controls(T=T, C=1)
            self._update_info()
            try:
                if layer.data.ndim >= 3:
                    self.z_index_spin.setRange(0, int(layer.data.shape[-3]) - 1)
                else:
                    self.z_index_spin.setRange(0, 0)
            except Exception:
                pass

    # -----------------------------------------------------------------
    # Viewer sync & UI updates
    # -----------------------------------------------------------------

    def _on_device_changed(self, *_):
        if self._last_frangi_ctx:
            self._last_frangi_ctx["device"] = self._resolve_device()
        self._update_segmentation_preview()

    def _on_active_changed(self, event=None):
        # Keep user-selected slice/zoom
        layer = self.viewer.layers.selection.active

        try:
            if hasattr(self, "_scale_conn") and self._scale_conn is not None:
                self._scale_conn.disconnect()
        except Exception:
            pass

        if layer is not None and hasattr(layer, "events") and hasattr(layer.events, "scale"):
            self._scale_conn = layer.events.scale.connect(self._update_info)
        else:
            self._scale_conn = None

        if layer is not None:
            md = getattr(layer, "metadata", {}) or {}
            if "channel_index" in md and self.c_spin.isEnabled():
                self.c_spin.blockSignals(True)
                self.c_spin.setValue(int(md["channel_index"]))  # 0-based
                self.c_spin.blockSignals(False)
            try:
                t_cur0 = int(self.viewer.dims.current_step[0])
                if self.t_spin.isEnabled():
                    self.t_spin.blockSignals(True)
                    self.t_spin.setValue(t_cur0)  # 0-based
                    self.t_spin.blockSignals(False)
            except Exception:
                pass

        self._update_info()

    def _on_dims_step_changed(self, event=None):
        try:
            t_cur0 = int(self.viewer.dims.current_step[0])
            if self.t_spin.isEnabled():
                self.t_spin.blockSignals(True)
                self.t_spin.setValue(t_cur0)  # 0-based
                self.t_spin.blockSignals(False)
        except Exception:
            pass
        self._sync_z_spin_to_view()
        self._update_info()

    def _sync_z_spin_to_view(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            return
        arr = np.asarray(layer.data)
        nd = arr.ndim
        if nd < 3:
            return
        try:
            zlen = int(arr.shape[-3])
            self.z_index_spin.setRange(0, max(zlen - 1, 0))  # 0-based
            if not self.z_index_spin.hasFocus():
                zcur = int(self.viewer.dims.current_step[-3])
                self.z_index_spin.setValue(max(min(zcur, zlen - 1), 0))  # 0-based
        except Exception:
            pass

    def _update_info(self, event=None):
        layer = self.viewer.layers.selection.active
        if layer is None or not hasattr(layer, "data"):
            self.lbl_shape.setText("–")
            self.lbl_phys.setText("–")
            self.lbl_voxpix.setText("–")
            return

        sc = getattr(layer, "scale", (1,) * layer.data.ndim)
        check_n = 3 if layer.data.ndim >= 3 else 2
        try:
            if all(abs(float(s) - 1.0) < 1e-12 for s in sc[-check_n:]):
                _try_autofill_scale_from_source(layer)
        except Exception:
            pass

        try:
            info = _describe_size_and_resolution(layer)
        except Exception as e:
            self.lbl_shape.setText(f"(error) {e!r}")
            return

        if info.get("ndim") == 3:
            nz, ny, nx = info["shape"]
            vz, vy, vx = info["voxel"]
            pz, py, px = info["phys"]
            self.lbl_shape.setText(f"Z={nz}  Y={ny}  X={nx}")
            self.lbl_phys.setText(f"{pz:.6g}×{py:.6g}×{px:.6g} {info['unit']}")
            self.lbl_voxpix.setText(f"vz={vz:.6g}  vy={vy:.6g}  vx={vx:.6g} {info['unit']}/px")
            self.lblZ.show()
            self.edit_vz.show()
            self.edit_vz.setEnabled(True)
            self.lblYX.setText("Y/X:")
            self.edit_vyx.show()
            self.edit_vyx.setEnabled(True)
            self.edit_vz.setValue(max(vz, 0))
            self.edit_vyx.setValue(max(vy, 0))
            self.btn_apply_voxel.setText("Update Voxel size")
        elif info.get("ndim") == 2:
            ny, nx = info["shape"]
            vy, vx = info["voxel"]
            py, px = info["phys"]
            self.lbl_shape.setText(f"Y={ny}  X={nx}")
            self.lbl_phys.setText(f"{py:.6g}×{px:.6g} {info['unit']}")
            self.lbl_voxpix.setText(f"py={vy:.6g}  px={vx:.6g} {info['unit']}/px")
            self.lblZ.hide()
            self.edit_vz.hide()
            self.edit_vz.setEnabled(False)
            self.lblYX.setText("Y/X:")
            self.edit_vyx.show()
            self.edit_vyx.setEnabled(True)
            self.edit_vyx.setValue(max(vy, 0))
            self.btn_apply_voxel.setText("Update Pixel size")
        else:
            self.lbl_shape.setText(str(info.get("shape")))
            self.lbl_phys.setText("–")
            self.lbl_voxpix.setText("–")

        _safe_axis_labels(self.viewer, layer)
        try:
            self.viewer.scale_bar.visible = True
            self.viewer.scale_bar.unit = _get_units_from_layer(layer)
        except Exception:
            pass

        self._update_segmentation_preview()

    def _fit_view_and_scalebar(self):
        layer = self.viewer.layers.selection.active
        if layer is None or not hasattr(layer, "data"):
            return

        if not self._view_initialized:
            self._view_initialized = True
            try:
                if np.all(np.isfinite(layer.extent.world[0])) and np.all(np.isfinite(layer.extent.world[1])):
                    self.viewer.reset_view()
                nd, sh = layer.data.ndim, layer.data.shape
                for ax in range(max(0, nd - 2)):
                    if sh[ax] > 1:
                        try:
                            self.viewer.dims.set_current_step(ax, int(sh[ax] // 2))
                        except Exception:
                            pass
            except Exception:
                pass
            
            # --- Sync Time spinbox with viewer after centering ---
            try:
                if nd >= 4 and self.t_spin.isEnabled():
                    cur_t = int(self.viewer.dims.current_step[0])
                    self.t_spin.blockSignals(True)
                    self.t_spin.setValue(cur_t)
                    self.t_spin.blockSignals(False)
            except Exception:
                pass

        try:
            self.viewer.scale_bar.visible = True
            unit = (getattr(layer, "metadata", {}) or {}).get("unit", "um")
            self.viewer.scale_bar.unit = unit
            self.viewer.scale_bar.position = "bottom_right"
            self.viewer.scale_bar.font_size = 8
        except Exception:
            pass

        _safe_axis_labels(self.viewer, layer)

    # -----------------------------------------------------------------
    # Voxel/pixel size handling
    # -----------------------------------------------------------------

    def _on_apply_voxel_clicked(self):
        layer = self.viewer.layers.selection.active
        if layer is None or not hasattr(layer, "data"):
            QMessageBox.information(self, "No image", "Please select an image layer first.")
            return

        nd = layer.data.ndim
        sc = list(getattr(layer, "scale", (1,) * nd))
        try:
            if nd >= 3:
                vz = float(self.edit_vz.value()) if self.edit_vz.isEnabled() and self.edit_vz.isVisible() else float(sc[-3])
                vxy = float(self.edit_vyx.value())
                sc[-3:] = [vz, vxy, vxy]
                new_triplet = (float(sc[-3]), float(sc[-2]), float(sc[-1]))
            else:
                vxy = float(self.edit_vyx.value())
                sc[-2:] = [vxy, vxy]
                new_triplet = (None, float(sc[-2]), float(sc[-1]))
            layer.scale = tuple(sc)

            md = dict(getattr(layer, "metadata", {}) or {})
            _ensure_unit_in_metadata(md)
            gid = _ensure_group_id(md)
            layer.metadata = md

            for lyr in list(self.viewer.layers):
                if not hasattr(lyr, "data") or lyr is layer:
                    continue
                md2 = getattr(lyr, "metadata", {}) or {}
                if md2.get("group_id") == gid:
                    sc2 = list(getattr(lyr, "scale", (1,) * lyr.data.ndim))
                    if lyr.data.ndim >= 3 and new_triplet[0] is not None:
                        sc2[-3:] = list(new_triplet)
                    else:
                        sc2[-2:] = [new_triplet[1], new_triplet[2]]
                    lyr.scale = tuple(sc2)

            if self._last_frangi_ctx:
                try:
                    dim = int(self._last_frangi_ctx.get("dim", 2))
                    self._last_frangi_ctx["pixel_size"] = _get_pixel_size_tuple(layer, dim)
                except Exception:
                    pass

            self._fit_view_and_scalebar()
            self._update_info()
        except Exception as e:
            QMessageBox.critical(self, "Apply failed", f"Failed to set pixel/voxel size: {e!r}")

    # -----------------------------------------------------------------
    # C/T controls (0-based)
    # -----------------------------------------------------------------

    def _set_ct_controls(self, T: int, C: int):
        self.t_spin.blockSignals(True)
        self.t_spin.setRange(0, max(int(T) - 1, 0))
        self.t_spin.setValue(0)
        self.t_spin.setEnabled(int(T) > 1)
        self.t_spin.blockSignals(False)

        self.c_spin.blockSignals(True)
        self.c_spin.setRange(0, max(int(C) - 1, 0))
        self.c_spin.setValue(0)
        self.c_spin.setEnabled(int(C) > 1)
        self.c_spin.blockSignals(False)

    def _on_t_changed(self, v0: int):
        try:
            self.viewer.dims.set_current_step(0, max(int(v0), 0))  # 0-based
        except Exception:
            pass
        self._update_info()

    def _on_c_changed(self, v0: int):
        idx0 = max(int(v0), 0)  # 0-based
        for lyr in self._group_layers:
            md = getattr(lyr, "metadata", {}) or {}
            if md.get("channel_index") == idx0:
                try:
                    self.viewer.layers.selection.active = lyr
                except Exception:
                    pass
                break
        self._update_info()

    # -----------------------------------------------------------------
    # Denoise (median filter via button)
    # -----------------------------------------------------------------

    def _on_apply_denoise_clicked(self):
        if not self.chk_denoise.isChecked():
            QMessageBox.information(self, "Denoise disabled", "Check 'Denoise' first, then choose filter size.")
            return
        base_layer = self.viewer.layers.selection.active
        if base_layer is None or not hasattr(base_layer, "data"):
            QMessageBox.information(self, "No image", "Please select an image layer first.")
            return

        try:
            # choose volume/slice consistent with Run on 2d slice
            arr_full = np.asarray(base_layer.data)
            t_idx0 = int(self.viewer.dims.current_step[0]) if arr_full.ndim >= 4 else 0
            vol = np.squeeze(arr_full[t_idx0]) if arr_full.ndim >= 4 else arr_full
            k = int(self.cmb_kernel.currentText())

            if self.chk_use_2d.isChecked() and vol.ndim == 3:
                z_idx = int(np.clip(int(self.z_index_spin.value()), 0, vol.shape[0] - 1))
                img2d = vol[z_idx]
                out = medfilt(img2d, kernel_size=k)
                scale = (float(base_layer.scale[-2]), float(base_layer.scale[-1]))
                dims_out = "YX"
            elif vol.ndim == 3:
                out = medfilt(vol, kernel_size=(k, k, k))
                scale = (float(base_layer.scale[-3]), float(base_layer.scale[-2]), float(base_layer.scale[-1]))
                dims_out = "ZYX"
            elif vol.ndim == 2:
                out = medfilt(vol, kernel_size=k)
                scale = (float(base_layer.scale[-2]), float(base_layer.scale[-1]))
                dims_out = "YX"
            else:
                raise ValueError(f"Unsupported data ndim for denoise: {vol.ndim}")

            name = "denoised"
            existing = {l.name for l in self.viewer.layers}
            c = 1
            while name in existing:
                name = f"denoised_{c}"
                c += 1

            md = dict(getattr(base_layer, "metadata", {}) or {})
            md["is_denoise"] = True
            md["dims_out"] = dims_out
            md["group_id"] = _get_group_id(base_layer) or _ensure_group_id(md)

            if out.ndim == 3:
                self._denoise_layer = self.viewer.add_image(np.asarray(out), name=name, scale=scale, metadata=md)
            else:
                self._denoise_layer = self.viewer.add_image(np.asarray(out), name=name, scale=scale, metadata=md)

            # cache for Frangi/Seg
            self._denoise_ctx = dict(
                image=np.asarray(out).astype(np.float32, copy=False),
                dim=3 if out.ndim == 3 else 2,
                pixel_size=(scale if isinstance(scale, tuple) else tuple(scale)),
                unit=_get_units_from_layer(base_layer),
                source=_get_source_from_layer(base_layer),
                channel_index=(getattr(base_layer, "metadata", {}) or {}).get("channel_index"),
            )
            self.kw_preview.append("Denoise OK | median filter applied")
            self._update_segmentation_preview()
        except Exception as e:
            QMessageBox.critical(self, "Denoise failed", f"Median filter error: {e!r}")

    # -----------------------------------------------------------------
    # Frangi
    # -----------------------------------------------------------------

    def _make_sigma_list(self) -> Optional[List[float]]:
        smin = float(self.sigma_min.value())
        smax = float(self.sigma_max.value())
        cnt = int(self.sigma_count.value())
        if smin < 0.1:
            self.sigma_min.setValue(0.1)
            smin = 0.1
        if smax > 5.0:
            self.sigma_max.setValue(5.0)
            smax = 5.0
        if smax <= smin:
            QMessageBox.warning(self, "Sigma range invalid", "Ensure: max > min within [0.1, 5.0].")
            return None
        if cnt < 1:
            self.sigma_count.setValue(1)
            cnt = 1
        return np.linspace(smin, smax, cnt, dtype=float).tolist()

    def _extract_current_2d_or_3d(self, layer) -> Tuple[np.ndarray, int, float]:
        """Return (img, dim, zx_ratio). Prefer denoised data if available & enabled."""
        if self.chk_denoise.isChecked() and self._denoise_ctx.get("image") is not None:
            img = np.asarray(self._denoise_ctx["image"]).astype(np.float32, copy=False)
            dim = int(self._denoise_ctx.get("dim", 2))

            if dim == 3 and self.chk_use_2d.isChecked():
                z_idx = int(np.clip(int(self.z_index_spin.value()), 0, img.shape[0] - 1))
                img2d = img[z_idx]
                return img2d.astype(np.float32, copy=False), 2, 1.0

            if dim == 3:
                sc = getattr(layer, "scale", (1,) * layer.data.ndim)
                z = float(sc[-3]) if len(sc) >= 3 else 1.0
                x = float(sc[-1]) if len(sc) >= 1 else 1.0
                ratio = (z / x) if (z > 0 and x > 0) else 1.0
            else:
                ratio = 1.0
            return img, dim, float(ratio)

        # fallback to raw active layer
        arr = np.asarray(layer.data)
        try:
            t_idx0 = int(self.viewer.dims.current_step[0]) if arr.ndim >= 4 else 0
        except Exception:
            t_idx0 = 0
        vol = np.squeeze(arr[t_idx0]) if arr.ndim >= 4 else arr

        if vol.ndim == 3:
            sc = getattr(layer, "scale", (1,) * layer.data.ndim)
            z = float(sc[-3]) if len(sc) >= 3 else 1.0
            x = float(sc[-1]) if len(sc) >= 1 else 1.0
            ratio = (z / x) if (z > 0 and x > 0) else 1.0
            if self.chk_use_2d.isChecked():
                z_ui0 = int(self.z_index_spin.value())  # 0-based
                z_idx = int(np.clip(z_ui0, 0, vol.shape[0] - 1))
                img2d = vol[z_idx]
                return img2d.astype(np.float32, copy=False), 2, 1.0
            return vol.astype(np.float32, copy=False), 3, float(ratio)

        if vol.ndim == 2:
            return vol.astype(np.float32, copy=False), 2, 1.0

        vol = np.squeeze(vol)
        if vol.ndim in (2, 3):
            return self._extract_current_2d_or_3d(layer)
        raise ValueError(f"Unsupported data shape after slicing: {vol.shape}")

    def _on_apply_frangi_clicked(self):
        sigmas = self._make_sigma_list()
        if sigmas is None:
            return
        layer = self.viewer.layers.selection.active
        if layer is None or not hasattr(layer, "data"):
            QMessageBox.information(self, "No image", "Please select an image layer first.")
            return

        device = self._resolve_device()
        try:
            img, dim, ratio = self._extract_current_2d_or_3d(layer)
            rng = float(img.max()) - float(img.min())
            img = (img - img.min()) / (rng if rng > 0 else 1.0) * 255.0
        except Exception as e:
            QMessageBox.critical(self, "Slice error", f"Failed to get 2D/3D data: {e!r}")
            return

        try:
            if dim == 3:
                Fr = FrangiFilter(
                    channels=1,
                    kernel_size=2 * int(self.kernel_spin.value()) + 1,
                    sigmas=sigmas,
                    dim=3,
                    zx_ratio=ratio,
                    device=device,
                )
            else:
                Fr = FrangiFilter(
                    channels=1,
                    kernel_size=2 * int(self.kernel_spin.value()) + 1,
                    sigmas=sigmas,
                    dim=2,
                    device=device,
                )
            frangi_result_raw = Fr(-np.expand_dims(img, 0))[0].cpu().numpy()
        except Exception as e:
            QMessageBox.critical(self, "Frangi failed", f"FrangiFilter error: {e!r}")
            return

        base_md = dict(getattr(layer, "metadata", {}) or {})
        base_md["source"] = base_md.get("source", _get_source_from_layer(layer))
        base_md["unit"] = _get_units_from_layer(layer)
        base_md["is_frangi"] = True
        base_md["group_id"] = _get_group_id(self.viewer.layers.selection.active) or _ensure_group_id(base_md)

        sc = getattr(layer, "scale", (1,) * layer.data.ndim)
        if frangi_result_raw.ndim == 3:
            desired_scale = (float(sc[-3]), float(sc[-2]), float(sc[-1]))
            base_md["dims_out"] = "ZYX"
        else:
            try:
                if self.chk_use_2d.isChecked() and layer.data.ndim >= 3:
                    _z, _y, _x = _get_pixel_size_tuple(layer, 3)
                    desired_scale = (float(_y), float(_x))
                else:
                    desired_scale = (float(sc[-2]), float(sc[-1]))
            except Exception:
                desired_scale = (float(sc[-2]), float(sc[-1]))
            base_md["dims_out"] = "YX"

        # IMPORTANT: avoid "TCZYX" to prevent a T slider for frangi
        base_md.pop("dims", None)

        name = "frangi"
        existing = {l.name for l in self.viewer.layers}
        k = 1
        while name in existing:
            k += 1
            name = f"frangi_{k}"

        self._frangi_layer = self.viewer.add_image(
            np.asarray(frangi_result_raw),
            name=name,
            scale=desired_scale,
            metadata=base_md,
        )

        self._last_frangi_ctx = dict(
            image=img,
            frangi=frangi_result_raw,
            frangi_raw=frangi_result_raw,
            dim=dim,
            pixel_size=_get_pixel_size_tuple(layer, dim),
            device=device,
            source=base_md.get("source"),
            channel_index=(getattr(layer, "metadata", {}) or {}).get("channel_index"),
            unit=base_md.get("unit", "um"),
        )

        self.chk_rescale.setEnabled(True)
        self._apply_low_high_binding(self._low_percent)
        self.range_slider.setEnabled(self.chk_rescale.isChecked())
        if self.chk_rescale.isChecked():
            self._recompute_rescaled_frangi()

        self._update_segmentation_preview()

        dimtxt = "2D (slice)" if (dim == 2 and self.chk_use_2d.isChecked()) else ("3D" if dim == 3 else "2D")
        self.kw_preview.append(f"Frangi OK | {dimtxt} | sigmas={np.array(sigmas)} | device={device}")

    # -- Frangi rescale UI bindings ----------------------------------

    def _on_toggle_rescale(self, v):
        enabled = bool(v) and (self._last_frangi_ctx.get("frangi_raw") is not None)
        self.range_slider.setEnabled(enabled)
        if enabled:
            self._apply_low_high_binding(self._low_percent)
            self._recompute_rescaled_frangi()
        else:
            if self._last_frangi_ctx.get("frangi_raw") is not None and self._frangi_layer is not None:
                raw = self._last_frangi_ctx["frangi_raw"]
                self._frangi_layer.data = np.asarray(raw)
                self._last_frangi_ctx["frangi"] = np.asarray(raw)
                self._update_segmentation_preview()

    def _apply_low_high_binding(self, low: int):
        low = max(0, min(int(low), 20))
        high = 100 - low
        if high < 80:
            high = 80
            low = 20
        self.range_slider.blockSignals(True)
        self.range_slider.setValue((low, high))
        self.range_slider.blockSignals(False)
        self._low_percent = low
        self.range_lbl.setText(f"percentiles: {low}%–{high}%")

    def _on_rescale_params_changed(self, *args):
        try:
            a, b = self.range_slider.value()
        except Exception:
            v = self.range_slider.value()
            a, b = (v[0], v[1]) if isinstance(v, (tuple, list)) else (0, 100)

        proposed_low_from_low = a
        proposed_low_from_high = 100 - b
        new_low = (
            proposed_low_from_low
            if abs(proposed_low_from_low - self._low_percent) >= abs(proposed_low_from_high - self._low_percent)
            else proposed_low_from_high
        )
        self._apply_low_high_binding(new_low)
        if self.chk_rescale.isChecked():
            self._recompute_rescaled_frangi()

    def _recompute_rescaled_frangi(self):
        ctx = self._last_frangi_ctx
        if not ctx or ctx.get("frangi_raw") is None or self._frangi_layer is None:
            return
        raw = np.asarray(ctx["frangi_raw"])
        low, high = self.range_slider.value()
        lo = float(np.percentile(raw, int(low)))
        hi = float(np.percentile(raw, int(high)))
        if hi <= lo:
            hi = lo + 1e-6
        rescaled = rescale_intensity(raw, in_range=(lo, hi))
        try:
            self._frangi_layer.data = np.asarray(rescaled)
        except Exception:
            md = dict(getattr(self._frangi_layer, "metadata", {}) or {})
            name = self._frangi_layer.name
            sc = tuple(getattr(self._frangi_layer, "scale", (1,) * rescaled.ndim))
            try:
                self.viewer.layers.remove(self._frangi_layer)
            except Exception:
                pass
            self._frangi_layer = self.viewer.add_image(np.asarray(rescaled), name=name, scale=sc, metadata=md)
        ctx["frangi"] = np.asarray(rescaled)
        self._update_segmentation_preview()

    # -----------------------------------------------------------------
    # Segmentation
    # -----------------------------------------------------------------

    def _build_segmentation_kwargs(self) -> Optional[Dict[str, Any]]:
        # prefer denoised image as the base if present & enabled
        base_image = None
        base_dim = None
        pixel_size = None

        if self.chk_denoise.isChecked() and self._denoise_ctx.get("image") is not None:
            base_image = np.asarray(self._denoise_ctx["image"]).astype(np.float32, copy=False)
            base_dim = int(self._denoise_ctx.get("dim", 2))
            pixel_size = tuple(self._denoise_ctx.get("pixel_size", ()))
        elif self._last_frangi_ctx:
            base_image = np.asarray(self._last_frangi_ctx["image"]).astype(np.float32, copy=False)
            base_dim = int(self._last_frangi_ctx.get("dim", 2))
            pixel_size = tuple(self._last_frangi_ctx.get("pixel_size", ()))

        if base_image is None or not self._last_frangi_ctx:
            return None

        layer = self.viewer.layers.selection.active
        try:
            fresh_px = _get_pixel_size_tuple(layer, base_dim)
        except Exception:
            fresh_px = pixel_size

        return dict(
            image=base_image,
            frangi=np.asarray(self._last_frangi_ctx["frangi"]).astype(np.float32, copy=False),
            pixel_size=fresh_px,
            beta1=float(self.beta1_spin.value()),
            beta2=float(self.beta2_spin.value()),
            cutoff=float(self.cutoff_spin.value()),
            n_fore=int(self.nfore_spin.value()),
            n_back=int(self.nback_spin.value()),
            max_iter=int(self.maxiter_spin.value()),
            device=self._resolve_device(),
        )

    def _update_segmentation_preview(self):
        kwargs = self._build_segmentation_kwargs()
        if kwargs is None:
            self.kw_preview.setPlainText("Run Frangi first to preview segmentation() call.")
            return
        low, high = self.range_slider.value()
        lines = [
            "segmentation() will be called with:",
            f"  image: float32 array, shape={np.asarray(kwargs['image']).shape}",
            f"  frangi: float32 array, shape={np.asarray(kwargs['frangi']).shape}",
            f"  pixel_size: {kwargs['pixel_size']} ({'z,y,x' if len(kwargs['pixel_size'])==3 else 'y,x'})",
            f"  beta1={kwargs['beta1']}  beta2={kwargs['beta2']}  cutoff={kwargs['cutoff']}",
            f"  n_fore={kwargs['n_fore']}  n_back={kwargs['n_back']}  max_iter={kwargs['max_iter']}",
            f"  device='{kwargs['device']}'",
            "Notes:",
            "  • If 'Run on 2d slice' is enabled, 'image' may be 2D and pixel_size is (y,x).",
            f"  • Intensity rescale: {'ON' if self.chk_rescale.isChecked() else 'OFF'}; percentiles={int(low)}%–{int(high)}% (high = 100 - low)",
            f"  • Denoise: {'ON' if (self.chk_denoise.isChecked() and self._denoise_ctx.get('image') is not None) else 'OFF'}; filter size={self.cmb_kernel.currentText()}",
        ]
        self.kw_preview.setPlainText("\n".join(lines))

    def _on_run_segmentation_clicked(self):
        if not self._last_frangi_ctx:
            QMessageBox.information(self, "Run Frangi first", "Please run Frangi before segmentation.")
            return

        kwargs = self._build_segmentation_kwargs()
        if kwargs is None:
            return

        try:
            self.viewer.text_overlay.visible = True
            self.viewer.text_overlay.position = "bottom_left"
            self.viewer.text_overlay.text = "Segmentation starting..."
        except Exception:
            pass

        self.seg_progress.setRange(0, 0)
        self.seg_progress.setFormat("Running segmentation…")
        self.seg_btn.setEnabled(False)

        self._seg_thread = QThread()
        self._seg_worker = SegWorker(kwargs)
        self._seg_worker.moveToThread(self._seg_thread)
        self._seg_thread.started.connect(self._seg_worker.run)
        self._seg_worker.started.connect(lambda: None)
        self._seg_worker.progress.connect(self._on_seg_progress)

        def _finished(payload, error):
            self.seg_progress.setRange(0, 100)
            if error is not None:
                self.seg_progress.setValue(0)
                QMessageBox.critical(self, "Segmentation failed", f"segmentation() error: {error!r}")
            else:
                self.seg_progress.setValue(100)
                seg, info = payload
                self._add_segmentation_result(seg, info)
            self.seg_btn.setEnabled(True)
            self._seg_thread.quit()
            self._seg_thread.wait()
            self._seg_worker.deleteLater()
            self._seg_thread.deleteLater()

        self._seg_worker.finished.connect(_finished)
        self._seg_thread.start()

    def _on_seg_progress(self, iteration: int, delta: float):
        try:
            self.viewer.text_overlay.visible = True
            self.viewer.text_overlay.position = "bottom_left"
            if np.isfinite(delta):
                self.viewer.text_overlay.text = f"Segmentation iter: {iteration} | ΔlogL: {delta:.3e}"
            else:
                self.viewer.text_overlay.text = f"Segmentation iter: {iteration} | ΔlogL: --"
        except Exception:
            pass

        try:
            base = self.kw_preview.toPlainText() if hasattr(self.kw_preview, "toPlainText") else ""
            head = f"[iter={iteration}] ΔlogL={delta:.3e}" if np.isfinite(delta) else f"[iter={iteration}] ΔlogL=--"
            lines = [head] + base.splitlines()
            self.kw_preview.setPlainText("\n".join(lines[:30]))
        except Exception:
            pass

    def _add_segmentation_result(self, seg, info=None):
        try:
            layer = self.viewer.layers.selection.active
            md_base = dict(getattr(layer, "metadata", {}) or {}) if layer else {}

            seg_np = np.asarray(seg)
            sc_layer = getattr(layer, "scale", None) if layer else None
            add_kwargs = {}
            if sc_layer is not None:
                if seg_np.ndim == 3:
                    add_kwargs["scale"] = (sc_layer[-3], sc_layer[-2], sc_layer[-1])
                elif seg_np.ndim == 2:
                    add_kwargs["scale"] = (sc_layer[-2], sc_layer[-1])

            name = "segmentation"
            existing = {l.name for l in self.viewer.layers}
            k = 1
            while name in existing:
                k += 1
                name = f"segmentation_{k}"

            mask8 = ((seg_np.astype(np.int64) > 0).astype(np.uint8) * 255)
            mask8 = np.ascontiguousarray(mask8)

            new_layer = self.viewer.add_image(
                mask8,
                name=name,
                rgb=False,
                blending="translucent_no_depth",
                **add_kwargs,
            )
            try:
                new_layer.contrast_limits = (0, 255)
            except Exception:
                pass

            try:
                md_base["dims_out"] = "ZYX" if mask8.ndim == 3 else "YX"
            except Exception:
                pass

            ctx = self._last_frangi_ctx
            new_md = dict(md_base)
            new_md["unit"] = ctx.get("unit", _get_units_from_layer(layer) if layer else "um")
            new_md["source"] = ctx.get("source", _get_source_from_layer(layer) if layer else None)
            new_md["is_segmentation"] = True
            new_md["group_id"] = _get_group_id(self.viewer.layers.selection.active) or _ensure_group_id(new_md)

            if info is not None:
                try:
                    new_md["seg_iterations_run"] = int(getattr(info, "iterations_run", 0))
                    new_md["seg_delta_loglh_trace"] = list(getattr(info, "deltas", []))
                    new_md["seg_converged"] = bool(getattr(info, "converged", False))
                except Exception:
                    pass

            new_layer.metadata = new_md

            try:
                self.viewer.scale_bar.visible = True
                self.viewer.scale_bar.unit = new_md["unit"]
            except Exception:
                pass

            self.kw_preview.append(
                "Segmentation OK | "
                f"dim={ctx['dim']} | "
                f"β1={float(self.beta1_spin.value())} β2={float(self.beta2_spin.value())} "
                f"cutoff={float(self.cutoff_spin.value())} nfore={int(self.nfore_spin.value())} "
                f"nback={int(self.nback_spin.value())} maxiter={int(self.maxiter_spin.value())} | "
                f"device={ctx['device']}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Add result failed", f"Failed to add segmentation result: {e!r}")


# ---------------------------------------------------------------------
# napari widget factory / optional reader
# ---------------------------------------------------------------------

def create_segment_confocal_widget(viewer):
    return SegmentConfocalWidget(viewer)


def napari_experimental_provide_dock_widget():
    return [SegmentConfocalWidget]


def napari_get_reader(paths):
    if isinstance(paths, (list, tuple)):
        if len(paths) != 1:
            return None
        path = paths[0]
    else:
        path = paths
    try:
        _ = AICSImage(path)
    except Exception:
        return None

    def _reader(_paths):
        p = _paths[0] if isinstance(_paths, (list, tuple)) else _paths
        data, meta = _load_image_tc_zyx(p)
        data = _rescale_0_255_tczyx(data)
        add_kwargs = dict(
            name=os.path.basename(p),
            channel_axis=1,
            rgb=False,
            blending="additive",
            metadata=dict(meta),
        )
        return [(data, add_kwargs, "image")]

    return _reader


if __name__ == "__main__":  # pragma: no cover
    try:
        import napari
        v = napari.Viewer()
        w = SegmentConfocalWidget(v)
        v.window.add_dock_widget(w, area="right")
        napari.run()
    except Exception as e:
        import traceback
        print("Failed to launch napari demo:", e)
        traceback.print_exc()
