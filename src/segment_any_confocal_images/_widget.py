from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
import os
import numpy as np
from uuid import uuid4
from aicsimageio import AICSImage

from magicgui import magicgui
from qtpy.QtWidgets import (
    QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QMessageBox,
    QHBoxLayout, QSpinBox, QDoubleSpinBox, QComboBox, QFormLayout, QGroupBox, QGridLayout,
    QCheckBox, QTextEdit, QProgressBar, QLineEdit, QSizePolicy
)
from qtpy.QtCore import QTimer, Qt, QThread, Signal, QObject, QLocale

# external Frangi
from frangi_filter.frangi_filter import *

# external segmentation entry
from SegmentAnyConfocal.segmentation_ import * 

# torch is optional
try:
    import torch
except Exception:
    torch = None


# ----------------------- helpers -----------------------

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
    nd = layer.data.ndim
    dims = (getattr(layer, "metadata", {}) or {}).get("dims")
    labels = ("T", "Z", "Y", "X") if (dims == "TCZYX" or nd >= 4) else \
             ("Z", "Y", "X") if nd == 3 else \
             ("Y", "X") if nd == 2 else tuple(str(i) for i in range(nd))
    try:
        viewer.dims.axis_labels = labels
    except Exception:
        pass

def _describe_size_and_resolution(layer) -> Dict[str, Any]:
    """Return dict describing ndim/shape, voxel/pixel size (from layer.scale),
    and physical size computed as n * scale (avoid shrink from extent.world)."""
    data = np.asarray(layer.data)
    result: Dict[str, Any] = {}
    sc = tuple(getattr(layer, "scale", (1,) * data.ndim))
    unit = _get_units_from_layer(layer)
    if data.ndim >= 3:
        nz, ny, nx = data.shape[-3], data.shape[-2], data.shape[-1]
        vz, vy, vx = float(sc[-3]), float(sc[-2]), float(sc[-1])
        phys = (nz * vz, ny * vy, nx * vx)
        result.update(dict(
            ndim=3, shape=(nz, ny, nx),
            voxel=(vz, vy, vx),
            phys=(float(phys[0]), float(phys[1]), float(phys[2])),
            unit=unit,
        ))
    elif data.ndim == 2:
        ny, nx = data.shape[-2], data.shape[-1]
        vy, vx = float(sc[-2]), float(sc[-1])
        phys = (ny * vy, nx * vx)
        result.update(dict(
            ndim=2, shape=(ny, nx),
            voxel=(vy, vx),  # pixel in UI
            phys=(float(phys[0]), float(phys[1])),
            unit=unit,
        ))
    else:
        result.update(dict(ndim=data.ndim, shape=tuple(data.shape)))
    return result

def _try_autofill_scale_from_source(layer) -> bool:
    src = _get_source_from_layer(layer)
    if not src:
        return False
    try:
        from aicsimageio import AICSImage
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

def _load_image_tc_zyx(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    img = AICSImage(path)
    data = img.get_image_data("TCZYX")
    pps = img.physical_pixel_sizes
    zyx_scale = [float(pps.Z) if pps.Z else 1.0,
                 float(pps.Y) if pps.Y else 1.0,
                 float(pps.X) if pps.X else 1.0]
    ch_names = None
    try:
        ch_names = [
            c.name if getattr(c, "name", None) else f"Channel {i+1}"
            for i, c in enumerate(img.metadata.images[0].pixels.channels)
        ]
    except Exception:
        pass
    meta: Dict[str, Any] = dict(
        dims="TCZYX", channel_names=ch_names, unit="um",
        scale_per_axis=(1.0, 1.0, *zyx_scale), source=path,
    )
    _ensure_group_id(meta)
    return data, meta

def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else ([x] if x is not None else [])

def _get_pixel_size_tuple(layer, ndim: int):
    sc = getattr(layer, "scale", (1,) * layer.data.ndim)
    if ndim == 3:
        return (float(sc[-3]), float(sc[-2]), float(sc[-1]))
    elif ndim == 2:
        return (float(sc[-2]), float(sc[-1]))
    else:
        raise ValueError(f"Unsupported ndim for pixel size: {ndim}")


# ----------------------- worker for segmentation (progress) -----------------------

class SegWorker(QObject):
    finished = Signal(object, object)  # (result_or_None, error_or_None)
    started = Signal()

    def __init__(self, kwargs: Dict[str, Any]):
        super().__init__()
        self.kwargs = kwargs

    def run(self):
        self.started.emit()
        try:
            seg = segmentation(**self.kwargs)
            seg = seg[0].cpu().numpy()
            self.finished.emit(seg, None)
        except Exception as e:
            self.finished.emit(None, e)


# ----------------------- main widget -----------------------

class SegmentConfocalWidget(QWidget):
    """
    Napari widget with TC support, pixel/voxel-size editor, optional 2D-slice processing,
    progress for segmentation, and external segmentation().
    """

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._group_layers: List = []
        self._last_frangi_ctx: Dict[str, Any] = {}

        # ---- overall layout ----
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        self.setMinimumWidth(420)
        self.setMaximumWidth(520)

        # --- Data / Axes / Device ---
        top_box = QGroupBox("Data / Axes / Device")
        top_form = QGridLayout()
        top_form.setHorizontalSpacing(12)
        top_form.setVerticalSpacing(6)

        # Select File + Device
        self.open_btn = QPushButton("Select File")
        self.open_btn.clicked.connect(self._open_image_tc_zyx)
        self.device_combo = QComboBox(); self._populate_devices()
        top_form.addWidget(self.open_btn, 0, 0)
        top_form.addWidget(QLabel("Device:"), 0, 1, alignment=Qt.AlignRight)
        top_form.addWidget(self.device_combo, 0, 2)

        # Path
        self.path_edit = QLineEdit(); self.path_edit.setReadOnly(True)
        top_form.addWidget(QLabel("Path:"), 1, 0, alignment=Qt.AlignRight)
        top_form.addWidget(self.path_edit, 1, 1, 1, 2)

        # Channel + Time
        self.c_spin = QSpinBox(); self.c_spin.setRange(1, 1); self.c_spin.setEnabled(False); self.c_spin.setValue(1)
        self.t_spin = QSpinBox(); self.t_spin.setRange(1, 1); self.t_spin.setEnabled(False); self.t_spin.setValue(1)
        self.c_spin.valueChanged.connect(self._on_c_changed)
        self.t_spin.valueChanged.connect(self._on_t_changed)
        row_ct = QWidget(); row_ct_layout = QHBoxLayout(); row_ct_layout.setContentsMargins(0,0,0,0)
        row_ct_layout.addWidget(QLabel("Channel:")); row_ct_layout.addWidget(self.c_spin)
        row_ct_layout.addSpacing(12)
        row_ct_layout.addWidget(QLabel("Time:")); row_ct_layout.addWidget(self.t_spin)
        row_ct.setLayout(row_ct_layout)
        top_form.addWidget(row_ct, 2, 0, 1, 3)

        top_box.setLayout(top_form)
        layout.addWidget(top_box)

        # === Image Info ===
        info_box = QGroupBox("Image Info")
        info_grid = QGridLayout(); info_grid.setHorizontalSpacing(12); info_grid.setVerticalSpacing(6)

        self.lbl_shape = QLabel("–"); self.lbl_phys = QLabel("–"); self.lbl_voxpix = QLabel("–")
        for w in (self.lbl_shape, self.lbl_phys, self.lbl_voxpix):
            w.setTextInteractionFlags(w.textInteractionFlags() | 0x02000000)
        info_grid.addWidget(QLabel("Shape:"), 0, 0, alignment=Qt.AlignRight)
        info_grid.addWidget(self.lbl_shape, 0, 1)
        info_grid.addWidget(QLabel("Physical size:"), 1, 0, alignment=Qt.AlignRight)
        info_grid.addWidget(self.lbl_phys, 1, 1)
        info_grid.addWidget(QLabel("Voxel size:"), 2, 0, alignment=Qt.AlignRight)
        info_grid.addWidget(self.lbl_voxpix, 2, 1)

        # editable voxel/pixel (columns Z/Y/X or Y/X)
        self.edit_vz = QDoubleSpinBox(); self.edit_vz.setDecimals(6); self.edit_vz.setRange(0, 1e3); self.edit_vz.setLocale(QLocale.c())
        self.edit_vyx = QDoubleSpinBox(); self.edit_vyx.setDecimals(6); self.edit_vyx.setRange(0, 1e3); self.edit_vyx.setLocale(QLocale.c())
        self.btn_apply_voxel = QPushButton("Update Voxel/Pixel size")

        vox_row = 3
        info_grid.addWidget(QLabel("Edit size:"), vox_row, 0, alignment=Qt.AlignRight)
        col_widget = QWidget(); col_layout = QGridLayout(); col_layout.setContentsMargins(0,0,0,0); col_layout.setHorizontalSpacing(8)
        # row 0: Z, Y, X
        self.lblZ = QLabel("Z:"); self.lblYX = QLabel("Y/X:")#; self.lblX = QLabel("X:")
        col_layout.addWidget(self.lblZ, 0, 0, alignment=Qt.AlignRight); col_layout.addWidget(self.edit_vz, 0, 1)
        col_layout.addWidget(self.lblYX, 0, 2, alignment=Qt.AlignRight); col_layout.addWidget(self.edit_vyx, 0, 3)
        #col_layout.addWidget(self.lblX, 0, 4, alignment=Qt.AlignRight); col_layout.addWidget(self.edit_vx, 0, 5)
        col_widget.setLayout(col_layout)
        info_grid.addWidget(col_widget, vox_row, 1)
        info_grid.addWidget(self.btn_apply_voxel, vox_row + 1, 1)

        self.btn_apply_voxel.clicked.connect(self._on_apply_voxel_clicked)

        info_box.setLayout(info_grid)
        layout.addWidget(info_box)

        # Scale bar defaults
        try:
            self.viewer.scale_bar.visible = True
            self.viewer.scale_bar.unit = "um"
        except Exception:
            pass

        # --- Frangi Params ---
        frangi_box = QGroupBox("Frangi Filter")
        fr_grid = QGridLayout(); fr_grid.setHorizontalSpacing(12); fr_grid.setVerticalSpacing(6)

        # kernel radius & sigma count one row
        self.kernel_spin = QSpinBox(); self.kernel_spin.setRange(1, 5); self.kernel_spin.setValue(4)
        self.sigma_count = QSpinBox(); self.sigma_count.setRange(1, 99); self.sigma_count.setValue(5)
        fr_grid.addWidget(QLabel("Kernel radius:"), 0, 0, alignment=Qt.AlignRight)
        fr_grid.addWidget(self.kernel_spin, 0, 1)
        fr_grid.addWidget(QLabel("Sigma count:"), 0, 2, alignment=Qt.AlignRight)
        fr_grid.addWidget(self.sigma_count, 0, 3)

        # sigma min & max one row
        self.sigma_min = QDoubleSpinBox(); self.sigma_min.setDecimals(2); self.sigma_min.setRange(0.1, 5.0); self.sigma_min.setSingleStep(0.1); self.sigma_min.setValue(0.1); self.sigma_min.setLocale(QLocale.c())
        self.sigma_max = QDoubleSpinBox(); self.sigma_max.setDecimals(2); self.sigma_max.setRange(0.1, 5.0); self.sigma_max.setSingleStep(0.1); self.sigma_max.setValue(1.0); self.sigma_max.setLocale(QLocale.c())
        fr_grid.addWidget(QLabel("Sigma min:"), 1, 0, alignment=Qt.AlignRight)
        fr_grid.addWidget(self.sigma_min, 1, 1)
        fr_grid.addWidget(QLabel("Sigma max:"), 1, 2, alignment=Qt.AlignRight)
        fr_grid.addWidget(self.sigma_max, 1, 3)

        # 2D slice option
        self.chk_use_2d = QCheckBox("Run on 2d slice")
        self.z_index_spin = QSpinBox(); self.z_index_spin.setRange(0, 0); self.z_index_spin.setValue(0); self.z_index_spin.setEnabled(False)
        def _toggle_slice(v):
            self.z_index_spin.setEnabled(bool(v))
            self._sync_z_spin_to_view()
        self.chk_use_2d.toggled.connect(_toggle_slice)
        row2d = QWidget(); row2d_layout = QHBoxLayout(); row2d_layout.setContentsMargins(0,0,0,0)
        row2d_layout.addWidget(self.chk_use_2d); row2d_layout.addWidget(QLabel("Frame Index:")); row2d_layout.addWidget(self.z_index_spin); row2d.setLayout(row2d_layout)
        fr_grid.addWidget(row2d, 2, 0, 1, 4)

        self.apply_btn = QPushButton("Run Frangi Filter")
        self.apply_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.apply_btn.setMinimumHeight(32)
        self.apply_btn.clicked.connect(self._on_apply_frangi_clicked)
        fr_grid.addWidget(self.apply_btn, 3, 0, 1, 4)

        frangi_box.setLayout(fr_grid)
        layout.addWidget(frangi_box)

        # --- Segmentation ---
        seg_box = QGroupBox("Segmentation")
        seg_grid = QGridLayout(); seg_grid.setHorizontalSpacing(16); seg_grid.setVerticalSpacing(8)

        # 2 decimals as requested
        self.beta1_spin = QDoubleSpinBox(); self.beta1_spin.setDecimals(2); self.beta1_spin.setRange(0.0, 10); self.beta1_spin.setSingleStep(0.01); self.beta1_spin.setValue(1.0); self.beta1_spin.setLocale(QLocale.c())
        self.beta2_spin = QDoubleSpinBox(); self.beta2_spin.setDecimals(2); self.beta2_spin.setRange(0.0, 10); self.beta2_spin.setSingleStep(0.01); self.beta2_spin.setValue(1.0); self.beta2_spin.setLocale(QLocale.c())
        self.cutoff_spin = QDoubleSpinBox(); self.cutoff_spin.setDecimals(2); self.cutoff_spin.setRange(0, 10); self.cutoff_spin.setSingleStep(0.01); self.cutoff_spin.setValue(2.0); self.cutoff_spin.setLocale(QLocale.c())
        self.nfore_spin = QSpinBox(); self.nfore_spin.setRange(0, 16); self.nfore_spin.setValue(8)
        self.nback_spin = QSpinBox(); self.nback_spin.setRange(0, 16); self.nback_spin.setValue(3)
        self.maxiter_spin = QSpinBox(); self.maxiter_spin.setRange(1, 200); self.maxiter_spin.setValue(50)

        seg_grid.addWidget(QLabel("beta1:"), 0, 0, alignment=Qt.AlignRight)
        seg_grid.addWidget(self.beta1_spin, 0, 1)
        seg_grid.addWidget(QLabel("beta2:"), 0, 2, alignment=Qt.AlignRight)
        seg_grid.addWidget(self.beta2_spin, 0, 3)
        seg_grid.addWidget(QLabel("cutoff:"), 1, 0, alignment=Qt.AlignRight)
        seg_grid.addWidget(self.cutoff_spin, 1, 1)
        seg_grid.addWidget(QLabel("maxiter:"), 1, 2, alignment=Qt.AlignRight)
        seg_grid.addWidget(self.maxiter_spin, 1, 3)
        seg_grid.addWidget(QLabel("nforeground:"), 2, 0, alignment=Qt.AlignRight)
        seg_grid.addWidget(self.nfore_spin, 2, 1)
        seg_grid.addWidget(QLabel("nbackground:"), 2, 2, alignment=Qt.AlignRight)
        seg_grid.addWidget(self.nback_spin, 2, 3)

        self.seg_progress = QProgressBar(); self.seg_progress.setValue(0); self.seg_progress.setTextVisible(True)
        seg_grid.addWidget(self.seg_progress, 3, 0, 1, 4)

        self.kw_preview = QTextEdit(); self.kw_preview.setReadOnly(True); self.kw_preview.setMinimumHeight(90)
        seg_grid.addWidget(self.kw_preview, 4, 0, 1, 4)

        self.seg_btn = QPushButton("Run segmentation")
        self.seg_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.seg_btn.setMinimumHeight(32)
        self.seg_btn.clicked.connect(self._on_run_segmentation_clicked)
        seg_grid.addWidget(self.seg_btn, 5, 0, 1, 4)

        seg_box.setLayout(seg_grid)
        layout.addWidget(seg_box)

        # Refresh
        self.refresh_btn = QPushButton("Refresh info")
        self.refresh_btn.clicked.connect(self._update_info)
        layout.addWidget(self.refresh_btn, alignment=Qt.AlignLeft)

        # listeners
        self.viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self.viewer.layers.events.removed.connect(self._update_info)
        self.viewer.layers.selection.events.active.connect(self._on_active_changed)
        try:
            self.viewer.dims.events.current_step.connect(self._on_dims_step_changed)
        except Exception:
            pass

        self._update_info()
        self._fit_view_and_scalebar()


    # ----------------------- device helpers -----------------------

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

    def _resolve_device(self) -> str:
        choice = self.device_combo.currentText().lower()
        if torch is None:
            return "cpu"
        if choice == "cuda" and torch.cuda.is_available():
            return "cuda"
        if choice == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if choice in ("cuda", "mps"):
            QMessageBox.information(self, "Device fallback", f"{choice} not available. Falling back to CPU.")
        return "cpu"

    # ----------------------- open & normalize -----------------------

    def _open_image_tc_zyx(self):
        dlg = QFileDialog(self, "Open image")
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter("Images (*.tif *.tiff *.czi *.nd2 *.lif *.lsm *.ome.tif *.ome.tiff *.png *.jpg *.jpeg);;All files (*)")
        if not dlg.exec_():
            return
        path = dlg.selectedFiles()[0]

        try:
            data, meta = _load_image_tc_zyx(path)  # (T,C,Z,Y,X)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", f"Failed to load: {e!r}")
            return

        self.path_edit.setText(path)

        added = self.viewer.add_image(
            data, name=os.path.basename(path), channel_axis=1, rgb=False,
            blending="additive", visible=True, metadata=dict(meta),
        )
        layers = _as_list(added)
        self._group_layers = layers

        sc5 = tuple(meta.get("scale_per_axis", (1, 1, 1, 1, 1)))
        unit = _ensure_unit_in_metadata(layers[0].metadata) if layers else "um"
        ch_names = meta.get("channel_names") or [f"Channel {i+1}" for i in range(len(layers))]
        gid = meta.get("group_id", uuid4().hex)

        for i, lyr in enumerate(layers):
            sc = (1.0, sc5[-3], sc5[-2], sc5[-1]) if lyr.data.ndim == 4 else (sc5[-2], sc5[-1])
            try:
                lyr.scale = sc
            except Exception:
                lyr.scale = (1.0,) * (lyr.data.ndim - 3) + (sc5[-3], sc5[-2], sc5[-1])
            md = dict(getattr(lyr, "metadata", {}) or {})
            md.setdefault("unit", unit); md.setdefault("dims", "TCZYX"); md.setdefault("source", path)
            md["group_id"] = gid
            md["channel_index"] = i
            md["channel_name"] = ch_names[i] if i < len(ch_names) else f"Channel {i+1}"
            lyr.metadata = md

        self._set_ct_controls(T=int(data.shape[0]), C=int(data.shape[1]))
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

        # set slice index range
        try:
            if data.shape[2] > 0:
                self.z_index_spin.setRange(0, int(data.shape[2] - 1))
        except Exception:
            pass

        self._fit_view_and_scalebar()
        self._update_info()

    def _on_layer_inserted(self, event=None):
        layer = getattr(event, "value", None)
        if layer is None:
            return
        def _deferred():
            try:
                self._maybe_normalize_dragdrop_layer(layer)
            except Exception:
                pass
            self._update_info()
        QTimer.singleShot(0, _deferred)

    def _maybe_normalize_dragdrop_layer(self, layer):
        # do not normalize derived layers (frangi/segmentation)
        md0 = getattr(layer, 'metadata', {}) or {}
        if md0.get('is_frangi') or md0.get('is_segmentation'):
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
            # unify group_id across same-source layers
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
                    self.z_index_spin.setRange(0, int(layer.data.shape[-3] - 1))
                else:
                    self.z_index_spin.setRange(0, 0)
            except Exception:
                pass
            return
        # try reload via AICS
        try:
            data, meta = _load_image_tc_zyx(src)
            if data.shape[0] == 1 and data.shape[1] == 1 and layer.data.ndim <= 3:
                raise RuntimeError("No need to replace")
            base = os.path.basename(src); name = base
            existing = {l.name for l in self.viewer.layers}; k = 1
            while name in existing:
                k += 1; name = f"{base} ({k})"
            added = self.viewer.add_image(
                data, name=name, channel_axis=1, rgb=False, blending="additive",
                visible=True, metadata=dict(meta)
            )
            new_layers = _as_list(added); self._group_layers = new_layers
            sc5 = tuple(meta.get("scale_per_axis", (1, 1, 1, 1, 1)))
            unit = meta.get("unit", "um")
            ch_names = meta.get("channel_names") or [f"Channel {i+1}" for i in range(len(new_layers))]
            gid = meta.get("group_id", uuid4().hex)
            for i, lyr in enumerate(new_layers):
                sc = (1.0, sc5[-3], sc5[-2], sc5[-1]) if lyr.data.ndim == 4 else (sc5[-2], sc5[-1])
                try: lyr.scale = sc
                except Exception: lyr.scale = (1.0,) * (lyr.data.ndim - 3) + (sc5[-3], sc5[-2], sc5[-1])
                md = dict(getattr(lyr, "metadata", {}) or {})
                md.setdefault("unit", unit); md.setdefault("dims", "TCZYX"); md.setdefault("source", src)
                md["group_id"] = gid
                md["channel_index"] = i; md["channel_name"] = ch_names[i] if i < len(ch_names) else f"Channel {i+1}"
                lyr.metadata = md
            if new_layers:
                self.viewer.layers.selection.active = new_layers[0]
                _safe_axis_labels(self.viewer, new_layers[0])
            try: self.viewer.layers.remove(layer)
            except Exception: pass
            self._set_ct_controls(T=int(data.shape[0]), C=int(data.shape[1]))
            try:
                self.z_index_spin.setRange(0, int(data.shape[2] - 1))
            except Exception:
                pass
            self._fit_view_and_scalebar(); self._update_info(); return
        except Exception:
            _try_autofill_scale_from_source(layer)
            try:
                md = layer.metadata if layer.metadata else {}; md["unit"] = _ensure_unit_in_metadata(md); _ensure_group_id(md); layer.metadata = md
            except Exception: pass
            try:
                self.viewer.scale_bar.visible = True; self.viewer.scale_bar.unit = _get_units_from_layer(layer)
            except Exception: pass
            _safe_axis_labels(self.viewer, layer)
            T = int(layer.data.shape[0]) if layer.data.ndim >= 4 else 1
            self._group_layers = [layer]; self._set_ct_controls(T=T, C=1); self._update_info()
            try:
                if layer.data.ndim >= 3:
                    self.z_index_spin.setRange(0, int(layer.data.shape[-3] - 1))
                else:
                    self.z_index_spin.setRange(0, 0)
            except Exception:
                pass

    # ----------------------- events -----------------------

    def _on_active_changed(self, event=None):
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
                self.c_spin.setValue(int(md["channel_index"]) + 1)
                self.c_spin.blockSignals(False)
            try:
                t_cur0 = int(self.viewer.dims.current_step[0])
                if self.t_spin.isEnabled():
                    self.t_spin.blockSignals(True)
                    self.t_spin.setValue(t_cur0 + 1)
                    self.t_spin.blockSignals(False)
            except Exception:
                pass

        self._update_info()
        self._fit_view_and_scalebar()

    def _on_dims_step_changed(self, event=None):
        try:
            t_cur0 = int(self.viewer.dims.current_step[0])
            if self.t_spin.isEnabled():
                self.t_spin.blockSignals(True)
                self.t_spin.setValue(t_cur0 + 1)
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
        if arr.ndim >= 3:
            try:
                dims = (getattr(layer, "metadata", {}) or {}).get("dims")
                if dims == "TCZYX":
                    zlen = int(arr.shape[2])
                    zcur = int(self.viewer.dims.current_step[1])
                else:
                    zlen = int(arr.shape[-3])
                    zcur = int(self.viewer.dims.current_step[-3])
                self.z_index_spin.setRange(0, max(zlen - 1, 0))
                if not self.z_index_spin.hasFocus():
                    self.z_index_spin.setValue(max(min(zcur, zlen - 1), 0))
            except Exception:
                pass

    # ----------------------- UI sync & view -----------------------

    def _update_info(self, event=None):
        layer = self.viewer.layers.selection.active
        if layer is None or not hasattr(layer, "data"):
            self.lbl_shape.setText("–"); self.lbl_phys.setText("–"); self.lbl_voxpix.setText("–")
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
            nz, ny, nx = info["shape"]; vz, vy, vx = info["voxel"]; pz, py, px = info["phys"]
            self.lbl_shape.setText(f"Z={nz}  Y={ny}  X={nx}")
            self.lbl_phys.setText(f"{pz:.6g}×{py:.6g}×{px:.6g} {info['unit']}")
            self.lbl_voxpix.setText(f"vz={vz:.6g}  vy={vy:.6g}  vx={vx:.6g} {info['unit']}/px")
            self.lblZ.show(); self.edit_vz.show(); self.edit_vz.setEnabled(True)
            self.lblYX.setText("Y/X:"); self.edit_vyx.show(); self.edit_vyx.setEnabled(True)
            self.edit_vz.setValue(max(vz, 0))
            self.edit_vyx.setValue(max(vy, 0)) 
            self.btn_apply_voxel.setText("Update Voxel size")

        elif info.get("ndim") == 2:
            ny, nx = info["shape"]; vy, vx = info["voxel"]; py, px = info["phys"]
            self.lbl_shape.setText(f"Y={ny}  X={nx}")
            self.lbl_phys.setText(f"{py:.6g}×{px:.6g} {info['unit']}")
            self.lbl_voxpix.setText(f"py={vy:.6g}  px={vx:.6g} {info['unit']}/px")
            self.lblZ.hide(); self.edit_vz.hide(); self.edit_vz.setEnabled(False)
            self.lblYX.setText("Y/X:")
            self.edit_vyx.show(); self.edit_vyx.setEnabled(True)
            self.edit_vyx.setValue(max(vy, 0))
            try:
                self.edit_vy.hide(); self.edit_vy.setEnabled(False)
                self.edit_vx.hide(); self.edit_vx.setEnabled(False)
            except Exception:
                pass
            self.btn_apply_voxel.setText("Update Pixel size")

        else:
            self.lbl_shape.setText(str(info.get("shape")))
            self.lbl_phys.setText("–"); self.lbl_voxpix.setText("–")

        _safe_axis_labels(self.viewer, layer)
        try:
            self.viewer.scale_bar.visible = True
            self.viewer.scale_bar.unit = _get_units_from_layer(layer)
        except Exception:
            pass

        # preview kwargs if frangi already run
        self._update_segmentation_preview()

    def _fit_view_and_scalebar(self):
        layer = self.viewer.layers.selection.active
        if layer is None or not hasattr(layer, "data"):
            return

        # Only center the view the first time after image is fully added
        if hasattr(self, "_view_initialized") and self._view_initialized:
            # Already initialized, only refresh scale bar and labels
            try:
                self.viewer.scale_bar.visible = True
                unit = (getattr(layer, "metadata", {}) or {}).get("unit", "um")
                self.viewer.scale_bar.unit = unit
                self.viewer.scale_bar.position = "bottom_right"
                self.viewer.scale_bar.font_size = 8
            except Exception:
                pass
            _safe_axis_labels(self.viewer, layer)
            return

        # Mark initialized immediately to avoid repeated triggering
        self._view_initialized = True

        try:
            # Ensure the layer's extent and scale are valid before centering
            if np.all(np.isfinite(layer.extent.world[0])) and np.all(np.isfinite(layer.extent.world[1])):
                self.viewer.reset_view()

            nd, sh = layer.data.ndim, layer.data.shape
            dims = (getattr(layer, "metadata", {}) or {}).get("dims")

            if dims == "TCZYX":
                if sh[0] > 1:
                    self.viewer.dims.set_current_step(0, int(sh[0] // 2))  # center T
                if sh[2] > 1:
                    self.viewer.dims.set_current_step(1, int(sh[2] // 2))  # center Z
            else:
                if nd >= 4 and sh[0] > 1:
                    self.viewer.dims.set_current_step(0, int(sh[0] // 2))
                if nd >= 3 and sh[-3] > 1:
                    self.viewer.dims.set_current_step(nd - 3, int(sh[-3] // 2))
        except Exception:
            pass

        # Update scale bar and axis labels
        try:
            self.viewer.scale_bar.visible = True
            unit = (getattr(layer, "metadata", {}) or {}).get("unit", "um")
            self.viewer.scale_bar.unit = unit
            self.viewer.scale_bar.position = "bottom_right"
            self.viewer.scale_bar.font_size = 8
        except Exception:
            pass
        _safe_axis_labels(self.viewer, layer)


    # ----------------------- voxel/pixel apply -----------------------

    def _on_apply_voxel_clicked(self):
        """Update active layer scale, sync all layers in same group_id,
        refresh contexts, and recenter view."""
        layer = self.viewer.layers.selection.active
        if layer is None or not hasattr(layer, "data"):
            QMessageBox.information(self, "No image", "Please select an image layer first.")
            return
        nd = layer.data.ndim
        sc = list(getattr(layer, "scale", (1,) * nd))
        try:
            # set new scale from editors
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

            # propagate to all layers with same group_id (includes frangi/seg/base)
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

            # refresh last_frangi_ctx pixel_size if exists to reflect new scale
            if self._last_frangi_ctx:
                try:
                    dim = int(self._last_frangi_ctx.get("dim", 2))
                    self._last_frangi_ctx["pixel_size"] = _get_pixel_size_tuple(layer, dim)
                except Exception:
                    pass

            # Recenter view
            try:
                self.viewer.reset_view()
            except Exception:
                pass
            self._fit_view_and_scalebar()
            self._update_info()
        except Exception as e:
            QMessageBox.critical(self, "Apply failed", f"Failed to set pixel/voxel size: {e!r}")

    # ----------------------- C/T controls -----------------------

    def _set_ct_controls(self, T: int, C: int):
        self.t_spin.blockSignals(True); self.t_spin.setRange(1, max(int(T), 1)); self.t_spin.setValue(1); self.t_spin.setEnabled(int(T) > 1); self.t_spin.blockSignals(False)
        self.c_spin.blockSignals(True); self.c_spin.setRange(1, max(int(C), 1)); self.c_spin.setValue(1); self.c_spin.setEnabled(int(C) > 1); self.c_spin.blockSignals(False)

    def _on_t_changed(self, v1: int):
        try:
            self.viewer.dims.set_current_step(0, max(int(v1) - 1, 0))
        except Exception:
            pass
        self._update_info()

    def _on_c_changed(self, v1: int):
        idx0 = max(int(v1) - 1, 0)
        for lyr in self._group_layers:
            md = getattr(lyr, "metadata", {}) or {}
            if md.get("channel_index") == idx0:
                try:
                    self.viewer.layers.selection.active = lyr
                except Exception:
                    pass
                break
        self._update_info()

    # ----------------------- sigma list -----------------------

    def _make_sigma_list(self) -> Optional[List[float]]:
        smin = float(self.sigma_min.value()); smax = float(self.sigma_max.value()); cnt = int(self.sigma_count.value())
        if smin < 0.1: self.sigma_min.setValue(0.1); smin = 0.1
        if smax > 5.0: self.sigma_max.setValue(5.0); smax = 5.0
        if smax <= smin:
            QMessageBox.warning(self, "Sigma range invalid", "Ensure: max > min within [0.1, 5.0]."); return None
        if cnt < 1: self.sigma_count.setValue(1); cnt = 1
        return np.linspace(smin, smax, cnt, dtype=float).tolist()

    # ----------------------- Frangi filter -----------------------

    def _extract_current_2d_or_3d(self, layer) -> Tuple[np.ndarray, int, float]:
        """
        Return (img, dim, z_over_x_ratio). dim in {2,3}. For 2D ratio=1.0.
        Accepts layer data shaped like (T,Z,Y,X) or (Z,Y,X) or (Y,X).
        Respects the 2D-slice option if enabled.
        """
        arr = np.asarray(layer.data)
        try:
            t_idx0 = int(self.viewer.dims.current_step[0]) if arr.ndim >= 4 else 0
        except Exception:
            t_idx0 = 0
        vol = np.squeeze(arr[t_idx0]) if arr.ndim >= 4 else arr
        dims = (getattr(layer, "metadata", {}) or {}).get("dims")

        if vol.ndim == 3:
            sc = getattr(layer, "scale", (1,) * layer.data.ndim)
            z = float(sc[-3]) if len(sc) >= 3 else 1.0
            x = float(sc[-1]) if len(sc) >= 1 else 1.0
            ratio = (z / x) if (z > 0 and x > 0) else 1.0
            if self.chk_use_2d.isChecked():
                # choose slice index from spin
                if dims == "TCZYX":
                    z_idx = int(np.clip(self.z_index_spin.value(), 0, vol.shape[0]-1))
                else:
                    z_idx = int(np.clip(self.z_index_spin.value(), 0, vol.shape[-3]-1))
                img2d = vol[z_idx]
                return img2d.astype(np.float32, copy=False), 2, 1.0
            return vol.astype(np.float32, copy=False), 3, float(ratio)
        elif vol.ndim == 2:
            return vol.astype(np.float32, copy=False), 2, 1.0
        else:
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
            QMessageBox.information(self, "No image", "Please select an image layer first."); return

        device = self._resolve_device()

        # Prepare input img for Frangi (and context for segmentation)
        try:
            img, dim, ratio = self._extract_current_2d_or_3d(layer)
            rng = float(img.max()) - float(img.min())
            img = (img - img.min()) / (rng if rng > 0 else 1.0) * 255.0
        except Exception as e:
            QMessageBox.critical(self, "Slice error", f"Failed to get 2D/3D data: {e!r}"); return

        # Run Frangi
        try:
            if dim == 3:
                Fr = FrangiFilter(channels=1, kernel_size=2*int(self.kernel_spin.value())+1,
                                  sigmas=sigmas, dim=3, zx_ratio=ratio, device=device)
            else:
                Fr = FrangiFilter(channels=1, kernel_size=2*int(self.kernel_spin.value())+1,
                                  sigmas=sigmas, dim=2, device=device)
            frangi_result = Fr(-np.expand_dims(img, 0))[0].cpu().numpy()
        except Exception as e:
            QMessageBox.critical(self, "Frangi failed", f"FrangiFilter error: {e!r}"); return

        # Prepare metadata BEFORE adding the layer so the inserted-callback can see is_frangi flag
        base_md = dict(getattr(layer, "metadata", {}) or {})
        base_md["source"] = base_md.get("source", _get_source_from_layer(layer))
        base_md["unit"] = _get_units_from_layer(layer)
        base_md["is_frangi"] = True
        base_md["group_id"] = _get_group_id(self.viewer.layers.selection.active) or _ensure_group_id(base_md)
        if "channel_index" in base_md:
            base_md["channel_index"] = base_md["channel_index"]

        # Add frangi layer with UPDATED scale from current active layer
        name = "frangi"; existing = {l.name for l in self.viewer.layers}; k = 1
        while name in existing: k += 1; name = f"frangi_{k}"
        sc = getattr(layer, "scale", (1,) * layer.data.ndim)
        # compute desired scale strictly from active layer.scale
        if frangi_result.ndim == 3:
            desired_scale = (float(sc[-3]), float(sc[-2]), float(sc[-1]))
        else:
            # When running 2D on a slice from 3D, make sure we use (vy, vx) from the 3D pixel size
            try:
                if self.chk_use_2d.isChecked() and layer.data.ndim >= 3:
                    _z, _y, _x = _get_pixel_size_tuple(layer, 3)
                    desired_scale = (float(_y), float(_x))
                else:
                    desired_scale = (float(sc[-2]), float(sc[-1]))
            except Exception:
                desired_scale = (float(sc[-2]), float(sc[-1]))

        # Create with both scale and metadata set so the inserted handler won't override it
        new_layer = self.viewer.add_image(np.asarray(frangi_result), name=name, scale=desired_scale, metadata=base_md)
        try:
            # be extra robust across napari versions
            new_layer.scale = desired_scale
        except Exception:
            pass

        md = base_md  # for context building below

        # Save context
        self._last_frangi_ctx = dict(
            image=img,
            frangi=frangi_result,
            dim=dim,
            pixel_size=_get_pixel_size_tuple(layer, dim),
            device=device,
            source=md.get("source"),
            channel_index=md.get("channel_index"),
            unit=md.get("unit", "um"),
        )

        # update preview text
        self._update_segmentation_preview()

        dimtxt = "2D (slice)" if (dim == 2 and self.chk_use_2d.isChecked()) else ("3D" if dim == 3 else "2D")
        self.kw_preview.append(f"Frangi OK | {dimtxt} | sigmas={np.array(sigmas)} | device={device}")

    # ----------------------- Segmentation -----------------------

    def _build_segmentation_kwargs(self) -> Optional[Dict[str, Any]]:
        if not self._last_frangi_ctx:
            return None
        ctx = self._last_frangi_ctx
        # IMPORTANT: recompute pixel_size from CURRENT active layer scale to reflect any updates
        layer = self.viewer.layers.selection.active
        try:
            fresh_px = _get_pixel_size_tuple(layer, ctx["dim"])
        except Exception:
            fresh_px = ctx["pixel_size"]
        return dict(
            image=ctx["image"],
            frangi=ctx["frangi"],
            pixel_size=fresh_px,
            beta1=float(self.beta1_spin.value()),
            beta2=float(self.beta2_spin.value()),
            cutoff=float(self.cutoff_spin.value()),
            n_fore=int(self.nfore_spin.value()),
            n_back=int(self.nback_spin.value()),
            max_iter=int(self.maxiter_spin.value()),
            device=ctx["device"],
        )

    def _update_segmentation_preview(self):
        kwargs = self._build_segmentation_kwargs()
        if kwargs is None:
            self.kw_preview.setPlainText("Run Frangi first to preview segmentation() call.")
            return
        lines = [
            "segmentation() will be called with:",
            f"  image: float32 array, shape={np.asarray(kwargs['image']).shape}",
            f"  frangi: float32 array, shape={np.asarray(kwargs['frangi']).shape}",
            f"  pixel_size: {kwargs['pixel_size']} ({'z,y,x' if len(kwargs['pixel_size'])==3 else 'y,x'})",
            f"  beta1={kwargs['beta1']}  beta2={kwargs['beta2']}  cutoff={kwargs['cutoff']}",
            f"  n_fore={kwargs['n_fore']}  n_back={kwargs['n_back']}  max_iter={kwargs['max_iter']}",
            f"  device='{kwargs['device']}'",
            "Notes:",
            "  • If 'Run on 2d slice' is enabled, 'image' is that 2D slice and pixel_size is (y,x)."
        ]
        self.kw_preview.setPlainText("\n".join(lines))

    def _on_run_segmentation_clicked(self):
        if not self._last_frangi_ctx:
            QMessageBox.information(self, "Run Frangi first", "Please run Frangi before segmentation."); 
            return

        kwargs = self._build_segmentation_kwargs()
        if kwargs is None:
            return

        self.seg_progress.setRange(0, 0)
        self.seg_progress.setFormat("Running segmentation…")
        self.seg_btn.setEnabled(False)

        self._seg_thread = QThread()
        self._seg_worker = SegWorker(kwargs)
        self._seg_worker.moveToThread(self._seg_thread)
        self._seg_thread.started.connect(self._seg_worker.run)
        self._seg_worker.started.connect(lambda: None)
        def _finished(result, error):
            self.seg_progress.setRange(0, 100)
            if error is not None:
                self.seg_progress.setValue(0)
                QMessageBox.critical(self, "Segmentation failed", f"segmentation() error: {error!r}")
            else:
                self.seg_progress.setValue(100)
                self._add_segmentation_result(result)
            self.seg_btn.setEnabled(True)
            self._seg_thread.quit(); self._seg_thread.wait(); self._seg_worker.deleteLater(); self._seg_thread.deleteLater()
        self._seg_worker.finished.connect(_finished)
        self._seg_thread.start()

    def _add_segmentation_result(self, seg):
        try:
            layer = self.viewer.layers.selection.active
            md_base = dict(getattr(layer, "metadata", {}) or {}) if layer else {}
            seg_np = np.asarray(seg)
            sc_layer = getattr(layer, "scale", None) if layer else None
            add_kwargs = {}
            if sc_layer is not None:
                if seg_np.ndim == 3: add_kwargs["scale"] = (sc_layer[-3], sc_layer[-2], sc_layer[-1])
                elif seg_np.ndim == 2: add_kwargs["scale"] = (sc_layer[-2], sc_layer[-1])

            name = "segmentation"; existing = {l.name for l in self.viewer.layers}; k = 1
            while name in existing: k += 1; name = f"segmentation_{k}"

            if np.issubdtype(seg_np.dtype, np.integer) or seg_np.dtype == bool:
                new_layer = self.viewer.add_labels(seg_np.astype(np.int32, copy=False), name=name, **add_kwargs)
            else:
                new_layer = self.viewer.add_image(seg_np, name=name, **add_kwargs)

            try:
                new_layer.opacity = 0.5
            except Exception:
                pass

            ctx = self._last_frangi_ctx
            new_md = dict(md_base)
            new_md["unit"] = ctx.get("unit", _get_units_from_layer(layer) if layer else "um")
            new_md["source"] = ctx.get("source", _get_source_from_layer(layer) if layer else None)
            new_md["is_segmentation"] = True
            new_md["group_id"] = _get_group_id(self.viewer.layers.selection.active) or _ensure_group_id(new_md)
            new_layer.metadata = new_md

            try:
                self.viewer.scale_bar.visible = True
                self.viewer.scale_bar.unit = new_md["unit"]
            except Exception:
                pass

            self.kw_preview.append(
                "Segmentation OK | "
                f"dim={ctx['dim']} | px={_get_pixel_size_tuple(layer, ctx['dim'])} | "
                f"β1={float(self.beta1_spin.value())} β2={float(self.beta2_spin.value())} cutoff={float(self.cutoff_spin.value())} "
                f"nfore={int(self.nfore_spin.value())} nback={int(self.nback_spin.value())} maxiter={int(self.maxiter_spin.value())} | "
                f"device={ctx['device']}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Add result failed", f"Failed to add segmentation result: {e!r}")


# ----------------------- napari factory / plugin hooks -----------------------

def create_segment_confocal_widget(viewer):
    """Factory for creating the widget in scripts."""
    return SegmentConfocalWidget(viewer)


def napari_experimental_provide_dock_widget():
    """Allow napari plugin discovery to pick up this widget."""
    return [SegmentConfocalWidget]


if __name__ == "__main__":
    # Minimal demo runner for manual testing
    try:
        import napari
        v = napari.Viewer()
        w = SegmentConfocalWidget(v)
        v.window.add_dock_widget(w, area='right')
        napari.run()
    except Exception as e:
        import traceback
        print("Failed to launch napari demo:", e)
        traceback.print_exc()

