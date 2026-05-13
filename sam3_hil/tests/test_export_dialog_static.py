from pathlib import Path


def test_export_object_label_combos_use_no_wheel_combo_box():
    source = Path("src/gui/main_window_server.py").read_text(encoding="utf-8")

    export_dialog_body = source.split("class ExportDialog", 1)[1]
    export_dialog_body = export_dialog_body.split("class STAMPMainWindow", 1)[0]
    label_combo_block = export_dialog_body.split("# Label combo", 1)[1]
    label_combo_block = label_combo_block.split("row_layout.addWidget(combo)", 1)[0]

    assert "combo = NoWheelComboBox()" in label_combo_block


def test_no_wheel_combo_box_ignores_wheel_events():
    source = Path("src/gui/main_window_server.py").read_text(encoding="utf-8")

    assert "class NoWheelComboBox(QComboBox):" in source
    no_wheel_body = source.split("class NoWheelComboBox(QComboBox):", 1)[1]
    no_wheel_body = no_wheel_body.split("class ExportDialog", 1)[0]

    assert "def wheelEvent(self, event):" in no_wheel_body
    assert "event.ignore()" in no_wheel_body
