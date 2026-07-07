"""Qt binding compatibility layer.

Prefers PySide6 and falls back to PySide2, exposing a uniform interface for
the rest of the UI code:

    from facebin.ui.qt_compat import qtc, qtw, qtg, Signal, Slot, exec_app
"""

from facebin.errors import DependencyError

try:
    from PySide6 import QtCore as qtc
    from PySide6 import QtGui as qtg
    from PySide6 import QtWidgets as qtw
    from PySide6.QtCore import Signal, Slot
    QT_BINDING = "PySide6"
except ImportError:
    try:
        from PySide2 import QtCore as qtc
        from PySide2 import QtGui as qtg
        from PySide2 import QtWidgets as qtw
        from PySide2.QtCore import Signal, Slot
        QT_BINDING = "PySide2"
    except ImportError as e:
        raise DependencyError(
            "Neither PySide6 nor PySide2 is installed; the Facebin GUI "
            "cannot start.",
            hint="Install the UI dependencies with "
            "`pip install 'facebin[ui]'` or run the headless server with "
            "`facebin server`.") from e


def exec_app(app):
    """Run a QApplication event loop on both PySide2 and PySide6."""
    if hasattr(app, "exec"):
        return app.exec()
    return app.exec_()
