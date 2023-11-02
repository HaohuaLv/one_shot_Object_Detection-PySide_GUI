from PySide6 import QtCore, QtGui, QtWidgets
import random
from PIL import Image

from model_utils import *


class RectangleDrawer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setFixedSize(512, 512)
        self.pixmap = QtGui.QPixmap(self.size())
        self.pixmap.fill(QtCore.Qt.GlobalColor.white)
        self.bg_img = self.pixmap.toImage()

        self.painter = QtGui.QPainter()

        self.rect_pen = QtGui.QPen(QtCore.Qt.GlobalColor.red)
        self.rect_pen.setWidth(3)
        self.cord_pen = QtGui.QPen(QtCore.Qt.GlobalColor.black)
        self.cord_pen.setWidth(1)

        self.box = QtCore.QRect()

        self.mousePress = False

        self.setMouseTracking(True)

    def paintEvent(self, event: QtGui.QPaintEvent):
        """Override method from QWidget

        Paint the Pixmap into the widget

        """
        with QtGui.QPainter(self) as painter:
            painter.drawPixmap(0, 0, self.pixmap)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        """Override from QWidget

        Called when user clicks on the mouse

        """
        self.box.setTopLeft(event.position().toPoint())
        self.mousePress = True
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        """Override method from QWidget

        Called when user moves and clicks on the mouse

        """
        current_pos = event.position().toPoint()
        if self.mousePress:
            self.box.setBottomRight(current_pos)

        self.reset()

        self.painter.begin(self.pixmap)

        self.painter.setPen(self.rect_pen)
        self.painter.drawRect(self.box)

        self.painter.setPen(self.cord_pen)
        self.painter.drawLine(current_pos.x(), 0, current_pos.x(), self.pixmap.width())
        self.painter.drawLine(0, current_pos.y(), self.pixmap.height(), current_pos.y())

        self.painter.end()

        self.update()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        """Override method from QWidget

        Called when user releases the mouse

        """
        current_pos = event.position().toPoint()
        self.box.setBottomRight(current_pos)

        self.reset()

        self.painter.begin(self.pixmap)

        self.painter.setPen(self.rect_pen)
        self.painter.drawRect(self.box)

        self.painter.setPen(self.cord_pen)
        self.painter.drawLine(current_pos.x(), 0, current_pos.x(), self.pixmap.width())
        self.painter.drawLine(0, current_pos.y(), self.pixmap.height(), current_pos.y())

        self.painter.end()

        self.update()
        self.mousePress = False

        super().mouseReleaseEvent(event)

    def reset(self):
        """Reset pixmap for drawing"""
        if not self.bg_img.isNull():
            self.pixmap = QtGui.QPixmap.fromImage(self.bg_img)
        else:
            self.pixmap.fill(QtCore.Qt.GlobalColor.white)

    def load_bg_img(self, filename: str):
        """Load pixmap from filename"""
        self.bg_img.load(filename)
        self.bg_img = self.bg_img.scaled(
            self.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio
        )
        self.pixmap = QtGui.QPixmap.fromImage(self.bg_img)
        self.setFixedSize(self.bg_img.size())


class AnnotatiedImage(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        self.image_label = QtWidgets.QLabel(self)
        layout.addWidget(self.image_label)
        self.pixmap = QtGui.QPixmap(512, 512)
        self.image_label.setFixedSize(self.pixmap.size())
        self.pixmap.fill(QtCore.Qt.GlobalColor.darkGray)
        self.image_label.setPixmap(self.pixmap)

    def set_image(self, image: QtGui.QImage):
        self.pixmap = QtGui.QPixmap.fromImage(image)
        self.pixmap = self.pixmap.scaled(
            self.image_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio
        )
        self.image_label.setPixmap(self.pixmap)
        self.image_label.setFixedSize(self.pixmap.size())


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("one-shot Object Detection")

        self.centralwidget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.centralwidget)

        central_vlayout = QtWidgets.QVBoxLayout(self.centralwidget)

        self.image_box = QtWidgets.QWidget(self.centralwidget)
        image_hlayout = QtWidgets.QHBoxLayout(self.image_box)
        self.rectangle_drawer = RectangleDrawer(self.image_box)
        self.annotatied_image = AnnotatiedImage(self.image_box)

        image_hlayout.addWidget(self.rectangle_drawer)
        image_hlayout.addWidget(self.annotatied_image)

        central_vlayout.addWidget(self.image_box)

        self.param_box = QtWidgets.QWidget(self.centralwidget)
        self.param_box.setMinimumHeight(50)
        self.param_box.setMinimumWidth(500)
        box_hlayout = QtWidgets.QHBoxLayout(self.param_box)
        self.threshold_box = QtWidgets.QDoubleSpinBox(self.param_box)
        self.threshold_box.setRange(0, 1)
        self.threshold_box.setSingleStep(0.01)
        self.threshold_box.setValue(0.95)
        self.threshold_box.setPrefix("Score Threshold: ")
        self.threshold_box.valueChanged.connect(self._threshold_changed)

        self.nms_box = QtWidgets.QDoubleSpinBox(self.param_box)
        self.nms_box.setRange(0, 1)
        self.nms_box.setSingleStep(0.05)
        self.nms_box.setValue(0.3)
        self.nms_box.setPrefix("NMS Threshold: ")
        self.nms_box.valueChanged.connect(self._threshold_changed)

        box_hlayout.addWidget(self.threshold_box)
        box_hlayout.addWidget(self.nms_box)

        central_vlayout.addWidget(self.param_box)

        self.bar = self.addToolBar("Menu")
        self.bar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)

        self._open_action = self.bar.addAction(
            QtWidgets.QApplication.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton
            ),
            "Open",
            self._open,
        )
        self._open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)

        self._run_action = self.bar.addAction(
            QtWidgets.QApplication.style().standardIcon(
                QtWidgets.QStyle.StandardPixmap.SP_DialogOkButton
            ),
            "Run",
            self._run,
        )
        self._run_action.setShortcut("Enter")

        self.resize(QtGui.QGuiApplication.primaryScreen().availableSize() * 3 / 5)

        self.outputs = None

    @QtCore.Slot()
    def _open(self):
        dialog = QtWidgets.QFileDialog(self, "Save File")
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
        dialog.setDefaultSuffix("png")

        if dialog.exec() == QtWidgets.QFileDialog.DialogCode.Accepted:
            if dialog.selectedFiles():
                self.rectangle_drawer.load_bg_img(dialog.selectedFiles()[0])

    @QtCore.Slot()
    def _run(self):
        self._get_outputs()
        self._get_result_and_paint()

    def _get_outputs(self):
        image = Image.fromqimage(self.rectangle_drawer.bg_img)
        target_sizes = torch.Tensor([image.size[::-1]])
        manul_box = (
            self.rectangle_drawer.box.left(),
            self.rectangle_drawer.box.top(),
            self.rectangle_drawer.box.right(),
            self.rectangle_drawer.box.bottom(),
        )

        inputs = processor(images=image.convert("RGB"), return_tensors="pt")
        self.outputs = model.box_guided_detection(
            **inputs, query_box=torch.Tensor([manul_box]), target_sizes=target_sizes
        )

    def _get_result_and_paint(self):
        if self.outputs is None:
            return
        image = Image.fromqimage(self.rectangle_drawer.bg_img)
        target_sizes = torch.Tensor([image.size[::-1]])
        results = processor.post_process_image_guided_detection(
            outputs=self.outputs,
            threshold=self.threshold_box.value(),
            nms_threshold=self.nms_box.value(),
            target_sizes=target_sizes,
        )
        painter = QtGui.QPainter()
        new_image = self.rectangle_drawer.bg_img.copy()
        pen = QtGui.QPen()
        pen.setWidth(3)
        pen.setColor(QtCore.Qt.GlobalColor.red)

        painter.begin(new_image)

        # pen.setColor(QtCore.Qt.GlobalColor.blue)
        # painter.setPen(pen)
        # painter.drawRect(
        #     manul_box[0],
        #     manul_box[1],
        #     manul_box[2] - manul_box[0],
        #     manul_box[3] - manul_box[1],
        # )
        # pen.setColor(QtCore.Qt.GlobalColor.red)

        boxes = results[0]["boxes"].type(torch.int64).tolist()
        for box in boxes:
            # color = random.choice(list(QtCore.Qt.GlobalColor.__members__.values()))
            # pen.setColor(color)
            painter.setPen(pen)
            painter.drawRect(box[0], box[1], box[2] - box[0], box[3] - box[1])

        painter.end()

        self.annotatied_image.set_image(new_image)

    @QtCore.Slot()
    def _threshold_changed(self):
        self._get_result_and_paint()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())
