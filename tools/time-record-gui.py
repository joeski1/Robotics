#!/usr/bin/env python3
import sys
import os
import time
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout

start_time = None
paused_time = None
window = None

def time_string(duration):
    m, s = divmod(duration, 60)
    return '{:02d}:{:02d}'.format(int(m), int(s))


class MyEdit(QLineEdit):
    def __init__(self, parent):
        super().__init__(parent)

    def keyPressEvent(self, event):
        global start_time
        global paused_time

        if start_time is None:
            if event.key() == Qt.Key_Return:
                start_time = time.time()
                paused_time = start_time
                window.submitTime()
        else:
            if event.key() == Qt.Key_Return:
                window.submitTime()
            elif paused_time is None:
                paused_time = time.time()
                window.displayTime()

        super().keyPressEvent(event)
        self.setText(self.text().upper())


class MyWin(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setFixedSize(640, 480)

        self.label = QLabel(time_string(0), self)
        f = self.label.font()
        f.setPointSize(48)
        self.label.setFont(f)

        self.edit = MyEdit(self)
        self.edit.setFont(f)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.label)
        self.layout().addWidget(self.edit)


        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.displayTime)
        self.timer.start()

    def displayTime(self):
        if start_time is None:
            self.label.setText(time_string(0))
        elif paused_time is not None:
            self.label.setText('paused: ' + time_string(paused_time - start_time))
        else:
            dur = time.time() - start_time
            if dur > 5*60:
                self.label.setStyleSheet('QLabel { color: red; }')
            else:
                self.label.setStyleSheet('QLabel { color: black; }')
            self.label.setText(time_string(dur))

    def submitTime(self):
        global start_time
        global paused_time
        if paused_time is None:
            paused_time = time.time() # enter key pressed, no text entered
        cell = self.edit.text().strip()
        self.edit.clear()
        cell_time = paused_time - start_time
        line = time_string(cell_time) + ', ' + cell
        print(line)
        with open('times.csv', 'a') as f:
            f.write(line + '\n')
        paused_time = None
        self.displayTime()






if __name__ == "__main__":
    if os.path.isfile('times.csv'):
        print('times.csv already exists')
        sys.exit(1)
    app = QApplication(sys.argv)
    window = MyWin()
    window.show()
    sys.exit(app.exec_())
