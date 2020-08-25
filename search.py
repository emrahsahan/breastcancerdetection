import sys, os
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5 import uic
from sqlitehelper import *
from PyQt5 import QtCore, QtGui, QtWidgets
from application import *

helper = SqliteHelper("database.db")

class Search(QMainWindow):
    def __init__(self):
        super().__init__()
        self.searchinitUI()
    
    def searchinitUI(self):
        self.window = uic.loadUi(r"search.ui")
        self.window.searchButton.clicked.connect(self.load_test)
        self.window.printButton.clicked.connect(self.print_pdf)
        self.window.deleteButton.clicked.connect(self.delete)
        self.window.show()
    
    def load_test(self):
        
        self.clear_table()
        tc = self.window.lineEdit_search.text()
        if tc!="":       
            tests = helper.select("SELECT * FROM test WHERE Tc=" + tc + " ORDER BY ID DESC")

            for row_number, test in enumerate(tests):
                self.window.tableWidget.insertRow(row_number)
                for column_number, data in enumerate(test):
                    cell = QtWidgets.QTableWidgetItem(str(data))
                    self.window.tableWidget.setItem(row_number, column_number, cell)        
    
    def clear_table(self):
        self.window.tableWidget.clearSelection()
        while self.window.tableWidget.rowCount() > 0:
            self.window.tableWidget.removeRow(0)
            self.window.tableWidget.clearSelection()
    
    def print_pdf(self):
        formatted_time = self.window.tableWidget.item(self.window.tableWidget.currentRow(), 1).text()
        tc = self.window.tableWidget.item(self.window.tableWidget.currentRow(), 2).text()
        name = self.window.tableWidget.item(self.window.tableWidget.currentRow(), 3).text()
        result = int(self.window.tableWidget.item(self.window.tableWidget.currentRow(), 4).text())
        App.pdf(self, formatted_time, tc, name, result)
        os.startfile("sonuc.pdf")
    
    def delete(self):
        try:
            id = self.window.tableWidget.item(self.window.tableWidget.currentRow(), 0).text()
            helper.delete("DELETE FROM test WHERE ID=" + id)
            self.load_test()
        except:
            QMessageBox.information(None, "Uyarı", "Silmek için tablodan test seçiniz.")       

if __name__ == "__main__":
    app = QApplication(sys.argv)
    src = Search()
    sys.exit(app.exec_())
