import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5 import uic
import numpy as np
import pandas as pd
import pickle
import os
from sqlitehelper import *
import time
from search import *

helper = SqliteHelper("database.db")

class App(QMainWindow):
    # program çalıştırıldığında initiliaze yapılır.
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model_fit()
    
    # ui dosyasını pencerede aç
    def initUI(self):
        self.filename = 'final_model.pkl'
        self.win = uic.loadUi(r"main.ui")
        self.win.pushButton_save.clicked.connect(self.save_show_print)
        self.win.pushButton_clear.clicked.connect(self.clear)
        self.win.action_search.triggered.connect(self.open)
        self.win.action_exit.triggered.connect(self.exit)
        self.win.show()
    
    # arama penceresini aç
    def open(self):
        self.dialog = Search()
        
    # mesaj göster
    def show_message(self, title='Uyarı', message='Hata oluştu'):
        QMessageBox.information(None, title, message)
        
    # modeli eğit
    def model_fit(self):
        df = pd.read_csv('data.csv')
        df.dropna(axis=1)
        
        # Encoding diagnosis data to 0 and 1
        from sklearn.preprocessing import LabelEncoder
        labelencoder_Y = LabelEncoder()
        df.iloc[:,1] = labelencoder_Y.fit_transform(df.iloc[:,1].values)
        
        X = df.iloc[:, 2:32].values
        Y = df.iloc[:, 1].values
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

        # Feature scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        self.scaler = StandardScaler()
        self.fitting = self.scaler.fit(X)
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        # logistic regression
        from sklearn.linear_model import LogisticRegression
        log = LogisticRegression(random_state=0)
        log.fit(X_train, Y_train)
        
        # kneighbors classifier
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        knn.fit(X_train, Y_train)
        
        # svc linear model
        from sklearn.svm import SVC
        svc_lin = SVC(kernel='linear', random_state=0)
        svc_lin.fit(X_train, Y_train)
        
        # SVC rbf model
        svc_rbf = SVC(kernel='rbf', random_state=0)
        svc_rbf.fit(X_train, Y_train)
        
        # gaussianNB model
        from sklearn.naive_bayes import GaussianNB
        gauss = GaussianNB()
        gauss.fit(X_train, Y_train)
        
        # DecisionTreeClassifier Model
        from sklearn.tree import DecisionTreeClassifier
        tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
        tree.fit(X_train, Y_train)
        
        # RandomForestClassifier Model
        from sklearn.ensemble import RandomForestClassifier
        forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
        forest.fit(X_train, Y_train)
        
        models = [log, knn, svc_lin, svc_rbf, gauss, tree, forest]
        
        for i in range(len(models)):
            print("Model {} score {}".format(i + 1, models[i].score(X_train, Y_train)))
      
        
        from sklearn.metrics import classification_report
        from sklearn.metrics import accuracy_score
        
        scores = []
        
        # model sonuçlarını yazdır.
        for i in range(len(models)):
            print('Model', i + 1)
            print( classification_report(Y_test, models[i].predict(X_test)))
            print('Accuracy Score',accuracy_score(Y_test, models[i].predict(X_test)))
            scores.append(accuracy_score(Y_test, models[i].predict(X_test)))
            print('--------------------------------------------')
        
        #○en büyük accuracy score index ini bul
        top_score_model_index = scores.index(max(scores))
        selected_model = models[top_score_model_index]
        print("Seçilen Model:", selected_model)
        
        # modeli kaydet
        pickle.dump(selected_model, open(self.filename, 'wb'))
        
        # Mesaj göster
        self.show_message("Bilgilendirme", 'Model eğitimi tamamlandı. Verileri girdikten sonra sonuçları gösterebilirsiniz.')
    
    # formdan gelen veriyi eğitilen modele göre sonuçlandır ve veritabanına kaydet.
    def save_show_print(self):
        try:
            tc = self.win.lineEdit_tc.text()   
            name = self.win.lineEdit_name.text()
            radiusmean = float(self.win.lineEdit_radiusmean.text())
            texturemean = float(self.win.lineEdit_texturemean.text())
            perimetermean = float(self.win.lineEdit_perimetermean.text())
            areamean = float(self.win.lineEdit_areamean.text())
            smoothnessmean = float(self.win.lineEdit_smoothnessmean.text())
            compactnessmean = float(self.win.lineEdit_compactnessmean.text())
            concavitymean = float(self.win.lineEdit_concavitymean.text())
            concavepointsmean = float(self.win.lineEdit_concavepointsmean.text())
            symmetrymean = float(self.win.lineEdit_symmetrymean.text())
            fractaldimensionmean = float(self.win.lineEdit_fractaldimensionmean.text())
            radiusse = float(self.win.lineEdit_radiusse.text())
            texturese = float(self.win.lineEdit_texturese.text())
            perimeterse = float(self.win.lineEdit_perimeterse.text())
            arease = float(self.win.lineEdit_arease.text())
            smoothnessse = float(self.win.lineEdit_smoothnessse.text())
            compactnessse = float(self.win.lineEdit_compactnessse.text())
            concavityse = float(self.win.lineEdit_concavityse.text())
            concavepointsse = float(self.win.lineEdit_concavepointsse.text())
            symmetryse = float(self.win.lineEdit_symmetryse.text())
            fractaldimensionse = float(self.win.lineEdit_fractaldimensionse.text())
            radiusworst = float(self.win.lineEdit_radiusworst.text())
            textureworst = float(self.win.lineEdit_textureworst.text())
            perimeterworst = float(self.win.lineEdit_perimeterworst.text())
            areaworst = float(self.win.lineEdit_areaworst.text())
            smoothnessworst = float(self.win.lineEdit_smoothnessworst.text())
            compactnessworst = float(self.win.lineEdit_compactnessworst.text())
            concavityworst = float(self.win.lineEdit_concavityworst.text())
            concavepointsworst = float(self.win.lineEdit_concavepointsworst.text())
            symmetryworst = float(self.win.lineEdit_symmetryworst.text())
            fractaldimensionworst = float(self.win.lineEdit_fractaldimensionworst.text())
            
            data = [radiusmean, texturemean, perimetermean, areamean, smoothnessmean, 
                compactnessmean, concavitymean, concavepointsmean, symmetrymean,
                fractaldimensionmean, radiusse, texturese, perimeterse, arease, smoothnessse, 
                compactnessse, concavityse, concavepointsse, symmetryse,
                fractaldimensionse, radiusworst, textureworst, perimeterworst, areaworst, smoothnessworst, 
                compactnessworst, concavityworst, concavepointsworst, symmetryworst, fractaldimensionworst]

            if len(tc)==11:
                data = pd.DataFrame([data])
                data_scaled = self.fitting.transform(data)
                #modeli yükle
                loaded_model = pickle.load(open(self.filename, 'rb'))
                
                #formdan gelen veriyi gönder ve tahmini sonucu ata.
                result = loaded_model.predict(data_scaled)[0]
                print("Sn.{}".format(name))
                print("T.C Kimlik NO:{}".format(tc))
                
                if result == 0:
                    rslt = 'iyi huylu'
                    print('Sonuç: İyi huylu')
                else:
                    rslt = 'kötü huylu'
                    print('Sonuç: Kötü huylu')
                
        
                formatted_time = time.strftime('%d-%m-%Y %H:%M:%S')
                formdata = (formatted_time, tc, name, str(result))
                helper.insert("INSERT INTO test(Date, Tc, Name, Result) VALUES(?,?,?,?)", formdata)
                
                self.pdf(formatted_time, tc, name, result)
                os.startfile("sonuc.pdf")
            else:
                self.show_message('Bilgilendirme', 'T.C Kimlik Numarası 11 haneli olmalıdır.')         
        
        except ValueError as e:
            print('Hata kodu:', e)
            self.show_message("Bilgilendirme", 'Girdiğiniz verilerde hata var veya veriler eksik. Ondalıklı sayıları .(nokta) ile yazınız')
        
    # pdf oluştur.
    def pdf(self, formatted_time, tc, name, result):
        from reportlab.lib.enums import TA_JUSTIFY
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        
        pdfmetrics.registerFont(TTFont('TNR', 'times.ttf'))

        doc = SimpleDocTemplate("sonuc.pdf",pagesize=letter,
                                rightMargin=72,leftMargin=72,
                                topMargin=72,bottomMargin=18)
        
        metin=[]
        logo = "logo.jpg"
        
        im = Image(logo, width=80, height=80)
        metin.append(im)
        metin.append(Spacer(1, 30))

        styles=getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY, fontName="TNR"))
        ptext = '<font size="20">Göğüs Kanseri Testi Sonuç Belgesi</font>'
        metin.append(Paragraph(ptext, styles["Justify"]))
        metin.append(Spacer(1, 30))
        
        ptext = '<font size="14"> Sonuç Tarihi: %s</font>' % formatted_time
        metin.append(Paragraph(ptext, styles["Justify"]))
        metin.append(Spacer(1, 12))

        new_tc = tc[:2] + '*******' + tc[9:]
        ptext = '<font size="14">T.C Kimlik NO: %s</font>' % new_tc
        metin.append(Paragraph(ptext, styles["Justify"]))
        metin.append(Spacer(1, 12))
        ptext = '<font size="14">Adı-Soyadı: %s</font>' % name
        metin.append(Paragraph(ptext, styles["Justify"]))
         
        metin.append(Spacer(1, 25))
        ptext = '<font size="14">Sayın %s:</font>' % name
        metin.append(Paragraph(ptext, styles["Justify"]))
        metin.append(Spacer(1, 12))

        if result == 0:
            rslt = '<b>iyi huylu</b>'
        else:
            rslt = '<b>kötü huylu</b>'        
        
        ptext = '<font size="14">Hastanemizde yaptırmış olduğunuz testlerin sonucunda Göğüs Kanseri Testiniz %s \
                çıkmıştır.</font>' % (rslt)
        
        metin.append(Paragraph(ptext, styles["Justify"]))
        metin.append(Spacer(1, 12))


        ptext = '<font size="14">Bizi tercih ettiğiniz için teşekkür ederiz.</font>'
        metin.append(Paragraph(ptext, styles["Justify"]))
        metin.append(Spacer(1, 12))
        ptext = '<font size="14">Görüşmek dileğiyle,</font>'
        metin.append(Paragraph(ptext, styles["Justify"]))
        metin.append(Spacer(1, 48))
        ptext = '<font size="14">Nazilli Devlet Hastanesi</font>'
        metin.append(Paragraph(ptext, styles["Justify"]))
        metin.append(Spacer(1, 12))
        doc.build(metin)    
    
    # çıkış
    def exit(self):
        sys.exit(app.exec_())
    
    # formu temizle
    def clear(self):
        self.win.lineEdit_tc.setText('')
        self.win.lineEdit_name.setText('')
        self.win.lineEdit_radiusmean.setText('')
        self.win.lineEdit_texturemean.setText('')
        self.win.lineEdit_perimetermean.setText('')
        self.win.lineEdit_areamean.setText('')
        self.win.lineEdit_smoothnessmean.setText('')
        self.win.lineEdit_compactnessmean.setText('')
        self.win.lineEdit_concavitymean.setText('')
        self.win.lineEdit_concavepointsmean.setText('')
        self.win.lineEdit_symmetrymean.setText('')
        self.win.lineEdit_fractaldimensionmean.setText('')
        self.win.lineEdit_radiusse.setText('')
        self.win.lineEdit_texturese.setText('')
        self.win.lineEdit_perimeterse.setText('')
        self.win.lineEdit_arease.setText('')
        self.win.lineEdit_smoothnessse.setText('')
        self.win.lineEdit_compactnessse.setText('')
        self.win.lineEdit_concavityse.setText('')
        self.win.lineEdit_concavepointsse.setText('')
        self.win.lineEdit_symmetryse.setText('')
        self.win.lineEdit_fractaldimensionse.setText('')
        self.win.lineEdit_radiusworst.setText('')
        self.win.lineEdit_textureworst.setText('')
        self.win.lineEdit_perimeterworst.setText('')
        self.win.lineEdit_areaworst.setText('')
        self.win.lineEdit_smoothnessworst.setText('')
        self.win.lineEdit_compactnessworst.setText('')
        self.win.lineEdit_concavityworst.setText('')
        self.win.lineEdit_concavepointsworst.setText('')
        self.win.lineEdit_symmetryworst.setText('')
        self.win.lineEdit_fractaldimensionworst.setText('')    
        
    
# programı çalıştır.
if __name__ == "__main__":
    app = QApplication(sys.argv)
    uyg = App()
    sys.exit(app.exec_())