
#Import library GUI
from tkinter import *
from tkinter.ttk import Combobox

window=Tk()
var = StringVar()
var.set("one")

#Membuat Class Table interface
class Table:
      
    def __init__(self,root,rows):
        # code for creating table
        newWindow = Toplevel(window)
        newWindow.title("Data Test")
        # sets the geometry of toplevel
        newWindow.geometry("600x400")

        for i in range(len(rows)):
            for j in range(10):
                self.e=Entry(newWindow)
                  
                self.e.grid(row=i, column=j)
                self.e.insert(END, rows[i][j])

#handle tombol klik untuk memilih fitur
def tombol_klik():
    tombol["state"] = DISABLED
    if (v0.get()<2):
        metoda=1
    else:
        if (v0.get()==2):
            metoda=2
        else:
            metoda=3
    print(metoda)
    V=[]
    if (v1.get()==1):
        if not V:
            V=[1]
        else:
            V.append(1)
    if (v2.get()==1):
        if not V:
            V=[2]
        else:
            V.append(2)
    if (v3.get()==1):
        if not V:
            V=[3]
        else:
            V.append(3)
    if (v4.get()==1):
        if not V:
            V=[4]
        else:
            V.append(4)
    if (v5.get()==1):
        if not V:
            V=[5]
        else:
            V.append(5)
    if (v6.get()==1):
        if not V:
            V=[6]
        else:
            V.append(6)
    if (v7.get()==1):
        if not V:
            V=[7]
        else:
            V.append(7)
    if not V:
        tombol["state"] = NORMAL
        print('Fitur tidak boleh kosong, Selesai ...')
        return

    #Import library untuk denoising
    from scipy.signal import kaiserord, lfilter, firwin, freqz
    import scipy as sp
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy import signal
    from numpy import savetxt
    from math import log, e
    import os, sys
    from scipy.signal import butter, iirnotch, lfilter

    #Membuat fungsi untuk denoising sinyal Butterworth
    
    ## A high pass filter allows frequencies higher than a cut-off value
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5*fs
        normal_cutoff = cutoff/nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False, output='ba')
        return b, a
    ## A low pass filter allows frequencies lower than a cut-off value
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5*fs
        normal_cutoff = cutoff/nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
        return b, a
    def notch_filter(cutoff, q):
        nyq = 0.5*fs
        freq = cutoff/nyq
        b, a = iirnotch(freq, q)
        return b, a

    def final_filter(data, fs, order=5):
        b, a = butter_highpass(cutoff_high, fs, order=order)
        x = lfilter(b, a, data)
        d, c = butter_lowpass(cutoff_low, fs, order = order)
        y = lfilter(d, c, x)
        f, e = notch_filter(powerline, 30)
        z = lfilter(f, e, y)     
        return z

    #Membaca dataset sinyal PPG sehat dan pasien

    J=0 # jumlah file
    directory_path = 'dataset/sehat'
    for iter in range(0,2):
        for x in os.listdir(directory_path):
            if not x.lower().endswith('.csv'):
                continue
            J=J+1
        directory_path = 'dataset/pasien'
    n = J #jumlah file
    m = 8
    FEAT = [] #Akan menjadi Feature.csv atau fitur untuk model
    for i in range(n): 
        FEAT.append([0] * m) #mengisi dengan angka 0 semua
    directory_path = 'dataset/sehat'
    J=-1
    K=0
    for iter in range(0,2):
        for x in os.listdir(directory_path):
            if not x.lower().endswith('.csv'):
                continue
            full_file_path = directory_path  +   '/'   + x
            J=J+1
            print ('Using file', full_file_path)
            try:
                dataraw = pd.read_csv(full_file_path,index_col='Timestamp', parse_dates=['Timestamp'])
                dataset = pd.DataFrame(dataraw['Value'])
            except:
                dataraw = pd.read_csv(full_file_path,index_col='timestamp', parse_dates=['timestamp'])
                dataset = pd.DataFrame(dataraw['values'])
            x1=np.array(dataset)
            Dat=[]
            Dat=[0 for i in range(x1.size)]
            for i in range(0,x1.size-1):
                Dat[i]=max(x1[i])
                
            # Denoising Butterworth filter
            fs = 1000
            cutoff_high = 0.5
            cutoff_low = 2
            powerline = 60
            order = 5
            filter_signal = final_filter(Dat, fs, order)
            y2_filtered=[]
            y2_filtered=[0 for i in range(filter_signal.size)]
            for i in range(0,filter_signal.size-1):
                y2_filtered[i]=filter_signal[i]+cutoff_high

            if (metoda==2):
                y_filtered=np.array(y2_filtered)

            #Import Library untuk ekstraksi fitur
            import collections
            import neurokit2 as nk
            from scipy.stats import entropy
            from scipy.stats import norm, kurtosis
            from scipy.stats import skew

            # FEATURE EXTRACTION

            #Fungsi menghitung shannon entropy (ESH)
            def shannon_entropy(data):
                bases = collections.Counter([tmp_base for tmp_base in data])
                # define distribution
                dist = [x / sum(bases.values()) for x in bases.values()]
                # use scipy to calculate entropy
                entropy_value = entropy(dist, base=2)
                return entropy_value

            try:
                # Shannon entropy
                ESH = shannon_entropy(y_filtered)
                FEAT[J][0] = ESH
                # MAD
                series = pd.Series(y_filtered)
                MAD = series.mad()
                # Kurtosis
                KURT = kurtosis(y_filtered)
                # Skewness
                SKEW = skew(y_filtered)
                # vlf,lf,hf
                info = nk.ppg_findpeaks(y_filtered)
                peak = info["PPG_Peaks"]
                hrv_freq = nk.hrv_frequency(peak, sampling_rate=39, normalize=True)
                VLF = hrv_freq['HRV_VLF'].values[0]
                LF = hrv_freq['HRV_LF'].values[0]
                HF = hrv_freq['HRV_HF'].values[0]
                FEAT[J][0] = ESH
                FEAT[J][1] = MAD
                FEAT[J][2] = KURT
                FEAT[J][3] = SKEW
                FEAT[J][4] = VLF
                FEAT[J][5] = LF
                FEAT[J][6] = HF
                FEAT[J][7] = K
            except:
                J = J - 1
        directory_path = 'dataset/pasien'
        K=1
    #sehat = 0, pasien = 1


    #Membaca sinyal PPG Data Uji
    J = 1
    m = 10
    directory_path = 'dataset/uji'
    for x in os.listdir(directory_path):
        if not x.lower().endswith('.csv'):
            continue
        J=J+1
    n = J
    FEAT2 = [] #Akan Menjadi jadi Feature2.csv untuk data uji
    for i in range(n): 
        FEAT2.append([0] * m) #mengisi dengan angka 0 semua
    J=-1
    for x in os.listdir(directory_path):
        if not x.lower().endswith('.csv'):
            continue
        full_file_path = directory_path  +   '/'   + x
        J=J+1
        print ('Using file', full_file_path)
        try:
            dataraw = pd.read_csv(full_file_path,index_col='Timestamp', parse_dates=['Timestamp'])
            dataset = pd.DataFrame(dataraw['Value'])
        except:
            dataraw = pd.read_csv(full_file_path,index_col='timestamp', parse_dates=['timestamp'])
            dataset = pd.DataFrame(dataraw['values'])
        x1=np.array(dataset)
        Dat=[]
        Dat=[0 for i in range(x1.size)]
        for i in range(0,x1.size-1):
            Dat[i]=max(x1[i])

        # Denoising Butterworth filter untuk data uji
        fs = 1000
        cutoff_high = 0.5
        cutoff_low = 2
        powerline = 60
        order = 5
        filter_signal = final_filter(Dat, fs, order)
        y2_filtered=[]
        y2_filtered=[0 for i in range(filter_signal.size)]
        for i in range(0,filter_signal.size-1):
            y2_filtered[i]=filter_signal[i]+cutoff_high

        # FEATURE EXTRACTION untuk fitur data uji
        try:
            # Shannon entropy
            ESH = shannon_entropy(y_filtered)
            FEAT2[J][0] = ESH

            # MAD
            series = pd.Series(y_filtered)
            MAD = series.mad()
            # Kurtosis
            KURT = kurtosis(y_filtered)
            # Skewness
            SKEW = skew(y_filtered)

            # vlf,lf,hf
            info = nk.ppg_findpeaks(y_filtered)
            peak = info["PPG_Peaks"]
            hrv_freq = nk.hrv_frequency(peak, sampling_rate=39, normalize=True)
            VLF = hrv_freq['HRV_VLF'].values[0]
            LF = hrv_freq['HRV_LF'].values[0]
            HF = hrv_freq['HRV_HF'].values[0]
            FEAT2[J][0] = ESH
            FEAT2[J][1] = MAD
            FEAT2[J][2] = KURT
            FEAT2[J][3] = SKEW
            FEAT2[J][4] = VLF
            FEAT2[J][5] = LF
            FEAT2[J][6] = HF
            FEAT2[J][7] = K
        except:
            J = J - 1


    #Membuat csv untuk fitur model dan fitur data uji
    import csv
    #Fitur Model
    with open("Feature.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(FEAT)
    #Fitur data uji
    with open("Feature2.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(FEAT2)

    # MACHINE LEARNING
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    #Membaca Dataset dari Feature.csv
    dataset = pd.read_csv('FEATURE.csv', names=['ESH', 'MAD', 'SKEW','KURT','VLF','LF','HF','Label'])
    x = np.random.randint(9, size=(3, 3))
    dataset['VLF'] = dataset['VLF'].fillna(dataset['VLF'].mean())
    print(dataset)

    #Memisah fitur dan label
    V = [q-1 for q in V]
    X = dataset.iloc[:, V].values
    y = dataset.iloc[:, -1].values

    # SPLIT DATA 80% TRAIN, 20% DATA TEST
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 69)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    #Membaca data yang diujikan
    data_uji = pd.read_csv('FEATURE2.csv')

    # PROSES KLASIFIKASI MACHINE LEARNING
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import metrics
    classifier = KNeighborsClassifier(n_neighbors = 28) #tuning parameter n_neighbors disini
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    #Uji Metrik
    from sklearn.metrics import confusion_matrix, accuracy_score

    ac = accuracy_score(y_test, y_pred)
    knn = ac

    def hitung_akurasi(tp,tn,fp,fn):
        acc = float
        acc = (tp+tn)/(tp+tn+fp+fn)
        return acc
    def hitung_sensitifiti(tp, fn):
        sens = float
        sens = tp/(tp+fn)
        return sens
    def hitung_spesifisiti(tn,fp):
        spes = float
        spes = tn/(tn+fp)
        return spes

    X_uji = data_uji.iloc[:, V]
    predicted = classifier.predict(X_uji)
    cm = confusion_matrix(y_test, y_pred)
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    acc_knn = hitung_akurasi(tp, tn, fp, fn)
    sens_knn = hitung_sensitifiti(tp, fn)
    spes_knn = hitung_spesifisiti(tn, fp)

    cm = confusion_matrix(y_test, y_pred)
    ac = accuracy_score(y_test,y_pred)

    if (metoda==2):
        print('Denoising Butterworth')

    #Menampilkan Hasil Performa Model

    print('--------------- ...')
    print("Akurasi KNN-", knn * 100, ' %')
    print("akurasi metrik uji : ", acc_knn)
    print("sensitiviti metrik uji : ", sens_knn)
    print("spesifisiti metrik uji : ", spes_knn)

    tombol["state"] = NORMAL
    print('Selesai ...')
    print('--------------- ...')


#Membuat GUI
var1 = StringVar()
label1 = Label(window, textvariable=var1, relief=RAISED ,width = 107,bg ='cyan'  )

var1.set("Metode denoising : ")
label1.place(x=20, y=25)
v0=IntVar()
v0.set(2)
r2=Radiobutton(window, text="Butterworth", variable=v0,value=2)
r2.place(x=20,y=50)

var1.set("Metode denoising : ")
label1.place(x=20, y=25)
v0=IntVar()
v0.set(2)
r2=Radiobutton(window, text="Butterworth", variable=v0,value=2)
r2.place(x=20,y=50)

var2 = StringVar()
label2 = Label(window, textvariable=var2, relief=RAISED ,width = 107,bg ='cyan'  )

var2.set("Fitur yang dipilih : ")
label2.place(x=20, y=100)
 
v1 = IntVar()
v2 = IntVar()
v3 = IntVar()
v4 = IntVar()
v5 = IntVar()
v6 = IntVar()
v7 = IntVar()
C1 = Checkbutton(window, text = "ESH", variable = v1)
C2 = Checkbutton(window, text = "MAD", variable = v2)
C3 = Checkbutton(window, text = "Kurtosis", variable = v3)
C4 = Checkbutton(window, text = "Skewness", variable = v4)
C5 = Checkbutton(window, text = "VLF", variable = v5)
C6 = Checkbutton(window, text = "LF", variable = v6)
C7 = Checkbutton(window, text = "HF ", variable = v7)
C1.place(x=20, y=125)
C2.place(x=180, y=125)
C3.place(x=340, y=125)
C4.place(x=500, y=125)
C5.place(x=660, y=125)
C6.place(x=20, y=150)
C7.place(x=180, y=150)

tombol = Button(window,
                   text="RUN Klasifikasi",
                   command=tombol_klik,bg='blue',fg='white')
tombol.place(x=350, y=200)

window.title('Klasifikasi PPG Machine Learning')
window.geometry("800x300+10+10")
window.mainloop()

