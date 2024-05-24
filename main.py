import librosa
import numpy as np
import matplotlib.pyplot as plt
import speech_recognition as sr
from pydub import AudioSegment
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pydub.playback import play
from pydub.utils import mediainfo
from sklearn.metrics import accuracy_score, f1_score
import pyaudio
import wave

AudioSegment.converter = "C:\\ffmpeg\\bin\\ffmpeg.exe"

def load_audio(file_path):
    y, sr = librosa.load(file_path)
    return y, sr

def create_histogram(y, sr):
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(chroma_stft.flatten(), bins=50)
    ax.set_title('Kroma Özelliği Histogramı')
    return fig

def extract_mfcc(y, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

def train_person_identifier(audio_files, labels):
    features = [extract_mfcc(librosa.load(file)[0], librosa.load(file)[1]) for file in audio_files]
    X = np.array(features)
    y = LabelEncoder().fit_transform(labels)
    model = SVC(kernel='linear')
    model.fit(X, y)
    return model, LabelEncoder().fit(labels)

def predict_speaker(model, encoder, audio_file):
    y, sr = load_audio(audio_file)
    mfcc = extract_mfcc(y, sr)
    prediction = model.predict([mfcc])
    return encoder.inverse_transform(prediction)

def convert_to_wav(file_path):
    audio = AudioSegment.from_file(file_path)
    wav_path = "temp.wav"
    audio.export(wav_path, format="wav")
    return wav_path

def recognize_speech(audio_file):
    recognizer = sr.Recognizer()
    wav_file = convert_to_wav(audio_file)
    with sr.AudioFile(wav_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language='tr-TR')
            return text
        except sr.UnknownValueError:
            return "Sesi anlaşılamadı"
        except sr.RequestError as e:
            return f"Sonuçlar alınamadı; {e}"

def count_words(text):
    words = text.split()
    return len(words)

def record_audio(filename, duration=5, sample_rate=44100, channels=1):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=channels, rate=sample_rate, input=True, frames_per_buffer=1024)

    frames = []
    for i in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Ses İşleme Projesi")
        self.root.geometry("600x400")
        self.create_widgets()

        # Eğitim verileri ve etiketler
        self.audio_files = ['audio_files/emirhanses.wav', 'audio_files/bekirses.wav']
        self.labels = ['emirhan', 'bekir']
        self.model, self.encoder = train_person_identifier(self.audio_files, self.labels)

    def process_audio(self, file_path):
        try:
            y, sr = load_audio(file_path)
            fig = create_histogram(y, sr)
            self.show_histogram(fig)

            prediction = predict_speaker(self.model, self.encoder, file_path)
            recognized_text = recognize_speech(file_path)
            word_count = count_words(recognized_text)

            # Tanımlayıcıyı etiketle değiştir
            predicted_speaker = prediction[0]

            # Tahmin edilen etiket için doğru ve yanlış değerleri belirleyin
            true_label = 'emirhan' if 'emirhan' in file_path else 'bekir'
            y_true = [self.encoder.transform([true_label])[0]]
            y_pred = [self.encoder.transform([predicted_speaker])[0]]

            # ACC ve F1 değerlerini hesapla
            acc = accuracy_score(y_true, y_pred) * 100
            f1 = f1_score(y_true, y_pred, zero_division=1) * 100

            result_text = f"Tanınan Metin: {recognized_text}\nKelime Sayısı: {word_count}\nTahmin Edilen Konuşmacı: {predicted_speaker}\nACC: {acc:.2f}%\nF1 Skoru: {f1:.2f}%"
            self.result_label.config(text=result_text)
        except Exception as e:
            self.result_label.config(text=f"Bir hata oluştu: {e}")

    def create_widgets(self):
        frame = ttk.Frame(self.root, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        self.label = ttk.Label(frame, text="Ses Dosyasını Seçin:")
        self.label.pack(pady=10)

        self.open_button = ttk.Button(frame, text="Dosya Aç", command=self.load_file)
        self.open_button.pack(pady=10)

        self.record_button = ttk.Button(frame, text="Sesimi Algıla", command=self.record_and_process_audio)
        self.record_button.pack(pady=10)

        self.result_label = ttk.Label(frame, text="", wraplength=400)
        self.result_label.pack(pady=10)

    def load_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Ses Dosyaları", "*.wav;*.mp3")])
        if self.file_path:
            self.process_audio(self.file_path)

    def record_and_process_audio(self):
        recorded_file = "live_recording.wav"
        record_audio(recorded_file, duration=5)
        self.process_audio(recorded_file)

    def show_histogram(self, fig):
        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = App(root)
        root.mainloop()
    except Exception as e:
        print(f"Program çalışırken bir hata oluştu: {e}")
