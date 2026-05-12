import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import csv
from datetime import datetime
import os
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ============= НАСТРОЙКИ =============
HISTORY_SIZE = 10


def save_csv_with_encoding(filename, data_rows):
    # Сохраняем в UTF-8 (обычный)
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data_rows)

    # Сохраняем для Excel
    filename_excel = filename.replace('.csv', '_excel.csv')
    with open(filename_excel, 'w', newline='', encoding='windows-1251', errors='ignore') as f:
        writer = csv.writer(f)
        writer.writerows(data_rows)

    return filename_excel


# ============= ФУНКЦИЯ ДЛЯ РИСОВАНИЯ РУССКОГО ТЕКСТА =============
def put_russian_text(img, text, position, font_size=20, color=(255, 255, 255)):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)

    font = None
    font_paths = [
        "arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Arial.ttf"
    ]

    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except:
            continue

    if font is None:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ============= ИНИЦИАЛИЗАЦИЯ =============
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

history = {
    "toe_raise": deque(maxlen=HISTORY_SIZE),
    "pushup": deque(maxlen=HISTORY_SIZE),
    "squat": deque(maxlen=HISTORY_SIZE)
}


# ============= ФУНКЦИИ ДЛЯ ВЫЧИСЛЕНИЙ =============
def calculate_angle(a, b, c):
    # Вычисляет угол между тремя точками (a-b-c) в градусах
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


# ============= АНАЛИЗ УПРАЖНЕНИЙ =============

def analyze_toe_raise(landmarks, frame_height, history_list):
    # Анализ упражнения: подъём на носочки
    feedback = []
    detailed_tips = []

    left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

    left_distance = abs(left_shoulder[1] - left_heel[1]) * frame_height
    right_distance = abs(right_shoulder[1] - right_heel[1]) * frame_height
    avg_distance = (left_distance + right_distance) / 2

    history_list.append(avg_distance)

    if not hasattr(analyze_toe_raise, "baseline_distance"):
        analyze_toe_raise.baseline_distance = avg_distance
        detailed_tips.append("КАЛИБРОВКА: запомнено исходное положение")

    baseline = analyze_toe_raise.baseline_distance
    if baseline > 0:
        lift_percent = ((avg_distance - baseline) / baseline) * 100
    else:
        lift_percent = 0

    if lift_percent < 0.4:
        feedback.append("подымайся")
        detailed_tips.append(f"ПОДНИМАЙТЕСЬ ВЫШЕ! +{lift_percent:.1f}% (цель 5-10%)")
    elif lift_percent < 0.8:
        feedback.append("повыше")
        detailed_tips.append(f"СТАРАЙТЕСЬ ВЫШЕ. Сейчас +{lift_percent:.1f}%")
    elif lift_percent < 1.2:
        feedback.append("ХОРОШО")
        detailed_tips.append(f"ХОРОШАЯ ВЫСОТА! +{lift_percent:.1f}%")
    elif lift_percent < 2:
        feedback.append("ОТЛИЧНО!")
        detailed_tips.append(f"ОТЛИЧНАЯ АМПЛИТУДА! +{lift_percent:.1f}%")
    else:
        feedback.append("СУПЕР!")
        detailed_tips.append(f"МАКСИМАЛЬНЫЙ ПОДЪЁМ! +{lift_percent:.1f}%")

    asymmetry = abs(left_distance - right_distance)
    if asymmetry > 15:
        feedback.append("НЕСИММЕТРИЧНО")
        if left_distance < right_distance:
            detailed_tips.append(f"ЛЕВАЯ СТОРОНА НИЖЕ на {asymmetry:.0f} пикс")
        else:
            detailed_tips.append(f"ПРАВАЯ СТОРОНА НИЖЕ на {asymmetry:.0f} пикс")

    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    body_angle = calculate_angle(left_shoulder, left_hip, [left_hip[0], left_hip[1] + 1])

    if body_angle < 170:
        feedback.append("НАКЛОН КОРПУСА")
        detailed_tips.append("СПИНА ПРЯМАЯ! Не наклоняйтесь")

    return feedback, detailed_tips, lift_percent


def analyze_pushup(landmarks, frame_height, history_list):
    # Анализ упражнения: отжимания от пола
    feedback = []
    detailed_tips = []

    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    avg_angle = (left_angle + right_angle) / 2

    history_list.append(avg_angle)

    if avg_angle > 160:
        feedback.append("приступай")
        detailed_tips.append("ОПУСКАЙТЕСЬ НИЖЕ! Цель: угол 90°")
    elif avg_angle > 130:
        feedback.append("НЕДОСТАТОЧНАЯ ГЛУБИНА")
        detailed_tips.append("ОПУСТИТЕСЬ НИЖЕ, цель 90°")
    elif avg_angle < 70:
        feedback.append("СЛИШКОМ ГЛУБОКО")
        detailed_tips.append("НЕ ОПУСКАЙТЕСЬ ТАК НИЗКО!")
    elif avg_angle < 95:
        feedback.append("ОТЛИЧНО!")
        detailed_tips.append("ИДЕАЛЬНАЯ ГЛУБИНА! Угол 90°")
    else:
        feedback.append("ХОРОШО")
        detailed_tips.append("ХОРОШАЯ ГЛУБИНА")

    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    body_alignment = calculate_angle(left_shoulder, left_hip, left_knee)

    if body_alignment < 160 or body_alignment > 180:
        feedback.append("ПРОГИБ/ВЫГИБ СПИНЫ")
        detailed_tips.append("ДЕРЖИТЕ СПИНУ ПРЯМОЙ!")

    if abs(left_angle - right_angle) > 15:
        feedback.append("АСИММЕТРИЯ РУК")
        detailed_tips.append("СТАРАЙТЕСЬ ОПУСКАТЬСЯ РАВНОМЕРНО")

    return feedback, detailed_tips, avg_angle


def analyze_squat(landmarks, frame_height, history_list):
    # Анализ упражнения: полуприсед
    feedback = []
    detailed_tips = []

    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    left_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_angle = calculate_angle(right_hip, right_knee, right_ankle)
    avg_angle = (left_angle + right_angle) / 2

    history_list.append(avg_angle)

    if avg_angle > 150:
        feedback.append("приседай")
        detailed_tips.append("ПРИСЕДАЙТЕ ГЛУБЖЕ! Цель: угол 90°")
    elif avg_angle > 120:
        feedback.append("НЕДОСТАТОЧНАЯ ГЛУБИНА")
        detailed_tips.append("ОПУСТИТЕСЬ НИЖЕ")
    elif avg_angle < 70:
        feedback.append("СЛИШКОМ ГЛУБОКО")
        detailed_tips.append("НЕ ПРИСЕДАЙТЕ ТАК НИЗКО!")
    elif avg_angle < 100:
        feedback.append("ОТЛИЧНО!")
        detailed_tips.append("ИДЕАЛЬНЫЙ УГОЛ! 90°")
    else:
        feedback.append("ХОРОШО")
        detailed_tips.append("ХОРОШИЙ УГОЛ")

    left_foot = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
    knee_over_toe_left = left_knee[0] - left_foot[0]

    if knee_over_toe_left > 0.05:
        feedback.append("КОЛЕНИ ВЫХОДЯТ ЗА НОСКИ")
        detailed_tips.append("ОТВЕДИТЕ ТАЗ НАЗАД!")

    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    back_angle = calculate_angle(left_shoulder, left_hip, left_knee)

    if back_angle < 140:
        feedback.append("НАКЛОН КОРПУСА ВПЕРЁД")
        detailed_tips.append("ДЕРЖИТЕ СПИНУ ПРЯМОЙ!")

    return feedback, detailed_tips, avg_angle


# ============= ГРАФИЧЕСКИЙ ИНТЕРФЕЙС =============

class ExerciseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Тренер по ЛФК")
        self.root.geometry("700x750")
        self.root.resizable(False, False)

        # Переменные для хранения выбора
        self.exercise_var = tk.StringVar(value="squat")
        self.source_var = tk.StringVar(value="camera")
        self.video_path = None

        self.setup_ui()

    def setup_ui(self):
        # Заголовок
        title_label = tk.Label(self.root, text="🏋️‍♂️ AI ТРЕНЕР ПО ЛФК 🏋️‍♀️",
                               font=("Arial", 18, "bold"), fg="#2c3e50")
        title_label.pack(pady=15)

        # Разделитель
        ttk.Separator(self.root, orient='horizontal').pack(fill='x', padx=20, pady=10)

        # Блок выбора упражнения
        exercise_frame = tk.LabelFrame(self.root, text="ВЫБЕРИТЕ УПРАЖНЕНИЕ",
                                       font=("Arial", 12, "bold"), padx=15, pady=10)
        exercise_frame.pack(fill='x', padx=20, pady=10)

        # Радиокнопки для упражнений
        exercises = [
            ("🦶 ПОДЪЁМ НА НОСОЧКИ", "toe_raise"),
            ("💪 ОТЖИМАНИЯ ОТ ПОЛА", "pushup"),
            ("🦵 ПОЛУПРИСЕД", "squat")
        ]

        for text, value in exercises:
            rb = tk.Radiobutton(exercise_frame, text=text, variable=self.exercise_var,
                                value=value, font=("Arial", 11))
            rb.pack(anchor='w', pady=3)

        # Блок выбора источника
        source_frame = tk.LabelFrame(self.root, text="ВЫБЕРИТЕ ИСТОЧНИК ВИДЕО",
                                     font=("Arial", 12, "bold"), padx=15, pady=10)
        source_frame.pack(fill='x', padx=20, pady=10)

        sources = [
            ("📁 ЗАГРУЗИТЬ ИЗ ФАЙЛА", "file"),
            ("📷 ВЕБ-КАМЕРА", "camera")
        ]

        for text, value in sources:
            rb = tk.Radiobutton(source_frame, text=text, variable=self.source_var,
                                value=value, font=("Arial", 11),
                                command=self.on_source_change)
            rb.pack(anchor='w', pady=3)

        # Блок выбора файла (изначально скрыт)
        self.file_frame = tk.Frame(self.root)
        self.file_label = tk.Label(self.file_frame, text="Выберите видео файл:", font=("Arial", 10))
        self.file_label.pack(pady=5)

        self.file_button = tk.Button(self.file_frame, text="📂 ОБЗОР",
                                     command=self.select_video_file,
                                     bg="#3498db", fg="white", font=("Arial", 10, "bold"),
                                     relief="raised", bd=2, padx=10, pady=5)
        self.file_button.pack(pady=5)

        self.file_info = tk.Label(self.file_frame, text="Файл не выбран",
                                  font=("Arial", 9), fg="red")
        self.file_info.pack(pady=5)

        # Блок с информацией о выборе
        info_frame = tk.LabelFrame(self.root, text="ИНФОРМАЦИЯ",
                                   font=("Arial", 10, "bold"), padx=15, pady=10)
        info_frame.pack(fill='x', padx=20, pady=10)

        self.info_text = tk.Text(info_frame, height=4, width=50, font=("Arial", 9),
                                 bg="#f0f0f0", relief="sunken")
        self.info_text.pack()
        self.update_info()

        # Кнопки управления
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)

        self.start_button = tk.Button(button_frame, text="▶️ НАЧАТЬ ТРЕНИРОВКУ",
                                      command=self.start_training,
                                      bg="#27ae60", fg="white", font=("Arial", 14, "bold"),
                                      relief="raised", bd=3, padx=30, pady=15)
        self.start_button.pack(side='left', padx=10)

        cancel_button = tk.Button(button_frame, text="❌ ВЫХОД",
                                  command=self.root.quit,
                                  bg="#e74c3c", fg="white", font=("Arial", 14, "bold"),
                                  relief="raised", bd=3, padx=30, pady=15)
        cancel_button.pack(side='left', padx=10)

    def on_source_change(self):
        """Обработка изменения источника видео"""
        if self.source_var.get() == "file":
            self.file_frame.pack(fill='x', padx=20, pady=5)
        else:
            self.file_frame.pack_forget()
            self.video_path = None
        self.update_info()

    def select_video_file(self):
        """Выбор видео файла через диалоговое окно"""
        file_path = filedialog.askopenfilename(
            title="Выберите видео файл",
            filetypes=[("Видео файлы", "*.mp4 *.avi *.mov *.mkv"), ("Все файлы", "*.*")]
        )
        if file_path:
            self.video_path = file_path
            filename = os.path.basename(file_path)
            self.file_info.config(text=f"✅ {filename}", fg="green")
        else:
            self.video_path = None
            self.file_info.config(text="❌ Файл не выбран", fg="red")
        self.update_info()

    def update_info(self):
        """Обновление информационного блока"""
        self.info_text.delete('1.0', tk.END)

        exercise_names = {
            "toe_raise": "ПОДЪЁМ НА НОСОЧКИ",
            "pushup": "ОТЖИМАНИЯ ОТ ПОЛА",
            "squat": "ПОЛУПРИСЕД"
        }

        exercise = exercise_names.get(self.exercise_var.get(), "Не выбрано")
        source = "ВЕБ-КАМЕРА" if self.source_var.get() == "camera" else "ФАЙЛ"

        info_text = f"📋 ВЫБРАНО:\n\n"
        info_text += f"Упражнение: {exercise}\n"
        info_text += f"Источник: {source}\n"

        if self.source_var.get() == "file":
            if self.video_path:
                info_text += f"Файл: {os.path.basename(self.video_path)}\n"
            else:
                info_text += f"Файл: НЕ ВЫБРАН!\n"

        info_text += f"\n▶️ Нажмите 'НАЧАТЬ ТРЕНИРОВКУ' для старта"

        self.info_text.insert('1.0', info_text)

    def start_training(self):
        """Запуск тренировки с выбранными параметрами"""
        # Проверка для файлового режима
        if self.source_var.get() == "file" and not self.video_path:
            messagebox.showerror("Ошибка", "Пожалуйста, выберите видео файл!")
            return

        # Получаем название упражнения
        exercise_names = {
            "toe_raise": "ПОДЪЁМ НА НОСОЧКИ",
            "pushup": "ОТЖИМАНИЯ ОТ ПОЛА",
            "squat": "ПОЛУПРИСЕД"
        }

        exercise_name = exercise_names[self.exercise_var.get()]

        # Скрываем главное окно
        self.root.withdraw()

        # Запускаем тренировку
        try:
            process_video(
                self.exercise_var.get(),
                exercise_name,
                self.source_var.get(),
                self.video_path
            )
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка:\n{str(e)}")
        finally:
            # Показываем главное окно после завершения
            self.root.deiconify()


# ============= ОСНОВНАЯ ФУНКЦИЯ ОБРАБОТКИ ВИДЕО =============
def process_video(exercise_key, exercise_name, source_type, video_path=None):
    # открытие источника видео
    if source_type == "camera":
        cap = cv2.VideoCapture(0)
        time.sleep(2)  # Уменьшил задержку для GUI
        if not cap.isOpened():
            print("Ошибка: не удалось открыть веб-камеру")
            return
        print("Веб-камера запущена. Нажмите ESC для выхода")
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Ошибка: не удалось открыть видео {video_path}")
            return
        print(f"Видео файл: {video_path}")

    # создаёт статистику
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_filename = f"training_stats_{exercise_name}_{timestamp}.csv"

    csv_rows = []
    csv_rows.append(["Кадр", "Время(мс)", "Показатель", "Оценка", "Рекомендации"])

    cv2.namedWindow('AI Тренер по ЛФК', cv2.WINDOW_NORMAL)

    # словарь упражнений
    exercises = {
        "toe_raise": {"analyzer": analyze_toe_raise, "history": history["toe_raise"], "color": (0, 255, 0)},
        "pushup": {"analyzer": analyze_pushup, "history": history["pushup"], "color": (0, 255, 255)},
        "squat": {"analyzer": analyze_squat, "history": history["squat"], "color": (255, 0, 255)}
    }

    current = exercises[exercise_key]
    frame_counter = 0
    all_metrics = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("\nВИДЕО ЗАКОНЧИЛОСЬ!")
            break

        frame_counter += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_height, frame_width = frame_bgr.shape[:2]

        if result.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame_bgr,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            feedback_levels, detailed_tips, metric = current["analyzer"](
                result.pose_landmarks.landmark,
                frame_height,
                current["history"]
            )

            all_metrics.append(metric)

            assessment = feedback_levels[0] if feedback_levels else "Нет оценки"
            tips_short = detailed_tips[0][:50] if detailed_tips else ""

            # сохраняет в список (вместо прямой записи в файл)
            csv_rows.append([frame_counter, int(cap.get(cv2.CAP_PROP_POS_MSEC)),
                             f"{metric:.2f}", assessment, tips_short])

            # ============= ОТРИСОВКА ИНТЕРФЕЙСА =============
            overlay = frame_bgr.copy()
            cv2.rectangle(overlay, (10, 10), (frame_width - 10, 300), (0, 0, 0), -1)
            frame_bgr = cv2.addWeighted(overlay, 0.65, frame_bgr, 0.35, 0)

            frame_bgr = put_russian_text(frame_bgr, f"УПРАЖНЕНИЕ: {exercise_name}", (20, 45), 22, current['color'])

            if feedback_levels:
                level_text = f"ОЦЕНКА: {feedback_levels[0]}"
                if "КРИТИЧЕСКАЯ" in level_text:
                    level_color = (0, 0, 255)
                elif "НИЖЕ" in level_text or "НЕДОСТАТОЧНАЯ" in level_text or "МЕЛКО" in level_text:
                    level_color = (0, 165, 255)
                elif "ОТЛИЧНО" in level_text:
                    level_color = (0, 255, 255)
                else:
                    level_color = (0, 255, 0)
                frame_bgr = put_russian_text(frame_bgr, level_text, (20, 85), 18, level_color)

            if exercise_key == "toe_raise":
                metric_text = f"Высота подъёма: {metric:.2f}%"
            else:
                metric_text = f"Угол: {metric:.1f}°"
            frame_bgr = put_russian_text(frame_bgr, metric_text, (20, 115), 16, (255, 255, 255))

            y_offset = 155
            frame_bgr = put_russian_text(frame_bgr, "РЕКОМЕНДАЦИИ:", (20, y_offset), 16, (255, 200, 100))
            y_offset += 25

            for i, tip in enumerate(detailed_tips[:5]):
                if "КРИТИЧЕСКАЯ" in tip or "ОШИБКА" in tip:
                    color = (0, 0, 255)
                elif "ВНИМАНИЕ" in tip or "ОСТОРОЖНО" in tip:
                    color = (0, 165, 255)
                elif "ОТЛИЧНО" in tip or "ИДЕАЛЬНАЯ" in tip:
                    color = (0, 255, 255)
                elif "ХОРОШО" in tip or "СПИНА ПРЯМАЯ" in tip:
                    color = (0, 255, 0)
                else:
                    color = (200, 200, 200)
                frame_bgr = put_russian_text(frame_bgr, tip[:55], (20, y_offset + i * 24), 14, color)

            frame_bgr = put_russian_text(frame_bgr, f"Кадр: {frame_counter}", (frame_width - 150, frame_height - 20),
                                         12, (150, 150, 150))
            frame_bgr = put_russian_text(frame_bgr, f"Сохранено: {stats_filename}", (20, frame_height - 20), 12,
                                         (150, 150, 150))

        else:
            frame_bgr = put_russian_text(frame_bgr, "ЧЕЛОВЕК НЕ ОБНАРУЖЕН",
                                         (frame_width // 2 - 150, frame_height // 2), 20, (0, 0, 255))

        cv2.imshow('AI Тренер по ЛФК', frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    # сохраняет csv в двух кодировках
    excel_file = save_csv_with_encoding(stats_filename, csv_rows)

    # Итоговая статистика
    if all_metrics:
        print("\n" + "=" * 50)
        print("       ИТОГОВАЯ СТАТИСТИКА")
        print("=" * 50)
        print(f"Максимум: {np.max(all_metrics):.2f}")
        print(f"Минимум: {np.min(all_metrics):.2f}")
        print(f"Всего кадров: {len(all_metrics)}")
        print(f"Статистика сохранена в файлы:")
        print(f"  - {stats_filename} (UTF-8)")
        print(f"  - {excel_file} (для Excel)")
        print("=" * 50)

    cap.release()
    cv2.destroyAllWindows()
    print("\nРабота программы завершена")


# ============= ЗАПУСК =============
if __name__ == "__main__":
    root = tk.Tk()
    app = ExerciseApp(root)
    root.mainloop()