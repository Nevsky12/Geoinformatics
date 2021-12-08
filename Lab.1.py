import requests
import numpy as np
from pyorbital.orbital import Orbital
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


# Собственный класс ошибок для проверки формата даты
class CheckDate(ValueError):
    # Конструктор класса, наследуемого от класса ValueError
    def __init__(self, word, message="Invalid date!"):
        self.message = message
        self.word = word
        super().__init__(self.message)

    # Сообщение пользователю в случае ошибки
    def __str__(self):
        return f'{self.message} -> {self.word}'


# Класс спутника
class Satellite_data:

    # По строкам TLE с помощью функции Orbital определяются широта, долгота и выоста спутника над Землёй
    def __set__(self, tle_1, tle_2, utc_time):
        orb = Orbital("N", line1=tle_1, line2=tle_2)
        lon, lat, height_st = orb.get_lonlatalt(utc_time)
        self.__longitude = lon
        self.__latitude = lat
        self.__height_sat = height_st

    # Геттер координат
    def get_satellite_coords(self):
        return self.__longitude, self.__latitude, self.__height_sat

    # Геттер всех данных спутника
    def get_satellite_data(self):
        return self.__data

    # Конструктор всех данных спутник из файла с TLE
    def __init__(self, file_with_tle, satellite_name):
        record = requests.get(file_with_tle, stream=True)
        open('TLE.txt', 'wb').write(record.content)
        file = open('TLE.txt', 'r')
        temporary = file.read().split("\n")[:-1]
        for i in range(len(temporary)):
            if temporary[i] == satellite_name:
                self.__data = [satellite_name, temporary[i + 1], temporary[i + 2]]
        self.__longitude = 0.
        self.__latitude = 0.
        self.__height_sat = 0.


# Класс со всеми используемыми математическими операциями
class Math:

    # Перевод из градусов в радианы
    @staticmethod
    def to_radians(value):
        return (value * np.pi) / 180

    # Перевод из радиан в градусы
    @staticmethod
    def to_degrees(value):
        return (value * 180) / np.pi

    # Перевод из сферических координат в декартовы
    @staticmethod
    def to_decarts(r, theta, phi):
        x = r * np.cos(theta) * np.cos(phi)
        y = r * np.cos(theta) * np.sin(phi)
        z = r * np.sin(theta)
        return x, y, z

    # Нормирование вектора
    @staticmethod
    def unit_vector(vector):
        return vector / np.linalg.norm(vector)

    # Вычисление угла между векторами
    @staticmethod
    def angle_V1_V2(V1, V2):
        v1_u = Math.unit_vector(V1)
        v2_u = Math.unit_vector(V2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    # Задание вектора
    @staticmethod
    def set_V(x1, y1, z1, x2, y2, z2):
        V = np.array([x2 - x1, y2 - y1, z2 - z1])
        V_length = np.sqrt(V.dot(V))
        return V, V_length

    # Расстояние до касательной к сфере плоскости от точки вне сферы
    @staticmethod
    def dist_to_PL(x1, y1, z1, x2, y2, z2):
        D = -x1 ** 2 - y1 ** 2 - z1 ** 2
        dif = (x1 * x2 + y1 * y2 + z1 * z2 + D) / ((x1 ** 2 + y1 ** 2 + z1 ** 2) ** 0.5)
        return dif


# Класс - хранилище промежуточных данных и данных для построения траекторий спутника
class Storage:
    ax = property()
    ay = property()
    az = property()
    at = property()
    ael = property()
    aaz = property()
    auel = property()
    auaz = property()
    teld = property()
    aut = property()
    tazd = property()
    ttd = property()

    # Конструктор хранилища
    def __init__(self, ):
        self._array_x = np.zeros(0, float)
        self._array_y = np.zeros(0, float)
        self._array_z = np.zeros(0, float)
        self._array_time = np.zeros(0, float)
        self._array_elevation = np.zeros(0, float)
        self._array_azimuth = np.zeros(0, float)
        self._array_usual_elevation = []
        self._array_usual_azimuth = []
        self._temporary_elevation_data = []
        self._array_usual_time = []
        self._temporary_azimuth_data = []
        self._temporary_time_data = []

    # Геттер x-ой координаты спутника
    @ax.getter
    def ax(self):
        return self._array_x

    # Сеттер x-ой координаты спутника
    @ax.setter
    def ax(self, value):
        self._array_x = np.append(self._array_x, value)

    # Геттер y-ой координаты спутника
    @ay.getter
    def ay(self):
        return self._array_y

    # Сеттер y-ой координаты спутника
    @ay.setter
    def ay(self, value):
        self._array_y = np.append(self._array_y, value)

    # Геттер массива, хранящего времена пролета спутника с шагом в 1 минуту
    @at.getter
    def at(self):
        return self._array_time

    # Сеттер массива времён
    @at.setter
    def at(self, value):
        self._array_time = np.append(self._array_time, value)

    # Геттер z-ой координаты спутника
    @az.getter
    def az(self):
        return self._array_z

    # Сеттер z-ой координаты спутника
    @az.setter
    def az(self, value):
        self._array_z = np.append(self._array_z, value)

    # Геттер массива, хранящего угол элевации спутника в каждый момент времени
    @ael.getter
    def ael(self):
        return self._array_elevation

    # Сеттер массива углов элевации спутника
    @ael.setter
    def ael(self, value):
        self._array_elevation = np.append(self._array_elevation, value)

    # Геттер массива, хранящего азимут спутника в каждый момент времени
    @aaz.getter
    def aaz(self):
        return self._array_azimuth

    # Сеттер массива азимутов спутника
    @aaz.setter
    def aaz(self, value):
        self._array_azimuth = np.append(self._array_azimuth, value)

    # Геттер массива, хранящего отобранные значения углов элевации
    @auel.getter
    def auel(self):
        return self._array_usual_elevation

    # Сеттер массива отобранных углов элевации
    @auel.setter
    def auel(self, value):
        self._array_usual_elevation.append(value)

    # Геттер массива, хранящего отобранные значения азимутов
    @auaz.getter
    def auaz(self):
        return self._array_usual_azimuth

    # Сеттер массива отобранных азимутов
    @auaz.setter
    def auaz(self, value):
        self._array_usual_azimuth.append(value)

    # Геттер массива, хранящего промежуточные результаты вычисления углов элевации
    @teld.getter
    def teld(self):
        return self._temporary_elevation_data

    # Сеттер массива временных углов элевации
    @teld.setter
    def teld(self, value):
        if not value:
            self._temporary_elevation_data = []
        else:
            self._temporary_elevation_data.append(value)

    # Геттер массива, хранящего отобранные значения времени
    @aut.getter
    def aut(self):
        return self._array_usual_time

    # Сеттер массива отобранных значений времени
    @aut.setter
    def aut(self, value):
        self._array_usual_time.append(value)

    # Геттер массива, хранящего промежуточные результаты вычисления азимута
    @tazd.getter
    def tazd(self):
        return self._temporary_azimuth_data

    # Сеттер массива временных азимутов
    @tazd.setter
    def tazd(self, value):
        if not value:
            self._temporary_azimuth_data = []
        else:
            self._temporary_azimuth_data.append(value)

    # Геттер массива, хранящего промежуточные значения времени
    @ttd.getter
    def ttd(self):
        return self._temporary_time_data

    # Сеттер массива промежуточных значений времени
    @ttd.setter
    def ttd(self, value):
        if not value:
            self._temporary_time_data = []
        else:
            self._temporary_time_data.append(value)


# Параметры ЛК:
h_LK = 0.197  # высота ЛК над морем
R = 6378.1375  # радиус Земли
latitude_LK_R = Math.to_radians(55.928895)  # широта ЛК
longitude_LK_R = Math.to_radians(37.521498)  # долгота ЛК
x_LK, y_LK, z_LK = Math.to_decarts(R + h_LK, latitude_LK_R, longitude_LK_R)  # координаты ЛК в декартовой системе
D = -(x_LK ** 2 + y_LK ** 2 + z_LK ** 2)  # свободный член в уравнении касательной плоскости
z_P = -D / z_LK  # z-ая координата пересечения плоскости с осью апликат Земли
y_P = -D / y_LK
x_P = -D / x_LK

# Направляющие векторы, для определения нужных параметров спутника:
North_V, North_V_L = Math.set_V(0, 0, z_P, x_LK, y_LK, z_LK)  # вектор, указывающий на север в данной плоскости
Normal_V, Normal_V_L = Math.set_V(0, 0, 0, x_LK, y_LK, z_LK)  # вектор нормали к плоскости
East_V = np.cross(North_V, Normal_V)
East_V_L = np.sqrt(East_V.dot(East_V))

# Создание экземпляров:
storage = Storage()
Satellite = Satellite_data("https://celestrak.com/NORAD/elements/active.txt", "NOAA 19                 ")
satellite_data = Satellite.get_satellite_data()
print(satellite_data)

# Обработка пользовательской ошибки
try:
    # ввод начальной даты(local):
    print("Your date start MM HH DD MM YYYY: ")
    word = input()
    start_time = np.array(word.split(), dtype=int)
    start_time = datetime(start_time[4], start_time[3], start_time[2], start_time[1], start_time[0])

    # ввод конечной даты(local):
    print("Your date end MM HH DD MM YYYY: ")
    end_time = np.array(input().split(), dtype=int)
    end_time = datetime(end_time[4], end_time[3], end_time[2], end_time[1], end_time[0])

except ValueError as ve:
    raise CheckDate(word)

minutes = int((end_time - start_time).total_seconds() / 60)  # разница между датами в минутах

for i in range(minutes):

    Satellite.__set__(satellite_data[1], satellite_data[2], start_time - timedelta(hours=3))
    longitude, latitude, height = Satellite.get_satellite_coords()  # время тут переводим в utc
    start_time = start_time + timedelta(minutes=1) # шаг по времени

    latitude_radians = Math.to_radians(latitude)
    longitude_radians = Math.to_radians(longitude)

    x, y, z = Math.to_decarts(height + R, latitude_radians, longitude_radians)
    storage.ax = x
    storage.ay = y
    storage.az = z

    Sat_V, Sat_V_L = Math.set_V(x_LK, y_LK, z_LK, x, y, z)  # определение вектора, указывающего на спутник от ЛК и его длины
    dist_to_P = Math.dist_to_PL(x_LK, y_LK, z_LK, x, y, z) # расстяние от ЛК до плоскости, касательной к сфере, определяемой вектором спутника
    dist_to_Sat = ((x - x_LK) ** 2 + (y - y_LK) ** 2 + (z - z_LK) ** 2) ** 0.5  # расстояние от ЛК до спутника
    elevation = np.arcsin(dist_to_P / dist_to_Sat)  # тут может быть отрицательные значения, потому что в формуле расстояния точки от плоскости убран модуль в числителе
    azimuth = 0

    if elevation >= 0:
        Sat_V_N = [(Normal_V[0] * np.dot(Normal_V, Sat_V)) / (Normal_V_L ** 2),
                   (Normal_V[1] * np.dot(Normal_V, Sat_V)) / (Normal_V_L ** 2),
                   (Normal_V[2] * np.dot(Normal_V, Sat_V)) / (
                           Normal_V_L ** 2)]  # проекция вектора спутника на вектор нормали
        Sat_V_Pr, Sat_V_Pr_L = Math.set_V(0, 0, 0, -Sat_V[0] + Sat_V_N[0], -Sat_V[1] + Sat_V_N[1],
                                          -Sat_V[2] + Sat_V_N[2])  # проекция вектора спутника на плоскость
        azimuth = Math.angle_V1_V2(Sat_V_Pr, North_V)

        if 3 * np.pi / 2 >= Math.angle_V1_V2(East_V, Sat_V_Pr) > np.pi / 2: # если угол между проекцией вектора спутника на плоскость и направлением на восток в пределах (90, 270], то спутник находится ниже экватора
            azimuth = 2 * np.pi - azimuth
    elevation = Math.to_degrees(elevation)
    storage.aaz = azimuth
    storage.at = start_time + timedelta(hours=3)  # перевод в локальное время
    storage.ael = elevation

for i in range(len(storage.at)):
    # Отбор нужных параметров времени, азимута и элевации для определения моментов пролёта спутника над ЛК
    if 180 >= storage.ael[i] >= 0:
        storage.teld = storage.ael[i]
        storage.tazd = storage.aaz[i]
        storage.ttd = storage.at[i]

    else:
        if len(storage.teld) != 0:
            storage.aut = storage.ttd
            storage.auaz = storage.tazd
            storage.auel = storage.teld

        storage.teld = []  # используются списки из списков,чтобы выводить данные из каждого списка в списки
        # отдельно(иначе если выводить данные подряд, то всё будет соединено ломаными линиями)
        storage.tazd = []
        storage.ttd = []

# Вывод параметров в моменты пролёта спутника над ЛК
for i in range(len(storage.aut)):
    print("[ Time: ", storage.aut[i][0], "] [ Azimuth: ", storage.auaz[i][0], "] [ ""Elevation: ",
          np.max(storage.auel[i]), "]")

# Зависимость угла элевации от азимута спутника в полярных координатах(проекция 3D графика траектории спутника на
# плоскость ЛК)
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_rlim(bottom=90, top=0)
for phi, theta in zip(storage.auaz, storage.auel):
    ax.plot(phi, theta)
fig.set_size_inches(7, 7)
plt.show()

# 3D график траектории спутника
sf = plt.figure()
ax = sf.add_subplot(111, projection='3d')
ax.plot(storage.ax, storage.ay, storage.az)
ax.scatter(x_LK, y_LK, z_LK, color='black')
sf.set_size_inches(7, 7)
plt.show()
