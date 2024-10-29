import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy 
import scipy.fftpack as fp


imagem2 = cv2.imread(r'C:\Users\Gabriel\Desktop\Jesga\coruja.jpg')
cv2.imshow('Imagem', imagem2)
cv2.waitKey(0)  # Espera até que uma tecla seja pressionada
cv2.destroyAllWindows()

gray_image = cv2.cvtColor(imagem2,cv2.COLOR_BGR2GRAY)

cv2.imshow('Imagem', gray_image)
cv2.waitKey(0)  # Espera até que uma tecla seja pressionada
cv2.destroyAllWindows()

# Transformada Rápida de Fourier
F = fp.fft2(gray_image)
Fm = np.absolute(F)  # Calcula o módulo, já que a transformada retorna números complexos
Fm = fp.fftshift(Fm)  # Centraliza a frequência de número de onda zero

# Adicione uma pequena constante para evitar log(0) e aplique o logaritmo
Fm = np.log(Fm + 1e-8)

# Normaliza os valores de Fm para o intervalo [0, 1]
Fm = Fm / Fm.max()

# Exibir o resultado
plt.figure()
plt.title("FFT em Escala Logarítmica")
plt.imshow(Fm, cmap='gray')
plt.colorbar()  # Adiciona uma barra de cores para referência
plt.show()

