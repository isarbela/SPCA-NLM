
"""
Standard NLM filter: preliminary studies

"""
import sys
import warnings
import time
import skimage
import statistics
import skimage.io
import skimage.measure
import numpy as np
from numpy.matlib import repmat
from skimage import transform
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.filters.edges import convolve
from skimage.filters import laplace
from skimage.metrics import normalized_root_mse
from skimage.color import rgba2rgb
from skimage.color import rgb2gray
from skimage.color import gray2rgba
import csv

# Para evitar warning de divisão por zero
warnings.simplefilter(action='ignore')

'''
Non-Local Means padrão

Parâmetros:

    img: imagem ruidosa de entrada
    h: parâmetro que controla o grau de suavização (quanto maior, mais suaviza)
    f: tamanho do patch (2f + 1 x 2f + 1) -> se f = 3, então patch é 7 x 7
    t: tamanho da janela de busca (2t + 1 x 2t + 1) -> se t = 10, então janela de busca é 21 x 21

'''
def NLM(img, h, f, t):
    # Dimenssões espaciais da imagem
    m, n = img.shape
    # Cria imagem de saída
    filtrada = np.zeros((m, n))
    # Problema de valor de contorno: replicar bordas
    img_n = np.pad(ruidosa, ((f, f), (f, f)), 'symmetric')
    # Loop principal do NLM
    for i in range(m):
        for j in range(n):
            im = i + f;   # compensar a borda adicionada artificialmente
            jn = j + f;   # compensar a borda adicionada artificialmente
            # Obtém o patch ao redor do pixel corrente
            W1 = img_n[im-f:(im+f)+1, jn-f:(jn+f)+1]
            # Calcula as bordas da janela de busca para o pixel corrente (se pixel próximo das bordas, janela de busca é menor)
            rmin = max(im-t, f);  # linha inicial
            rmax = min(im+t, m+f);  # linha final
            smin = max(jn-t, f);  # coluna inicial
            smax = min(jn+t, n+f);  # coluna final
            # Calcula média ponderada
            NL = 0      # valor do pixel corrente filtrado
            Z = 0       # constante normalizadora
            # Loop para todos os pixels da janela de busca
            for r in range(rmin, rmax):
                for s in range(smin, smax):
                    # Obtém o patch ao redor do pixel a ser comparado
                    W2 = img_n[r-f:(r+f)+1, s-f:(s+f)+1]
                    # Calcula o quadrado da distância Euclidiana
                    d2 = np.sum((W1 - W2)*(W1 - W2))
                    # Calcula a medida de similaridade
                    sij = np.exp(-d2/(h**2))               
                    # Atualiza Z e NL
                    Z = Z + sij
                    NL = NL + sij*img_n[r, s]
            # Normalização do pixel filtrado
            filtrada[i, j] = NL/Z
    return filtrada


########################################################
# Início do script
########################################################

if len(sys.argv) < 2:
    print("Usage: python nlm.py <image_name>")
    sys.exit(1)

img_name = sys.argv[1]
img = skimage.io.imread(img_name)

# Checa se imagem é monocromática
if len(img.shape) > 2:
    if img.shape[2] == 3:
        img = rgb2gray(img)   # valores convertidos ficam entre 0 e 1
        print("Imagem RGB")    
    elif img.shape[2] == 4:
        img = rgb2gray(rgba2rgb(img))   # valores convertidos ficam entre 0 e 1
        print("Imagem RGBA")
    img = 255*img

img = transform.resize(img, (256, 256), anti_aliasing=True)    
img = img.astype(np.uint8) # Converte para uint8
m, n = img.shape

print('Num. linhas = %d' %m)
print('Num. colunas = %d' %n)
print()

# Variancia do ruído Gaussiano
sigma = 20
ruido = np.random.normal(0, sigma, (m, n))

# Cria imagem ruidosa
ruidosa = img + ruido

# Clipa imagem para intervalo [0, 255]
ruidosa[np.where(ruidosa > 255)] = 255
ruidosa[np.where(ruidosa < 0)] = 0
arquivo = 'output/NLM/' + img_name[11:-5] + '_ruido_' + str(sigma) + '.png'
skimage.io.imsave(arquivo, ruidosa.astype(np.uint8))

p = peak_signal_noise_ratio(img, ruidosa.astype(np.uint8))
print('PSNR Ruido (NLM padrão): %f' %p)
# Calcula SSIM
s = structural_similarity(img, ruidosa.astype(np.uint8))
print('SSIM Ruido (NLM padrão): %f' %s)
print()

# Define parâmetros do filtro NLM
f = 5    # tamanho do patch (2f + 1 x 2f + 1) -> 11 x 11
t = 8    # tamanho da janela de busca (2t + 1 x 2t + 1) -> 17 x 17

# Parâmetro que controla a suavização (importante calibrar para cada imagem)
lista_h = [70, 80, 90, 100, 110] 

# Listas para métricas
psnrs_nlm = []
ssims_nlm = []
tempo_nlm = []

# Cria imagem de saída
filtrada_padrao = np.zeros((m, n))

# Problema de valor de contorno: replicar bordas
img_n = np.pad(ruidosa, ((f, f), (f, f)), 'symmetric')

for h in lista_h:
    print('h = %d' %h)
    # Mede tempo de processamento
    inicio = time.time()
    # Filtra com NLM padrão
    print('NLM processing...')
    filtrada_padrao = NLM(ruidosa, h, f, t)
    fim = time.time()
    tempo = fim - inicio
    tempo_nlm.append(tempo)
    print('Tempo de execução: %d segundos' %(tempo))
    # Calcula PSNR
    p = peak_signal_noise_ratio(img, filtrada_padrao.astype(np.uint8))
    psnrs_nlm.append(p)
    print('PSNR (NLM padrão): %f' %p)
    # Calcula SSIM
    s = structural_similarity(img, filtrada_padrao.astype(np.uint8))
    ssims_nlm.append(s)
    print('SSIM (NLM padrão): %f' %s)
    print()
    # Salva arquivo
    arquivo = 'output/NLM/' + img_name[11:-5] + '_h_' + str(h) + '.png'
    skimage.io.imsave(arquivo, filtrada_padrao.astype(np.uint8)) 

csv_name = 'nlm_results' + str(sigma) + '.csv'
with open(csv_name, 'a') as data_file:
    for i in range(len(lista_h)):
        csv_writer = csv.writer(data_file, delimiter=',')
        csv_writer.writerow([f, t, lista_h[i], psnrs_nlm[i], ssims_nlm[i], tempo_nlm[i], img_name])

# Plota resultados na tela
print()
print('Métricas quantitativas (melhores resultados)')
print('----------------------------------------------')
print()
print('PSNR (NLM): %f' %(max(psnrs_nlm)))
print('SSIM (NLM): %f' %(max(ssims_nlm)))