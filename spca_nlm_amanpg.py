"""
SPCA-NLM filter: preliminary studies

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
from sparsepca import spca
from scipy.sparse.linalg import svds
import csv

class SparsePCA:
    def __init__(self, lambda1, lambda2, k):
        self.lambda2 = lambda2
        self.lambda1 = lambda1
        self.k = k
        self.components_ = None

    def fit(self, X):
        components = spca(conjunto_patches, lambda1, lambda2, k=k)
        self.components_ = components['loadings']

    def transform(self, X):
        X = np.asarray(X)
        X_transformed = X - np.mean(X, axis=0)
        return np.dot(X_transformed, self.components_)
    


# Para evitar warning de divisão por zero
warnings.simplefilter(action='ignore')

########################################################
# Início do script
########################################################

if len(sys.argv) < 2:
    print("Usage: python spca_nlm_amanpg.py <image_name>")
    sys.exit(1)

img_name = sys.argv[1]
img = skimage.io.imread(img_name)

# Checa se imagem é monocromática
if len(img.shape) > 2:
    if img.shape[2] == 3:
        img = rgb2gray(img)   # valores convertidos ficam entre 0 e 1    
    elif img.shape[2] == 4:
        img = rgb2gray(rgba2rgb(img))   # valores convertidos ficam entre 0 e 1
    img = 255*img

img = transform.resize(img, (256, 256), anti_aliasing=True)  
img = img.astype(np.uint8) # converte para uint8
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

# Define parâmetros do filtro NLM
f = 5    # tamanho do patch (2f + 1 x 2f + 1) -> 11 x 11
t = 8    # tamanho da janela de busca (2t + 1 x 2t + 1) -> 17 x 17

# Parâmetro que controla a suavização (global)
lista_h = [70, 80, 90, 100, 110]    

psnrs_nlmspca = []
ssims_nlmspca = []
tempo_nlmspca = []

# Cria imagem de saída
filtrada = np.zeros((m, n))

# Problema de valor de contorno: replicar bordas
img_n = np.pad(ruidosa, ((f, f), (f, f)), 'symmetric')

# Para treinar o modelo podemos pular alguns patches
# Percorre imagem de 2 em 2
linhas = list(range(0, m, 2))
colunas = list(range(0, n, 2))

num_linhas = len(linhas)
num_colunas = len(colunas)

conjunto_patches = np.zeros((num_linhas*num_colunas, (2*f+1)*(2*f+1)))

# Extrai patches da imagem toda
ind = 0
for i in range(0, m, 2):
    for j in range(0, n, 2):
        im = i + f
        jn = j + f
        patch = img_n[im-f:(im+f)+1, jn-f:(jn+f)+1]
        vetor = np.reshape(patch, [1, patch.shape[0]*patch.shape[1]])
        conjunto_patches[ind, :] = vetor
        ind += 1

print('Dataset para Sparse PCA: ')
print(conjunto_patches.shape)
print()

# Aplica SPCA em dataset
k = 25
lambda1 = 0.1 * np.ones((k, 1))
lambda2 = 1
model = SparsePCA(lambda1=lambda1, lambda2=lambda2, k=k) 
# Aplica PCA em dataset
model.fit(conjunto_patches)

for h in lista_h:
    print('h = %d' %h)
    # Mede tempo de processamento
    inicio = time.time()
    # Loop principal do PCA-NLM
    print('PCA-NLM processing...')
    for i in range(m):
        #if i % 10 == 0:
            #print(i, end=' ')
        sys.stdout.flush()
        for j in range(n):
            im = i + f;   # compensar a borda adicionada artificialmente
            jn = j + f;   # compensar a borda adicionada artificialmente
            # Obtém o patch ao redor do pixel corrente
            patch_central = img_n[im-f:(im+f)+1, jn-f:(jn+f)+1]
            central = np.reshape(patch_central, [1, patch_central.shape[0]*patch_central.shape[1]])
            # Calcula as bordas da janela de busca para o pixel corrente
            rmin = max(im-t, f);  # linha inicial
            rmax = min(im+t, m+f);  # linha final
            smin = max(jn-t, f);  # coluna inicial
            smax = min(jn+t, n+f);  # coluna final
            # Calcula média ponderada
            NL = 0      # valor do pixel corrente filtrado
            Z = 0       # constante normalizadora
            # Cria dataset com patches da janela de busca como vetores
            num_elem = (rmax - rmin)*(smax - smin)
            tamanho_patch = (2*f + 1)*(2*f + 1)
            dataset = np.zeros((num_elem, tamanho_patch))
            k = 0
            pixels_busca = []
            # Loop para montar o dataset com todos os patches da janela
            for r in range(rmin, rmax):
                for s in range(smin, smax):
                    # Obtém o patch ao redor do pixel a ser comparado
                    W = img_n[r-f:(r+f)+1, s-f:(s+f)+1]                
                    dataset[k, :] = np.reshape(W, [1, W.shape[0]*W.shape[1]])
                    pixels_busca.append(img_n[r, s])
                    k = k + 1
            # Aplica PCA em dataset
            dados_pca = model.transform(dataset)                
            # Encontra o índice do elemento central em A
            indice = statistics.mode(np.where(dataset==central)[0])
            # Calcula os quadrados das distancias Euclidianas entre os patches no subespaço PCA
            meio = dados_pca[indice, :]
            matriz = repmat(meio, k, 1)
            distancias = np.sum((dados_pca - matriz)*(dados_pca - matriz), axis=1)
            # Calcula similaridades
            similaridades = np.exp(-distancias/(h**2))
            pixels = np.array(pixels_busca)
            # Normalização do pixel filtrado
            NL = sum(similaridades*pixels)
            Z = sum(similaridades)
            filtrada[i, j] = NL/Z
           
    # Tempo gasto
    fim = time.time()
    tempo = fim - inicio
    tempo_nlmspca.append(tempo)
    print('\nParâmetro h = %d' %h)    
    print('Tempo de execução: %d segundos' %(tempo))
    # Salva imagem no arquivo de saída
    # Converte para uint8 (8 bits)
    output = filtrada.astype(np.uint8) 
    # Calcula PSNR
    p = peak_signal_noise_ratio(img, output)
    psnrs_nlmspca.append(p)
    print('PSNR (SPCA-NLM): %f' %p)
    # Calcula SSIM
    s = structural_similarity(img, output)
    ssims_nlmspca.append(s)
    print('SSIM (SPCA-NLM): %f' %s)
    print()
    # Salva arquivojn
    arquivo = 'output/AMANPGNLMSPCA/' + img_name[11:-5] + '_h_' + str(h) + '.png'
    skimage.io.imsave(arquivo, output)

csv_name = 'amanpg_spcanlm_results'+ str(sigma) +'.csv'
with open(csv_name, 'a') as data_file:
    for i in range(len(lista_h)):
        csv_writer = csv.writer(data_file, delimiter=',')
        csv_writer.writerow([f, t, lista_h[i], psnrs_nlmspca[i], ssims_nlmspca[i], tempo_nlmspca[i], img_name])

print()
print('Métricas quantitativas (melhores resultados)')
print('----------------------------------------------')
print()
print('PSNR (NLM SPCA): %f' %(max(psnrs_nlmspca)))
print('SSIM (NLM SPCA): %f' %(max(ssims_nlmspca)))
