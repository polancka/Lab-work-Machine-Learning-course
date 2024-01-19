# LAB 9

##tenzorske mreže
#primerjaj svojo implementacijo kompresije v primerjavi s svd kompresijo
#izračunaj normoo

# Download a large image https://www.publicdomainpictures.net/pictures/540000/
# velka/seamless-flowers-pattern-16952868310zL.jpg, convert it to a grayscale, and
# resize it such that the height will be 4096 by keeping the aspect ratio constant. Finally,
# crop the image to the size 4096 × 4096 and convert it to a NumPy array A.

import numpy as np
import pandas as pd
from PIL import Image
from scipy.linalg import svd
import matplotlib.pyplot as plt



def process_image(input_path, output_path, target_size):
    # Open the image
    original_image = Image.open(input_path)

    # Convert the image to grayscale
    grayscale_image = original_image.convert('L')

    # Resize the image to the target size
    resized_image = grayscale_image.resize(target_size)

    # Save the processed image
    return resized_image



class MPSCompression:
    def __init__(self, image_path, bond_dimension):
        self.image_path = image_path
        self.bond_dimension = bond_dimension
        self.original_image = np.array(Image.open(image_path))
        self.compressed_mps = None
        self.decompressed_image = None

    def compress_image(self):
        # Reshape the 2D array to a tensor
        tensor = np.reshape(self.original_image, (4096, 4096))

        # Transpose the tensor
        tensor = np.transpose(tensor, (0, 1))

        # Reshape the tensor by combining even and odd indices
        tensor = np.reshape(tensor, (4096**2, 2, 2048))

        # Compress the tensor using successive SVD decompositions
        for i in range(4096):
            matrix = np.reshape(tensor[i, :, :], (2, 2048))
            u, s, v = svd(matrix, full_matrices=False)
            u = u[:, :self.bond_dimension]
            s = np.diag(s[:self.bond_dimension])
            v = v[:self.bond_dimension, :]
            tensor[i, :, :] = np.dot(u, np.dot(s, v))

        self.compressed_mps = tensor

    def decompress_image(self):
        # Decompress the tensor using the inverse of the compression process
        tensor = self.compressed_mps.copy()

        for i in range(4096):
            matrix = np.reshape(tensor[i, :, :], (2, 2048))
            u, s, v = svd(matrix, full_matrices=False)
            u = u[:, :self.bond_dimension]
            s = np.diag(s[:self.bond_dimension])
            v = v[:self.bond_dimension, :]
            tensor[i, :, :] = np.dot(u, np.dot(s, v))

        # Reshape and transpose the tensor to get the original image
        tensor = np.reshape(tensor, (4096, 4096, 2))
        tensor = np.transpose(tensor, (0, 1, 2))
        self.decompressed_image = tensor.astype(np.uint8)

    def get_compressed_image(self):
        return self.compressed_mps

    def get_decompressed_image(self):
        return self.decompressed_image

    def calculate_parameters(self):
        # Calculate the number of parameters in the compressed MPS
        return np.prod(self.compressed_mps.shape)

if __name__ == "__main__":
    
    #given image was preprocessed once and saved as processed_image.jpg
    image_path = "processed_image.jpg"
    bond_dimension = 512
 
    mps_compression = MPSCompression(image_path, bond_dimension)
    mps_compression.compress_image()
    mps_compression.decompress_image()

    # Compare original and decompressed images
    original_image = mps_compression.original_image
    decompressed_image = mps_compression.get_decompressed_image()

    # Calculate the number of parameters
    compressed_parameters = mps_compression.calculate_parameters()

    print(f"Number of parameters in compressed MPS: {compressed_parameters}")

    #Visual comparison
    # plt.subplot(1, 2, 1)
    # plt.imshow(original_image, cmap='gray')
    # plt.title("Original Image")

    # plt.subplot(1, 2, 2)
    # plt.imshow(decompressed_image, cmap='gray')
    # plt.title("Decompressed Image")

    # plt.show()


## Notes from the lab 
# 2^12 ---[]----2^12 -->
    
#T1 = np.reshape(T0, [2]*24) -> dobimo tenzor T1 oblike [2]*24
#Potem moramo grupirati: indeksi i pomenijo ali smo na spodnji ali zgornji polovici kvradrata
#j indeks pa ali smo na desni ali levi strani kvadratka. Tako iondeksiramo vsako vrstivo/piksel slike. 
#Indeksi služijo tudi kot skale.    Skupaj grupiramo indekse ki so enaki (?). T2 = np.transpose(T1, [0,12,1,13,...,11,23]) -> zaporedja indeksov so en i en j drug i drug j trethi i tretji j -- itd. 
# par (i1, j1) je alfa1 itd. dobimo tenzor alf tako da rečemo T3 = np.reshape(T2, [4]*12)


##glavni del naloge: dobit MPS  iz T3.
## Imamo tenzor z alfa vrednostmi. Dodati moramo mi parametre. 
#T4 = np.reshape(T3, [1]+[4]*12+[1]) <- na začetku in na koncu doda še eno dimenzijo
#T4 = np.reshape(T3, [-1,4^12-1] )->
#-> SVD no.linalg.SVD(T4) -> (mi0,alfa0)--U--S--V--(alfa2,...alfa12,mi12)
#D = min([Dmax,len(S)]) <- št singularnih vrednosti
#U = U[:,:D] <- ustrezno št vektorjev
# S = S[:D] <- ustrezno število singularnih vrednosti
# V = V[:D,:]

#to računaj v for loopu (mora biti 10 iteracij)
#A1 = np.reshape(U,[-1,4,D])
#T4 = S@V (?)
#parm = sum(np.p............)
#KONTRAKICJE: T4(nov) = np.einsum("i,ij->ij",S,V) Vsako vrstico matirke V pomnoži z ustrezno vrednostjo iz matrike S

#T3' se v for zanki posodablja
#T3' = np.array([[1]])
#for(int i = 0, i <= 11, ){
#T3' = np.einsum("...i,ijk->...jk",T3',A2)
#
#}

#velikost mi parametrov je 1 [1,4,.....,4,1]
#iz tega nazaj zgradimo tenzor T3'. 
#nazaj naredimo inverz T2' = np.reshape(T3', [2]*24)
#še nazaj naredimo T1' = np.transpose(T2',[0,2,4,..,22,1,3,...,23)
#sestavimo nazja sliko T0' = np.reshape(T1', 2^12,2^12)

#izračunaj 2normo razliko med T0 in TO'
#np.linalg.norm(T0') <- zdaj to normo oz čas računanja te norme primerjamo z normo in čas računanja T0.
#Izračunamo kontrakcijo samega s sabo (?)
#B1 = np.linsum("ijk,ijl->kl",A1,A1)
#C1 = np.linsum("i,kl,kij->lij",B1,A2)
#B2 = np.einsum("lij,lik->jk",C1,A1)


