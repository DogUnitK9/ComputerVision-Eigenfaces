import PIL.Image, PIL.ImageOps, PIL.ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import random
import glob

img_width = 24
img_height = 24
p = img_width * img_height
data = []
fname_list = glob.glob("**/*.pgm")
for fname in fname_list:
    #img_orig = Pillow load, rescale, extract numpy matrix
    img_orig = PIL.Image.open(fname).convert("L").resize((img_width,img_height))
    row = np.reshape(img_orig, (p,))
    data.append(row)
me_img = PIL.Image.open("daniel.png").convert("L").resize((img_width, img_height))
me_data = np.asarray(me_img).reshape((p,))
img2 = np.asarray(PIL.Image.open("Abdulaziz_Kamilov_0001.pgm").convert("L").resize((img_width, img_height))).reshape((p,))
img3 = np.asarray(PIL.Image.open("Karen_Clarkson_0001.pgm").convert("L").resize((img_width, img_height))).reshape((p,))
img4 = np.asarray(PIL.Image.open("Keanu_Reeves_0002.pgm").convert("L").resize((img_width, img_height))).reshape((p,))
img5 = np.asarray(PIL.Image.open("Lance_Armstrong_0009.pgm").convert("L").resize((img_width, img_height))).reshape((p,))
data = np.array(data)

print("Images Loaded into Data, Now Calculating Eigenvectors\n")
center = np.mean(data, axis=0)
center_img = data - center
#S is the Eigenvalues, V is the Eigenvectors,
U, S, V = np.linalg.svd(center_img)
S_as_percent = S / np.sum(S) #array of nums, add for a, b and c.
print("Percetange kept by the first 5 eigenvectors:", int(round(S_as_percent[0]+S_as_percent[1]+S_as_percent[2]+S_as_percent[3]+S_as_percent[4],2)* 100), "%")
sumfifty = 0
for i in range(50):
    sumfifty += S_as_percent[i]
print("Percetange kept by the first 50 eigenvectors:", int(round(sumfifty,2) * 100), "%")
num = 0
total = 0
for i in range(len(S_as_percent)):
    num += S_as_percent[i]
    if num >= 0.85:
        total = i
        break
print("Total number of Eigenfaces needed to be kept to preserve 85%:", total,'\n')
n = len(data)
kept = total
compressed_data = np.zeros((n, kept))
#print statement to make sure the code is running
print("Eigenvectors loaded, Compressing Data\n")
for i in range(n):
    #compress image n onto our kept eigenvectors and store in compressed_data
    orig_img = data[i,:]
    #loop through each kept eigenface (in V)
    for j in range(kept):
        Q = orig_img - center
        proj = np.dot(Q, V[j,:])  
        compressed_data[i,j] = proj

def compress(img):
    me_compressed = np.zeros((kept))
    for i in range(kept):
        Q = img - center
        proj = np.dot(Q, V[i,:])
        me_compressed[i] = proj
    return me_compressed
me_compressed = compress(me_data)
img2_compressed = compress(img2)
img3_compressed = compress(img3)
img4_compressed = compress(img4)
img5_compressed = compress(img5)

print("Data Compressed, Finding Best Face Match\n")

def search(inp, faces, num = 0):
    best_face = None
    best_dist = 5000000
    file_name = ""
    for face in faces:
        dist = np.sum((inp - face) ** 2)
        if dist < best_dist:
            best_dist = dist
            file_name = fname_list[num]
            best_face = data[num]
        num+=1
    return best_face, file_name
def betterName(file_name):
    name = ''
    for i in range(0,len(file_name)):
        if i >= 7:
            name = name + file_name[i]
    name = name.split('_')
    if len(name) == 4:
        name = name[0]+" "+name[1]+" "+name[2]
    if len(name) == 3:
        name = name[0]+" "+name[1]
    if len(name) == 2:
        name = name[0]
    return name
best_match, file_name = search(me_compressed, compressed_data)
best_match2, file_name2 = search(img2_compressed, compressed_data)
best_match3, file_name3 = search(img3_compressed, compressed_data)
best_match4, file_name4 = search(img4_compressed, compressed_data)
best_match5, file_name5 = search(img5_compressed, compressed_data)
name = betterName(file_name)
name2 = betterName(file_name2)
name3 = betterName(file_name3)
name4 = betterName(file_name4)
name5 = betterName(file_name5)
                
fig, axes = plt.subplots(4,10)
image_to_show = center.reshape(img_height, img_width)
axes[0,0].imshow(image_to_show)
axes[0,0].set_title("Average Face")
sel = random.sample(range(0,len(fname_list)),10)
e = 0
for i in sel:
    axes[1,e].set_title("EFace"+str(e))
    axes[1,e].imshow(V[e,:].reshape(img_height, img_width))
    axes[2,e].set_title(betterName(fname_list[i]))
    axes[2,e].imshow(data[i,:].reshape(img_height, img_width))
    e+=1
axes[3,0].set_title("Me lmao")
axes[3,0].imshow(me_data.reshape((img_height, img_width)))
axes[3,1].set_title(name)
axes[3,1].imshow(best_match.reshape((img_height, img_width)))
axes[3,2].set_title("Abdulaziz Kamilov")
axes[3,2].imshow(img2.reshape((img_height, img_width)))
axes[3,3].set_title(name2)
axes[3,3].imshow(best_match2.reshape((img_height, img_width)))
axes[3,4].set_title("Karen Clarkson")
axes[3,4].imshow(img3.reshape((img_height, img_width)))
axes[3,5].set_title(name3)
axes[3,5].imshow(best_match3.reshape((img_height, img_width)))
axes[3,6].set_title("Keanu Reeves")
axes[3,6].imshow(img4.reshape((img_height, img_width)))
axes[3,7].set_title(name4)
axes[3,7].imshow(best_match4.reshape((img_height, img_width)))
axes[3,8].set_title("Lance Armstrong")
axes[3,8].imshow(img5.reshape((img_height, img_width)))
axes[3,9].set_title(name5)
axes[3,9].imshow(best_match5.reshape((img_height, img_width)))
for i in range(4):
    for e in range(10):
        axes[i,e].axis('off')
plt.show()
