#%%
import numpy as np
import matplotlib.pyplot as plt
import skimage.color as clr
from skimage.exposure import histogram
import skimage.exposure as expo
import skimage.filters as flt
import skimage.morphology as morph
import multiprocessing
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import sklearn.metrics as mtr
#%%


def score(custom_mask, case_nr, show_img = False, save = False):

    mask = plt.imread("masks\mask_{}.png".format(case_nr))
    mask = clr.rgb2gray(mask)
    mask = mask > 0.5

    if show_img == True:

        combined = np.invert(mask)
        combined = combined.astype(float)
        combined = clr.gray2rgb(combined)

        combined[custom_mask == True, 0] = 1 
        combined[custom_mask == True, 1] = 0 
        combined[custom_mask == True, 2] = 0 

        combined[mask == True, 0] = 0 
        combined[mask == True, 1] = 0 
        combined[mask == True, 2] = 1 

        combined[np.logical_and(mask, custom_mask) == True , 0] = 0
        combined[np.logical_and(mask, custom_mask) == True, 1] = 0 
        combined[np.logical_and(mask, custom_mask) == True, 2] = 0



        fig,ax = plt.subplots(1, 1, figsize = (50,50))
        ax.imshow(combined)
        ax.set_title("Case: {}".format(case_nr))

        if save == True:
            plt.imsave("combined\combined_{}.png".format(case_nr), combined)
            plt.imsave("custom_masks\custom_masks_{}.png".format(case_nr), custom_mask)

    acc = mtr.accuracy_score(mask.flatten(), custom_mask.flatten())
    jaccard = mtr.jaccard_score(mask.flatten(), custom_mask.flatten())
    
    return acc, jaccard

def make_segment(case_nr, params):

    img = plt.imread(R"result\r{}.png".format(case_nr))

    img_hsv = clr.rgb2hsv(img)

    img_value = img_hsv[:, :, 2]

    v_std = np.std(img_value)

    img_value = (img_value - params[0])*(params[1]/v_std)

    manipulation = np.abs(img_value)

    if params[2] == 0:
        thresh = flt.threshold_otsu(manipulation)

        binary = manipulation > thresh
    elif params[2] == 1:
        thresh = flt.threshold_isodata(manipulation)

        binary = manipulation > thresh
    elif params[2] == 2:
        thresh = flt.threshold_li(manipulation)

        binary = manipulation > thresh
    elif params[2] == 3:
        thresh = flt.threshold_mean(manipulation)

        binary = manipulation > thresh
    elif params[2] == 4:
        thresh = flt.threshold_niblack(manipulation)

        binary = manipulation > thresh
    elif params[2] == 5:
        thresh = flt.threshold_sauvola(manipulation)

        binary = manipulation > thresh
    elif params[2] == 6:
        thresh = flt.threshold_triangle(manipulation)

        binary = manipulation > thresh
    else:
        thresh = flt.threshold_yen(manipulation)

        binary = manipulation > thresh

    binary = morph.remove_small_holes(binary,area_threshold=100)
    binary = morph.remove_small_objects(binary, min_size=70, connectivity=2)

    return binary


def segment_and_score(params):

    acc_scores = []
    jac_scores = []

    for j in range(30):

        mask = make_segment(case_nr = j+1, params = params[0:3])
        scores = score(mask, case_nr = j+1)

        acc_scores.append(scores[0])
        jac_scores.append(scores[1])

    print("set nr: {}".format(params[3]))
    print("acc mean: {}".format(np.mean(acc_scores)))
    print("jac mean: {}".format(np.mean(jac_scores)))           #niestety te printy nie działają w momencie kiedy używa się ich w 
    combined_score = [np.mean(acc_scores),np.mean(jac_scores)]  #funkcji w multithreadingu

    return combined_score

    
#%%
num_sample_sets = 1000 #zmiana ilości wylosowanych zestawów parametrów


random_params_hsv = np.random.random_sample((num_sample_sets,2)).astype(np.float16)

random_params_thresh = np.random.randint(low = 0, high=7, size = (num_sample_sets,1))

random_params = np.concatenate((random_params_hsv,random_params_thresh, np.expand_dims(np.arange(num_sample_sets), axis = 1)), axis = 1)

#Zestaw parametrów zawiera nową średnią oraz odchylenie standardowe kanału value oraz integer od 0 do 7, które oznaczają różne metody
#thresholdingu oraz ID zestawu

# %%
np.savetxt("rand_params_with_thresh.txt",random_params)

#%%

#random_params = np.loadtxt("rand_params_with_thresh.txt")
#W razie jeśli będzie potrzeba wczytania poprzednio wygenerowanego zestawu wylosowanych parametrów

# %%

set_nr_list = list(range(num_sample_sets))

num_cores = 4 #multiprocessing.cpu_count()
                #Zlicza ilość dostępnych cpu
                
#%%
scores_list = Parallel(n_jobs=num_cores)(delayed(segment_and_score)(random_params[set_nr,:]) for set_nr in set_nr_list)
scores_all = np.array(scores_list)

#dzialanie funkcji na wielu watkach
#jesli chcemy zmienic ilosc watkow poswieconych na to zadanie nalezy zmienic zmienna num_cores na tyle watkow ile chcemy wykorzystac


#%%

scores_all = np.array(scores_all)

np.savetxt("scores.txt",scores_all)

highest = np.argmax(scores_all[:,1])
print(highest)

#znajduje dla ktorego indeksu mamy najwyzszy wynik jaccarda
#%%
params_best = random_params[highest,:] # zapomoca znalezonego indeksu wybieramy zestaw parametrow
scores_best = []

for j in range(30):

    mask = make_segment(case_nr = j+1, params = params_best)
    scores = score(mask, case_nr = j+1, show_img = False, save = False)
    scores_best.append(scores)

#Przepuszczamy przez wszystkie obrazki funkcje make segment oraz score dla najlepszego znalezionego zestawu parametrów
#Jeśli chcemy, żeby funkcja pokazała otrzymaną maske wraz z porównaniem jej z wzorcową maską zmienna show_img należy ustawić na True
# Podobnie jeśli chcemy aby funkcja zapisała obraz należy ustawić save na True(funkcjonalność save nie zadziała jeśli show_img będzie na False) 

#%%
scores_best = np.array(scores_best)

print("set nr: {}".format(highest))
print("acc mean: {}".format(np.mean(scores_best[:,0])))
print("jac mean: {}".format(np.mean(scores_best[:,1])))

#Drukuje nam najlepsze uzyskane wyniki
# %%
