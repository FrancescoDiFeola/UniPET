import time
from options.train_options import TrainOptions
from data.dataset_CACHE_CL import CreateDynamicDataloader
from models.models import create_model
from util.visualizer import Visualizer
import tensorflow as tf
from tensorflow.image import psnr
import os
import numpy as np
import random

"""Il codice è predisposto per funzionare in 5 configurazioni: 
1. SORTED --> distretti ordinati secondo difficoltà. La lista district contiene già i distretti ordinati 
2. RANDOM --> distretti ordinati in maniera randomica. Viene fatta una operazione di randimizzazione alla lista district 
3. SORTED+GROUPED --> distretti raggruppati e ordinati secondo difficoltà. La lista district contiene gia i distretti 
                    ordinati secondo difficoltà. I distretti appartenenti allo stesso gruppo sono vicini tra loro. 
                    I gruppo sono [Adrenal gland, gallbladder], [bladder, trachea ] [pancreas e stomach]. Per fare in 
                    modo di in serir i distretti appartenenti allo stesso gruppo simultaneamente, c'è un controllo per cui
                    se il distretto aggiunti si chiama Adrenal gland, o bladder o pancreas, bisogna aggiungere anche 
                    il successivo, cioè l'altro elemento del gruppo
4. RANDOM+GROUPED --> distretti raggruppati ma introdotti in ordine randomico. i distretti nella lista district vengono
                    ordinati in ordine randomico ma poi, poichè anche l'opzione GROUPED è attiva, Adrenal gland, bladder 
                    e pancrea vengono spostati in modo che siano vicini ai distretti del loro gruppo.
5. SORTED+GROUPED+WARMUP --> Analogo a SORTED+GROUPED ma in questo caso il periodo di warm up dipede dalla dimensione 
                    del distretto aggiunti. Quindi nel parser viene passata una lista con i valori dei periodi di 
                    warmup che devono essere applicati ogni volta che viene aggiunti un nuovo distretto.
                    In tutti gli altri casi (1,2,3,4) il periodo di warm-up è tenuto fisso e impostato a 200


INFO AGGIUNTIVE: 
a) Quando si verifica la condizione di aggiunta di un nuovo distretto ( Early Stopping), viene caricato il modello 
migliore e viene salvato come "Switch_Net_G"

b) I distretti aggiunti nel processo di addestramento sono contenuti nella lista "included-district". Se l'addestramemto 
parte da zero, questa lista è vuota, altrimenti, se si parte da un checkpoint (es. switch_2) si deve andare a vedere 
i distretti aggiunti fino a quel momento (vedere il grafico). 

c) Cosa fare quando si parte da un checkpoint ?
    1- passare la variabile parto_da-switch
    2- se l'esperimento prevede i distretti ordinati in maniera randomica, si deve risalire all'ordine con cui i 
    distretti erano stati ordinati dopo la randomizzazione (vedere file di output) e scrrivere la lista in distrcit 
    (in base_options)
    3- vedere quali sono i distretti che erano stati aggiunti fino a quel checkpoint (vedere grafico plot_loss) e 
    scriverli nella lista included_districts
"""

opt = TrainOptions().parse()
district = opt.district # lista contenente i tutti i distretti ordinati secondo difficoltà.
                        # l'ordine è stato definito a piori prednendo come metrica di difficoltà la dimensione dei distratti
                        # se GROUPED -->  i distretti all'interno della lista sono ordinati sempre secondo difficoltà ma in modo che i distretti appartenenti allo stesso gruppo siano vincini.
                                        # Gallbladder sta dopo Adrenal Gland,  Trachea sta dopo Bladder e Stomach sta dopo pancreas
                        # se NOT GROUPED --> i distretti sono ordinati per difficoltà ma poichè i distretti non sono stati raggruppati,
                                        # Gallbladder potrebbe stare lontano da Adrenal Gland, Trachea lontano da bladder  e Stomach lonatno da pancreas.


# ---> se l'opzione RANDOM è impostata i distretti contenuti all'interno della lista district vengono riordinati in maniera randomica.
# ---> se i distretti sono anche GROUPED, una volta che gli elementi della lista sono stati ordinati in maniera randomica i distretti
# vengono riorganizzati in modo che distretti appartenti allo stesso gruppo siamo vincini.
# In questo modo, durante il tranining se l'opzione GROUPED è activa, quando il nuovo distretto aggiunto è Adrenal Gland,
# Bladder o Pancreas, verrà preso anche il successivo, cioè l'altro elemento del gruppo.
# ---> NB: se l'addestramento parte da un checkpoint (parto_da_switch) indipendentemente e l'opzione RANDOM è attiva,
# non viene fatta nuovamente la randomizzazione ma bisogna vedere come erano stati ordinati i distretti dopo la randomizzazione
# e mettere questa lista in district.
if opt.random_district and not opt.parto_da_switch:
    random.shuffle(district)
    if opt.grouped_district:
        district.remove("adrenal_gland")
        district.insert(district.index("gallbladder"), "adrenal_gland")
        district.remove("bladder")
        district.insert(district.index("trachea"), "bladder")
        district.remove("pancreas")
        district.insert(district.index("stomach"), "pancreas")
    print(f'Distretti in ordine randomico: {district}')

# Parametri per il training
num_districts = len(district)


if opt.parto_da_switch:
    included_districts = opt.included_districts
    added_districts = len(included_districts)-1
    switch_counter = opt.switch_counter
else:
    included_districts = [district[0]]  # Inizia con un solo distretto
    added_districts = 0
    switch_counter = 1
    if opt.grouped_district: # se l'opzione GROUPED è attiva dovro aggiungere gli elementi dello stesso gruppo insieme
                             # Sia se l'opzione RANDOM è attiva sia se non è attiva, Adrenal Gland e Gallbladder, Bladder e Trachea,
                             # Pancreas e stomach stanno vicini. Quindi se il primo distretto è Adrenal gland, bladder o pancreas,
                             # per includere  gli altri elementi del gruppo, aggiungero anche il distretto successivo.
        if district[0] == 'adrenal_gland' or district[0] == 'bladder' or district[0] == 'pancreas':
            included_districts.append(district[1])
            added_districts += 1
print(f'Included districts: {included_districts}')
print(f'Added districts: {added_districts}')
print(f'num_districts: {num_districts}-1')
print(f'Switching counter: {switch_counter}')

if opt.warm_up_variabile:
    all_warm_up_epochs = opt.warm_up_epochs  # lista contenente i valori di Warm-up per ogni switch. I valori sono stati calcolati a priori
                                            # [200, 200, 203, 205, 209, 216, 221, 227, 261, 263, 400]
    if opt.parto_da_switch:
        warm_up_index= opt.switch_counter - 1
    else:
        warm_up_index=0
else:
    all_warm_up_epochs= [opt.warm_up_epochs]*num_districts # se warm_up è fisso creo una lista lunga quanto il numero di distretti con tutti valori uguali e pari a quelli specifictai in train.options
    warm_up_index = 0                                       # es. [200, 200, 200, 200, 200, 200, 200, 200, 200, 200]

warm_up_epochs=all_warm_up_epochs[warm_up_index] # primo elemento della lista warm_up epochs
print(f"warm-up epochs: {all_warm_up_epochs}")
print(f"warm-up index: {warm_up_index}")
print(f"warm up epochs: {warm_up_epochs}")

patience = opt.patience #100
best_loss = float('inf')
no_improvement_count = 0





current_data_loader = CreateDynamicDataloader(opt, included_districts, shuffle=True, cache=True)

dataset_sizes = [len(current_data_loader.dataset)]
for i, size in enumerate(dataset_sizes):
    print(f'#training images distretto {included_districts[i]} = {size}')

model = create_model(opt)  # importa il modello
visualizer = Visualizer(opt)
total_steps = 0

epochs = []
loss_G_epoch = []
loss_D_epoch = []
switch_epochs = []
switch_labels = []


for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):

    epochs.append(epoch)  # Aggiungi l'epoca corrente alla lista
    epoch_start_time = time.time()
    epoch_iter = 0

    # Dopo che è trascorso un periodo di warm-up iniziale, se la loss non migliora per un certo numero di epoche (patience),
    # se non sono stati gia aggiunti tutti i distretti, si carica riparte dal modello migliore ( cioè prima che la loss inizia a diminuire)
    # e si aggiunge un nuovo distretto. Anche in questo caso, se il distretto è Adrenal gland, bladder o pancreas, viene aggiunto anche
    # distretto successivo, per il motivo scritto sopra per il primo distretto.
    if epoch >= warm_up_epochs and no_improvement_count >= patience :
        if added_districts < num_districts - 1: # se NON sono stati ancora aggiunti tutti i distretti
            print(f"No improvement for {patience} epochs. Adding a new district.")
            # Ripristina il modello dall'ultimo checkpoint
            model.load_best_network()
            print(f"caricamento modello epoca: {epoch_best_model}")
            model.save(f"switch{switch_counter}")
            print(f"Salvato modello come 'switch{switch_counter}'")
            switch_counter += 1
            best_loss = float('inf')

            # Aggiungi un nuovo distretto
            added_districts += 1
            new_district = district[added_districts]
            included_districts.append(new_district)

            if opt.grouped_district:
                if new_district == 'adrenal_gland' or new_district == 'bladder' or new_district == 'pancreas':
                    added_districts += 1
                    included_districts.append(district[added_districts])

            current_data_loader = CreateDynamicDataloader(opt, included_districts, shuffle=True, cache=True)

            # Aggiungi epoca e distretti per la linea di switch
            switch_epochs.append(epoch)
            switch_labels.append(", ".join(included_districts))

            # Reset del contatore e periodo di warm-up
            no_improvement_count = 0
            warm_up_index += 1
            warm_up_epochs = epoch + all_warm_up_epochs[warm_up_index]
            print(f"Warm-up epochs: {epoch} + {all_warm_up_epochs[warm_up_index]}")

        else: # se sono stati aggiunti tutti i distretti
            print(f"No improvement for {patience} epochs. Adding a new district.")
            # Ripristina il modello dall'ultimo checkpoint
            model.load_best_network()
            print(f"caricamento modello epoca: {epoch_best_model}")
            model.save(f"BEST_final_400")
            print("Salvato modello come BEST_final_400")
            print("-------------- Fine addestramento -------------")
            break



    loss_G_batch = []
    loss_D_batch = []

    for i, data in enumerate(current_data_loader):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        model.set_input(data)
        model.set_labels(data['label'])  # Passa le etichette (relative al distretto) al modello

        model.optimize_parameters()

        loss_G_batch.append(model.loss_G.item())
        loss_D_batch.append(model.loss_D.item())

    avg_loss_G = np.mean(loss_G_batch)
    avg_loss_D = np.mean(loss_D_batch)

    loss_G_epoch.append(avg_loss_G)
    loss_D_epoch.append(avg_loss_D)



    # Early stopping: aggiorna best_loss e il contatore no_improvement_count
    # if epoch >= warm_up_epochs and added_districts != num_districts - 1:
    if epoch >= warm_up_epochs:
        if avg_loss_G < best_loss:
            best_loss = avg_loss_G  # Aggiorna la migliore loss
            no_improvement_count = 0
    
            # Salva il miglior checkpoint sovrascrivendo il file
            model.save('best')
            epoch_best_model = epoch
            print(f"New best model found at epoch {epoch_best_model} with loss {best_loss}")
            #print(f"Best model updated at {checkpoint_path} with loss {best_loss}")
        else:
            no_improvement_count += 1

        print( f"Epoch {epoch}, Avg Loss G: {avg_loss_G}, Best Loss: {best_loss}, No Improvement Count: {no_improvement_count}")


    visualizer.plot_loss(epochs, loss_G_epoch, switch_epochs, switch_labels)
    #visualizer.plot_loss1(epochs, loss_G_epoch, switch_epochs, switch_labels, initial_district)
    visualizer.plot_loss_dynamic(epochs, loss_G_epoch, switch_epochs, switch_labels)


    if epoch % opt.save_images_loss_freq == 0: # 1000
        # Salvataggio IMMAGINI
        #visualizer.display_current_results(model.get_current_visuals(), epoch) # ho aggiunto district per salvare le immagini associandogli il nome del distretto

        # Salvataggio LOSS
        errors = model.get_current_errors()
        t = (time.time() - iter_start_time) / opt.batchSize
        visualizer.print_current_errors(epoch, epoch_iter, errors, t)

        # Salvataggio del MODELLO
    if epoch % opt.save_latest_freq == 0: # 200
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        #model.save('latest')
        model.save(epoch)

    #print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps)) # SE VOGLIO SALVARE A OGNI EPOCA!!!!
    #model.save('latest') # si sovrascrive!




    # CALCOLO METRICHE
    visuals = model.get_current_visuals()
    # Accesso a real_B e fake_B dal dizionario
    real_B = visuals['real_B']
    fake_B = visuals['fake_B']

    if (epoch + 1) % opt.save_metrics_freq == 0:  # 200
        # Calcola MAE
        mae = tf.reduce_mean(tf.abs(real_B - fake_B))

        # Calcola MSE
        mse = tf.reduce_mean(tf.square(real_B - fake_B))

        # Calcola PSNR
        psnr = 10 * tf.math.log(1.0 ** 2 / mse) / tf.math.log(10.0)

        # LOSS FUNCTIONS:
        classification_loss = model.classification_loss.item()  # Loss classificatore --> cross entropy
        generator_loss = model.loss_G.item()  # Loss generatore
        discriminator_loss = model.loss_D.item()  # Loss del discriminatore

        # Calcola SSIM
        # ssim_axes = [ssim(real_B[:, :, :, :, i], fake_B[:, :, :, :, i], max_val=1.0) for i in range(real_B.shape[4])]
        # ssim = tf.reduce_mean(ssim_axes)
        with open(os.path.join(opt.checkpoints_dir, opt.name, 'metrics_pix2pix_last.txt'),
                  'a') as metrics_file:  # se metrics_pix2pix.txt non esiste in questa directory, viene creata
            metrics_file.write(
                f"Iteration {total_steps}, Epoch {epoch + 1}, "
                f"MAE: {mae}, PSNR: {psnr}, Classification Loss: {classification_loss}, Generator Loss: {generator_loss}, Discriminator Loss {discriminator_loss}\n")

        print( f"Iteration {total_steps}, Epoch {epoch + 1}, MAE: {mae}, PSNR: {psnr}, Classification Loss: {classification_loss}, Generator Loss: {generator_loss}, Discriminator Loss {discriminator_loss}")

    print('End of epoch %d / %d \t Time Taken: %d sec' %(epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    if epoch > opt.niter:
        model.update_learning_rate()



