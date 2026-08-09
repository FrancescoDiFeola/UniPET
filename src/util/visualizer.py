import numpy as np
import os
import ntpath
import time
import matplotlib.pyplot as plt
from . import util
from . import html
from .animator3d import MedicalImageAnimator
from PIL import Image
import plotly.graph_objects as go

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port = opt.display_port)
            self.display_single_pane_ncols = opt.display_single_pane_ncols

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log_last.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    #def display_current_results(self, visuals, epoch):
    def display_current_results(self, visuals, epoch, district ):
        #if self.display_id > 0: # show images in the browser
        if False:
            if self.display_single_pane_ncols > 0:
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
    table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
    table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
</style>""" % (w, h)
                ncols = self.display_single_pane_ncols
                title = self.name
                label_html = ''
                label_html_row = ''
                nrows = int(np.ceil(len(visuals.items()) / ncols))
                images = []
                idx = 0
                for label, image_numpy in visuals.items():
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1]))*255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win = self.display_id + 2,
                              opts=dict(title=title + ' labels'))
            else:
                idx = 1
                for label, image_numpy in visuals.items():
                    #image_numpy = np.flipud(image_numpy)
                    self.vis.image(image_numpy.transpose([2,0,1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                    idx += 1

        if self.use_html: # save images to a html file --> Salvataggio solo all'epoca corrente!!
            for label, image_numpy in visuals.items(): # in questo caso label è real_A, fake_B, real_B
                if len(image_numpy.shape) == 4:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_%s.gif' % (epoch, label, district)) # MODIFICA: ho aggiunto anche il nome del distretto
                    animator = MedicalImageAnimator(image_numpy[0], [], 0, save=True)
                    animate = animator.run(img_path)
                else:
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s_%s.png' % (epoch, label, district))
                    util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            #for n in range(epoch, 0, -1): # salvataggio immagini per ogni epoca --> al posto di epoch qua sotto va messo n!!
            webpage.add_header('epoch [%d] - %s' % (epoch,district))
            ims = []
            txts = []
            links = []

            for label, image_numpy in visuals.items():
                img_path = 'epoch%.3d_%s_%s.gif' % (epoch, label, district)
                ims.append(img_path)
                txts.append(label)
                links.append(img_path)
            webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, opt, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X':[],'Y':[], 'legend':list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

    import numpy as np
    from PIL import Image

    def save_images1(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)

            # Rimuovi dimensioni extra e scala i valori per renderli compatibili con PIL
            if image_numpy.ndim == 4:
                image_numpy = image_numpy[0, 0, :, :]
            elif image_numpy.ndim == 3:
                image_numpy = image_numpy[0, :, :]

            # Scala i valori e converti in uint8
            image_numpy = (image_numpy * 255).clip(0, 255).astype(np.uint8)

            # Salva l'immagine
            image_pil = Image.fromarray(image_numpy)
            image_pil.save(save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)

        webpage.add_images(ims, txts, links, width=self.win_size)


    # Funzione aggiornata plot_loss
    def plot_loss(self, epochs, loss_G, switch_epochs, switch_labels): # funziona !!!
        plot_loss_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        os.makedirs(plot_loss_dir, exist_ok=True)
        plot_loss_path = os.path.join(plot_loss_dir, 'loss_plot_last.png')

        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot delle perdite
        ax.plot(epochs, loss_G, label="Generator Loss", linewidth=0.8)

        # Aggiungi linee verticali per i cambi di dataset
        #for switch_epoch, label in zip(switch_epochs, switch_labels):
            #ax.axvline(x=switch_epoch, color='green', linestyle='--', label=f"Switch: {label}", ha='center', va='bottom')

        colors = ["blue", "purple", "green", "orange", "red", "cyan", "magenta", "brown", "pink", "gray", "yellow", "lime", "teal", "gold"]
        for i, (switch_epoch, label) in enumerate(zip(switch_epochs, switch_labels)):
            ax.axvline(x=switch_epoch, color=colors[i % len(colors)], linestyle='--', label=f"Switch: {label}")
            ax.text(switch_epoch + 0.2, max(loss_G), str(switch_epoch), color=colors[i % len(colors)], fontsize=9, ha='center')

        ax.set_title("Generator Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        # Sposta le etichette dei distretti fuori dal grafico
        #ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), title="Districts", fontsize=9, title_fontsize=10)
        ax.legend(
            loc='upper center',  # Sposta la legenda sopra il grafico
            bbox_to_anchor=(0.5, -0.15),  # Centra la legenda sotto il grafico
            ncol=3,  # Dividi la legenda su 3 colonne
            fontsize=8,  # Riduci il font
            title="Districts",
            title_fontsize=10
        )

        plt.tight_layout()

        plt.savefig(plot_loss_path, bbox_inches='tight')
        plt.close(fig)


        #print(f"Grafico delle perdite salvato in: {plot_loss_path}")



    def plot_loss_dynamic(self, epochs, loss_G, switch_epochs, switch_labels):
        # Directory di salvataggio
        plot_loss_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        os.makedirs(plot_loss_dir, exist_ok=True)
        plot_loss_path = os.path.join(plot_loss_dir, 'loss_plot_last.html')

        # Colori per le linee verticali
        colors = ["blue", "purple", "green", "orange", "red", "cyan", "magenta", "brown", "pink", "gray", "yellow", "lime", "teal", "gold"]

        # Crea il grafico dinamico
        fig = go.Figure()

        # Aggiungi la curva del Generator Loss
        fig.add_trace(go.Scatter(
            x=epochs,
            y=loss_G,
            mode='lines',
            name="Generator Loss",
            line=dict( width=1.5)  # Modifica colore e spessore
        ))

        # Aggiungi linee verticali per i cambi di dataset
        for i, (switch_epoch, label) in enumerate(zip(switch_epochs, switch_labels)):
            fig.add_trace(go.Scatter(
                x=[switch_epoch, switch_epoch],
                y=[min(loss_G), max(loss_G)],
                mode='lines',
                name=f"Switch: {label}",
                line=dict(color=colors[i % len(colors)], dash='dash', width=1.5)  # Linea tratteggiata
            ))

            # Valore di switch_epoch dove finisce la linea tratteggiata
            fig.add_trace(go.Scatter(
                x=[switch_epoch],  # Posizione lungo l'asse x
                y=[max(loss_G)],  # Altezza al massimo valore di loss_G
                text=[str(switch_epoch)],  # Testo da visualizzare
                mode='text',
                textposition='top center',  # Testo centrato in alto
                showlegend=False  # Non mostrare questa traccia nella legenda
            ))

        # Imposta layout
        fig.update_layout(
            title="Generator Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            legend=dict(
                orientation="h",  # Orientazione orizzontale
                yanchor="top",
                y=-0.2,  # Sposta la legenda sotto il grafico
                xanchor="center",
                x=0.5,
                font=dict(size=10)
            ),
            margin=dict(l=40, r=40, t=40, b=100),  # Margini del grafico
            width=1200,  # Larghezza del grafico
            height=700  # Altezza del grafico
        )

        # Salva il grafico in formato HTML
        fig.write_html(plot_loss_path)
        print(f"Grafico interattivo salvato in: {plot_loss_path}")



    def plot_loss1(self, epochs, loss_G, switch_epochs, switch_labels, first_district):
        plot_loss_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        os.makedirs(plot_loss_dir, exist_ok=True)
        plot_loss_path = os.path.join(plot_loss_dir, 'loss_plot_last.png')

        if isinstance(first_district, list):
            labels= first_district + switch_labels + [""]
        else:
            labels= [first_district] + switch_labels + [""]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Aggiungi segmenti colorati per ogni distretto
        start_idx = 0
        colors = ["blue", "purple", "green", "orange", "red", "cyan", "magenta", "brown", "pink", "gray", "yellow", "lime", "teal", "gold"]

        for i, (switch_epoch, label) in enumerate(zip(switch_epochs + [epochs[-1]], labels)):
            try:
                end_idx = epochs.index(switch_epoch)
            except ValueError:
                end_idx = len(epochs)+1  # Usa tutti i dati rimanenti se l'epoca non è trovata

            # Assicurati che ci siano dati validi da plottare
            if start_idx < end_idx:
                ax.plot(epochs[start_idx:end_idx],loss_G[start_idx:end_idx],label=f"Districts: {label}" if label else None, color=colors[i % len(colors)], linewidth=0.8)
            start_idx = end_idx

        # Aggiungi linee verticali per i cambiamenti dei dataset
        for switch_epoch in switch_epochs:
            ax.axvline(x=switch_epoch, color='black', linestyle='--')
            ax.text(switch_epoch, max(loss_G), f"{switch_epoch}", rotation=90, verticalalignment='bottom', fontsize=8)

        ax.set_title("Generator Loss with District Changes")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        # Sposta le etichette dei distretti fuori dal grafico
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), title="Districts", fontsize=9, title_fontsize=10)

        plt.tight_layout()
        plt.savefig(plot_loss_path, bbox_inches='tight')

        plt.close(fig)
        #print(f"Grafico delle perdite salvato in: {plot_loss_path}")


















