# UniPET: A Unified Approach for Whole-Body CT to PET Translation

<img width="757" height="707" alt="Screenshot 2026-08-09 at 16 09 08" src="https://github.com/user-attachments/assets/a6d16688-0416-45ce-a568-c20e8ead20f4" />

Overall framework of UniPET. 
(a) A segmentation model $S$ is applied to each CT scan $x$ to produce a multi-label anatomical mask $M$, which is used to extract region-specific subvolumes from paired CT and PET scans. 
(b) Anatomical regions are grouped into subsets $\Gamma_1, \Gamma_2, \dots, \Gamma_k$ via hierarchical clustering based on intensity range and texture variance,  and are ordered by increasing average volume of each region, assuming that larger regions are more challenging to model.
(c) During training, a data scheduler progressively introduces subsets to the model.
A generator $G = G_e \circ G_d$ synthesizes PET volumes, while an auxiliary classifier $C$ predicts anatomical regions from the latent representation $z$. 
A discriminator $D$ evaluates the realism of the generated output. 
Curriculum progression is guided by the classifier loss $\mathcal{L}_{\text{cls}}$, which decreases as more difficult subsets are introduced.
