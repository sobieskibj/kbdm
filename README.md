# Knowledge Base - Diffusion Models

## Introductory materials

- [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)  
Probably the most comprehensive introductory 'paper'. Gives good intuitions, shows all of the math from scratch (almost everything up to 2022 state of knowledge). Probably the longest but also one of the best choices to start with.

- [Yang Song webpage](https://arxiv.org/abs/2208.11970)  
One of the founding fathers of diffusion models in their current shape. He made a great blog post titled 'Generative Modeling by Estimating Gradients of the Data Distribution', which is shorter than the paper above but shows how to intuitively grasp diffusion models.

- [Awesome-Diffusion-Models GitHub repo](https://github.com/diff-usion/Awesome-Diffusion-Models)  
Currently the biggest collection of papers regarding diffusion models. For a long time, it was periodically updated. Now this trend stopped, but almost anything until 2024 can be found there. It starts with a ``Resources`` section that lists almost all introductory materials known to the community.

## Papers

**WR** (worth reading) scale from 1 to 3 indicates how important it is to read a paper (higher means more important).

### Foundations

#### 2022

- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) **WR=3**  
Simple but very practical idea. Instead of training an additional classifier for guidance, we can simply condition diffusion models with some signal and train them simultaneously with and without this signal.

#### 2021

- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) **WR=2**  
Very practical paper that shows how to scale diffusion models to beat GANs. They introduce *classifier guidance*, important technique for conditional generation, which is used very often.

- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)  **WR=3**  
At this point, it turns out that score-matching and diffusion models are the same thing, and can be generalized with a framework that uses stochastic differential equations.

#### 2020

- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)  **WR=3**  
One of the most important papers which shows that a trained diffusion models actually approximates an entire family of objectives, together with a deterministic process (referred to as DDIM) which enables faster inference and direct mapping from image to noise and back. DDIM is used in almost every paper today. Main author is Jiaming Song, so it seems like having *Song* somewhere in your name makes you good at diffusion models. 

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)  **WR=3**  
This paper revived diffusion models after a few years and made them go mainstream. Shows that diffusion models work great at practical resolutions like $256 \times 256$.

#### 2019

- [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) **WR=2**  
Follow-up from Yang Song.

- [Sliced Score Matching: A Scalable Approach to Density and Score Estimation](https://arxiv.org/abs/1905.07088) **WR=2**
Here is where Yang Song steps into the field. Score-based models were being developed in parallel to diffusion models at this time.

#### 2015

- [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)  **WR=1**  
Here is where diffusion models were introduced for the first time (and then not used for long). I include it mostly for historical reasons. Also check out the [webpage of Jascha Sohl-Dickstein](http://sohldickstein.com) - the main author of this paper. Very inspirational.

