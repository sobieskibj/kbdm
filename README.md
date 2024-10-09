# Knowledge Base - Diffusion Models

## Introductory materials

- [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)  
Probably the most comprehensive introductory 'paper'. Gives good intuitions, shows all of the math from scratch (almost everything up to 2022 state of knowledge). Probably the longest but also one of the best choices to start with.

- [Yang Song webpage](https://arxiv.org/abs/2208.11970)  
One of the founding fathers of diffusion models in their current shape. He made a great blog post titled 'Generative Modeling by Estimating Gradients of the Data Distribution', which is shorter than the paper above but shows how to intuitively grasp diffusion models.

- [Awesome-Diffusion-Models GitHub repo](https://github.com/diff-usion/Awesome-Diffusion-Models)  
Currently the biggest collection of papers regarding diffusion models. For a long time, it was periodically updated. Now this trend stopped, but almost anything until 2024 can be found there. It starts with a ``Resources`` section that lists almost all introductory materials known to the community.

- [DDPM implementation from labmlai](https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/diffusion/ddpm)  
Quite clean implementation of DDPM with great comments about each code line. Strongly recommended if you want to see how diffusion models can be implemented and how they work in practice.

- [How to Train Your Energy-Based Models](https://arxiv.org/abs/2101.03288)  
Introduction to energy-based models from Yang Song and Diederik Kinga from 2021. Covers important basics despite not being so fresh.

- [Tutorial on Diffusion Models for Imaging and Vision](https://arxiv.org/abs/2403.18103)  
Comprehensive and up-to-date tutorial on diffusion models already praised by the community.

- [Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling](https://vdeborto.github.io/publication/schrodinger_bridge/)  
Introduction to Diffusion Schrödinger Bridges based on a [NeurIPS 2021 Spotlight](https://arxiv.org/abs/2106.01357).

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

### Representation learning

#### 2024

- [Exploring Diffusion Time-steps for Unsupervised Representation Learning](https://arxiv.org/abs/2401.11430) **WR=3**    
The authors propose to learn a specifc feature for each timestep t to compensate for the attribute lost through noising.

#### 2023

- [SODA: Bottleneck Diffusion Models for Representation Learning](https://arxiv.org/abs/2311.17901) **WR=1**    
In essence, an extension of DiffAE to different modalities for the conditioning signal. 

- [Diffusion Model as Representation Learner](https://arxiv.org/abs/2308.10916) **WR=2**    
Shows that an off-the-shelft diffusion model can be adapted to representation learning tasks via reinforcement learning and student networks.

- [InfoDiffusion: Representation Learning Using Information Maximizing Diffusion Models](https://proceedings.mlr.press/v202/wang23ah.html) **WR=3**    
Improves representation learning capabilities of DiffAE by extending it with information theory related aspects.

- [Self-Discovering Interpretable Diffusion Latent Directions for Responsible Text-to-Image Generation](https://arxiv.org/abs/2311.17216) **WR=2**    
Proposes an approach to find interpretable direction in the h-space for user-defined concepts by learning a latent vector.

- [Unsupervised Discovery of Interpretable Directions in h-space of Pre-trained Diffusion Models](https://arxiv.org/abs/2310.09912) **WR=3**    
Unsupervised approach to finding editing directions in the h-space. 

- [Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry](https://arxiv.org/abs/2307.12868) **WR=3**   
Proposes to use riemannian geometry to find the connection between the x-space (image space) and h-space. It allows for unsupervised discovery of editing directions.

- [Diffusion Models already have a Semantic Latent Space](https://arxiv.org/abs/2210.10960) **WR=3**  
First paper that proposes to use the U-Net bottleneck and shows that it contains meaningful representations that allow for a variety of modifications. Importantly, the changes are rarely disentangled and in general the entire image is modified in some way.

#### 2022

- [Unsupervised Representation Learning from Pre-trained Diffusion Probabilistic Models](https://arxiv.org/abs/2212.12990) **WR=3**    
Shows that a standard diffusion model can be fine-tuned to possess a semantic encoder, which means that DiffAE does not have to be trained from scratch.

- [Diffusion Autoencoders: Toward a Meaningful and Decodable Representation](https://arxiv.org/abs/2111.15640) **WR=3**    
Oral paper from CVPR 2022. One of the first, if not the first one, approach to obtain a handy semantic latent space in diffusion models. They train a standard diffusion model jointly with a semantic encoder that outputs a representation which is then used as conditioning signal for the denoising network.

### Inverse problems

#### 2024

- [Divide-and-Conquer Posterior Sampling for Denoising Diffusion Priors](https://arxiv.org/abs/2403.11407)  
Math-heavy. Combines standard langevin dynamics with Feynman-Kac models. Evaluates on faces, churches and bedrooms.

- [Noisy Image Restoration Based on Conditional Acceleration Score Approximation](https://ieeexplore.ieee.org/abstract/document/10446531?casa_token=FBEMqWOGRTcAAAAA:S_ve43-219LPPCDaQc98ZG7Y15cvfUmCRijcFwz7eaq9Eqg7YWqDVtBHFCmR61HT3kTi7imKSw)  
Adapts the position-velocity-acceleration framework to inverse problems. Evaluates on faces and dogs. Nothing special, but beats the considered SOTA.

- [ODE-DPS: ODE-based Diffusion Posterior Sampling for Inverse Problems in Partial Differential Equation](https://arxiv.org/abs/2404.13496)  
Considers inverse problems in the context of partial differential equations. Adapts the DPS algorithm by adding adaptive step size and removing stochasticity

- [Improving Diffusion Models for Inverse Problems Using Optimal Posterior Covariance](https://arxiv.org/abs/2402.02149)  
Propose a formulation of recent diffusion-based inverse problem solvers in which they differ only in handcrafted design of isotropic posterior covariances. The authors propose to optimize the isotropic posterior covariance to further enhance the performance in inverse problems. They evaluate on ImageNet.

- [Image Restoration by Denoising Diffusion Models with Iteratively Preconditioned Guidance](https://arxiv.org/abs/2312.16519)  
During optimization, uses a sequence of preconditioners that smoothly translate the problem from back-projection to least-squares. Allows for DDPM and DDIM. 

- [Conditional Velocity Score Estimation for Image Restoration](https://openaccess.thecvf.com/content/WACV2024/html/Shi_Conditional_Velocity_Score_Estimation_for_Image_Restoration_WACV_2024_paper.html)  
Adapts the position-velocity diffusion framework to solve inverse problems. Does not evaluate on ImageNet.

- [Solving Inverse Problem With Unspecified Forward Operator Using Diffusion Models](https://openreview.net/forum?id=Ec2rYpP42y)  
Assumes no access to degradation operator and aims to recover it from the sample. 

- [Diffusion Posterior Sampling for Linear Inverse Problem Solving: A Filtering Perspective](https://openreview.net/forum?id=tplXNcHZs1)  
Another paper from Yang 'Diffusion God' Song. In general, it is based on a sampling a set of solutions to the inverse problem. 

- [From Posterior Sampling to Meaningful Diversity in Image Restoration](https://arxiv.org/abs/2310.16047)  
'In this paper, we initiate the study of meaningfully diverse image restoration. We explore several post-processing approaches that can be combined with any diverse image restoration method to yield semantically meaningful di- versity. Moreover, we propose a practical approach for allowing diffusion based image restoration methods to generate meaningfully diverse outputs, while incur- ring only negligent computational overhead.' How to effectively increase the diversity of inpaints?

- [Decomposed Diffusion Sampler for Accelerating Large-Scale Inverse Problems](https://arxiv.org/abs/2303.05754)  
Focuses on medical imaging inverse problems. Requires 20~50 NFE.

#### 2023

- [Conditional score-based generative models for solving physics-based inverse problems](https://openreview.net/forum?id=ZL5wlFMg0Y)  
Focuses on physics-based problems and requires training on a dataset with labeled conditions.

- [Quantized Generative Models for Solving Inverse Problems](https://openaccess.thecvf.com/content/ICCV2023W/RCV/html/Reddy_Quantized_Generative_Models_for_Solving_Inverse_Problems_ICCVW_2023_paper.html)  
How to effectively quantize generative models to preserve the quality of 32-bit models in inverse problems.

- [Beyond First-Order Tweedie: Solving Inverse Problems using Latent Diffusion](https://arxiv.org/abs/2312.00852). 
Extend first-order Tweedie to efficient second-order method with hessian estimator that requires only its trace. Inverse problems are solved in the latent space of an LDM.

- [INDigo: An INN-Guided Probabilistic Diffusion Algorithm for Inverse Problems](https://arxiv.org/abs/2306.02949)  
Attempts to solve problems without a closed-form expression of the degradation model. 

- [Inverse problem regularization with hierarchical variational autoencoders](https://arxiv.org/abs/2303.11217)  
Pretty fast, but the paper seems to somehow ignore the vast diffusion literature.

- [Direct Diffusion Bridge using Data Consistency for Inverse Problems](https://arxiv.org/abs/2305.19809). 
The authors propose a Consistent Direct Diffusion Bridge framework as a generalization of solving inverse problems with diffusion models. They show that NFE can be adapted to the budget at hand, and achieve great results in 20-1000 NFE range. Equation 15 concisely describes replacing x_t in guidance with estimated x_0. Great results on inpainting. Use ImageNet model from BeatGANs, i.e. classifier guidance.

- [Score-Based Diffusion Models as Principled Priors for Inverse Imaging](https://arxiv.org/abs/2304.11751)  
Requires a trained normalizing flow for sampling. No examples for inpainting.

- [Solving Inverse Problems with Latent Diffusion Models via Hard Data Consistency](https://arxiv.org/abs/2307.08123)  
The authors propose an approach to solving inverse problems in the latent space of an autoencoder, thus utilizing Latent Diffusion Models. To do that, they incorporate an additional optimization problem that finds the latent representation at a given timestep which, when decoded, approximates the measurement sufficiently well.

- [A Variational Perspective on Solving Inverse Problems with Diffusion Models](https://arxiv.org/abs/2305.04391)  
In addition to previous methods, this one seems to be very fast.

- [Pseudoinverse-Guided Diffusion Models for Inverse Problems](https://openreview.net/forum?id=9_gsMA8MRKQ)  
Extends to solving non-linear and non-differentiable inverse problems via pseudoinverse guidance term. It it also stochastic and possibly allows for class-conditioning with classifier guidance.

#### 2022

- [Parallel Diffusion Models of Operator and Image for Blind Inverse Problems](https://arxiv.org/abs/2211.10656)  
Non-blind means that the forward operator is known and blind means that it is not. This work addapts diffusion models to blind inverse problems.

- [Improving Diffusion Models for Inverse Problems using Manifold Constraints](https://arxiv.org/abs/2206.00941)  
Propose a simple improvement to general application of diffusion models for inverse problems. It is based on computing the gradient of the condition with the use of approximate final generation, similarly to replacing the input to classifier guidance with the approximation of the final step. Also a plus: 'The proposed method is inherently stochastic since the diffusion model is the main workhorse of the algorithm'.

- [Denoising Diffusion Restoration Models](https://arxiv.org/abs/2201.11793)  
Great approach for e.g. stochastic inpainting that allows for additional class conditioning. It speeds up the whole process significantly.

### Consistency Models

#### 2024

- [Easy Consistency Tuning](https://gsunshine.notion.site/Consistency-Models-Made-Easy-954205c0b4a24c009f78719f43b419cc) **WR=3**  
Great blog post that will probably be converted into a paper. It begins with an intuitive introduction to Consistency Models and proceeds with showing how the original framework can be improved by replacing distillation with fine-tuning of pretrained diffusion models.

- [Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion](https://arxiv.org/abs/2310.02279) **WR=3**    
A general framework that encompasses diffusion distillation techniques and consistency models, allowing for jumps from and to arbitrary timesteps of the PF ODE.

#### 2023

- [Improved Techniques for Training Consistency Models](https://arxiv.org/abs/2310.14189)  **WR=2**  
Follow-up from Yang Song showing some general improvements to original Consistency Models.

- [Consistency Models](https://arxiv.org/abs/2303.01469)  **WR=3**  
Yang Song introduces a new distillation technique for diffusion models together with a new class of generative models that builds upon it.

### Flow Matching

#### 2024

- [An Introduction to Flow Matching](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html#fn:log_pdf)  
Nice tutorial about Flow Matching from an ML research group at Cambridge. Interestingly, eq. 22 seems to recover I2SB with flow matching formulation.