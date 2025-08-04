## Fair RLHF 

We propose two optimizations for improving fairness in RLHF -- an exploratory intrinsic reward in the PPO training of LLMs (explored in past papers), 
and an info-theoretic constraint enforcing invariance of the reward model distribution with respect to different categories of bias (our novelty) 

To run our fair RLHF reward model training pipeline, 
```./train_rm.sh ```