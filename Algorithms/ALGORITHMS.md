#### Algorithms

We include 13 TST algorithms for evaluation in our benchmark, as shown in the Table. 

![image](https://github.com/FayeXXX/A-Benchmark-of-Text-Style-Transfer/blob/main/Algorithms/algorithms.png)

These algorithms are either well-known or the state-of-the art, representing the latest advancements in the field. Specifically, seven algorithms (STYTRANS, TSST, NAST, STRAP, BSRR, TYB and CTAT) are classic methods that span three strategies of FPFT. 

The remaining six algorithms (LORA, LORA-INST, CHATGPT, CHATGPT-FS, GPT4, GPT4-FS) are recently published methods based on LLMs.

Thanks to these open source codes, we reimplement these algorithms based on the original repo of each paper and the citations are as follows:

------

**STYTRANS**

@article{2019style,
  title={Style transformer: Unpaired text style transfer without disentangled latent representation},

  author={Dai, Ning and Liang, Jianze and Qiu, Xipeng and Huang, Xuanjing},

  journal={arXiv preprint arXiv:1905.05621},

  year={2019}
}

https://github.com/MarvinChung/HW5-TextStyleTransfer/tree/master

------

**TSST**

@article{xiao2021transductive,
  title={Transductive learning for unsupervised text style transfer},
  author={Xiao, Fei and Pang, Liang and Lan, Yanyan and Wang, Yan and Shen, Huawei and Cheng, Xueqi},
  journal={arXiv preprint arXiv:2109.07812},
  year={2021}
}

https://github.com/xiaofei05/TSST 

------

**NAST**

@article{huang2021nast,
  title={NAST: A non-autoregressive generator with word alignment for unsupervised text style transfer},
  author={Huang, Fei and Chen, Zikai and Wu, Chen Henry and Guo, Qihan and Zhu, Xiaoyan and Huang, Minlie},
  journal={arXiv preprint arXiv:2106.02210},
  year={2021}
}

https://github.com/thu-coai/NAST

_____

**STRAP**

@article{krishna2020reformulating,
  title={Reformulating unsupervised style transfer as paraphrase generation},
  author={Krishna, Kalpesh and Wieting, John and Iyyer, Mohit},
  journal={arXiv preprint arXiv:2010.05700},
  year={2020}
}

https://github.com/martiansideofthemoon/style-transfer-paraphrase

____

**BSRR**

@article{liu2022learning,
  title={Learning from Bootstrapping and Stepwise Reinforcement Reward: A Semi-Supervised Framework for Text Style Transfer},
  author={Liu, Zhengyuan and Chen, Nancy F},
  journal={arXiv preprint arXiv:2205.09324},
  year={2022}
}

https://github.com/seq-to-mind/semi-style-transfer

____

**TYB** 

@article{lai2021thank,
  title={Thank you BART! rewarding pre-trained models improves formality style transfer},
  author={Lai, Huiyuan and Toral, Antonio and Nissim, Malvina},
  journal={arXiv preprint arXiv:2105.06947},
  year={2021}
}

https://github.com/laihuiyuan/pre-trained-formality-transfer

____

**CTAT**

@article{wang2019controllable,
  title={Controllable unsupervised text attribute transfer via editing entangled latent representation},
  author={Wang, Ke and Hua, Hang and Wan, Xiaojun},
  journal={Advances in Neural Information Processing Systems},
  volume={32},
  year={2019}
}

https://github.com/Nrgeup/controllable-text-attribute-transfer

_____

**LORA**

@article{hu2021lora,
  title={Lora: Low-rank adaptation of large language models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}

____

**CHATGPT**

@misc{openai,
    author = "OpenAI",
    year = {2022},
    title = "Introducing chatgpt",
    url = "https://openai.com/blog/chatgpt"
}
