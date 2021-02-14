# Referring Expression Generation Using Entity Profiles

This is the official repository for Referring Expression Generation Using Entity Profiles.

# Citation
========
If you find this useful in your research, please consider citing:
```
@inproceedings{cao-cheung-2019-referring,
    title = "Referring Expression Generation Using Entity Profiles",
    author = "Cao, Meng  and
      Cheung, Jackie Chi Kit",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1312",
    doi = "10.18653/v1/D19-1312",
    pages = "3163--3172",
    abstract = "Referring Expression Generation (REG) is the task of generating contextually appropriate references to entities. A limitation of existing REG systems is that they rely on entity-specific supervised training, which means that they cannot handle entities not seen during training. In this study, we address this in two ways. First, we propose task setups in which we specifically test a REG system{'}s ability to generalize to entities not seen during training. Second, we propose a profile-based deep neural network model, ProfileREG, which encodes both the local context and an external profile of the entity to generate reference realizations. Our model generates tokens by learning to choose between generating pronouns, generating from a fixed vocabulary, or copying a word from the profile. We evaluate our model on three different splits of the WebNLG dataset, and show that it outperforms competitive baselines in all settings according to automatic and human evaluations.",
}
```
