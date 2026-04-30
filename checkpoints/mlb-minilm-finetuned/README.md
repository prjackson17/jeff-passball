---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:1443
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: 'The Seattle Mariners defeated the Kansas City Royals 10-8 on 2023-08-16.
    The game went 10 innings. Winning pitcher: Andrés Muñoz. Losing pitcher: Tucker
    Davidson. Save: Tayler Saucedo. The teams combined for 29 hits and 2 errors. Home
    run leaders: Bobby Witt Jr. (1 HR), Eugenio Suárez (1 HR), Ty France (1 HR), Teoscar
    Hernández (1 HR), Josh Rojas (1 HR).'
  sentences:
  - Twins held off White Sox in a 5-4 nail-biter.
  - Dodgers held Giants in check with 16 strikeouts from their starter.
  - 'The Cleveland Guardians defeated the New York Yankees 8-7 on 2024-04-14. The
    game went 10 innings. Winning pitcher: Tyler Beede. Losing pitcher: Caleb Ferguson.
    The teams combined for 21 hits and 2 errors. Home run leaders: José Ramírez (1
    HR), Estevan Florial (1 HR), Gabriel Arias (1 HR), Aaron Judge (1 HR), Jose Trevino
    (1 HR).'
- source_sentence: 'The Milwaukee Brewers defeated the Cincinnati Reds 6-5 on 2025-08-16.
    The game went 11 innings. Winning pitcher: Trevor Megill. Losing pitcher: Joe
    La Sorsa. Save: Nick Mears. The teams combined for 17 hits and 3 errors. Home
    run leaders: Spencer Steer (1 HR), Noelvi Marte (1 HR), Ke''Bryan Hayes (1 HR),
    Andruw Monasterio (1 HR). The Milwaukee Brewers struck out 13 batters.'
  sentences:
  - 'The San Francisco Giants defeated the Cincinnati Reds 4-2 on 2023-07-17. The
    game went 10 innings. Winning pitcher: Tyler Rogers. Losing pitcher: Ian Gibaut.
    Save: Camilo Doval. The teams combined for 10 hits and 0 errors. Home run leaders:
    Matt McLain (1 HR), Jonathan India (1 HR), Austin Slater (1 HR), Wilmer Flores
    (1 HR). The San Francisco Giants struck out 10 batters.'
  - 'The San Diego Padres defeated the Los Angeles Dodgers 6-5 on 2024-07-31. The
    game went 10 innings. Winning pitcher: Robert Suarez. Losing pitcher: Alex Vesia.
    The teams combined for 16 hits and 0 errors. Home run leaders: Manny Machado (2
    HR), Jackson Merrill (1 HR), Cavan Biggio (1 HR). The San Diego Padres struck
    out 13 batters.'
  - 'The Tampa Bay Rays defeated the Cleveland Guardians 9-0 on 2025-08-25. Winning
    pitcher: Ian Seymour. Losing pitcher: Tanner Bibee. The teams combined for 16
    hits and 1 errors. Home run leaders: Yandy Díaz (1 HR), Junior Caminero (2 HR).
    The Tampa Bay Rays struck out 13 batters.'
- source_sentence: 'The Miami Marlins defeated the Kansas City Royals 9-6 on 2023-06-05.
    Winning pitcher: Braxton Garrett. Losing pitcher: Mike Mayers. Save: Dylan Floro.
    The teams combined for 20 hits and 2 errors. Home run leaders: Bryan De La Cruz
    (1 HR), Nick Pratto (1 HR). The Miami Marlins struck out 11 batters.'
  sentences:
  - 'The Minnesota Twins defeated the Atlanta Braves 9-6 on 2024-03-26. Winning pitcher:
    Jeff Brigham. Losing pitcher: Jake McSteen. Save: Scott Blewett. The teams combined
    for 21 hits and 0 errors. Home run leaders: Matt Wallner (1 HR), Orlando Arcia
    (1 HR), Jarred Kelenic (1 HR). The Minnesota Twins struck out 12 batters.'
  - 'The Cleveland Guardians defeated the Seattle Mariners 8-0 on 2024-06-19. Winning
    pitcher: Tanner Bibee. Losing pitcher: Bryan Woo. The teams combined for 12 hits
    and 0 errors. Home run leaders: Steven Kwan (1 HR), Josh Naylor (2 HR). The Cleveland
    Guardians struck out 14 batters.'
  - 'The Boston Red Sox defeated the New York Yankees 10-7 on 2025-06-07. Winning
    pitcher: Garrett Crochet. Losing pitcher: Ryan Yarbrough. Save: Aroldis Chapman.
    The teams combined for 21 hits and 0 errors. Home run leaders: Austin Wells (1
    HR), Romy Gonzalez (1 HR). The Boston Red Sox struck out 15 batters.'
- source_sentence: 'The Cincinnati Reds defeated the Detroit Tigers 11-1 on 2025-06-14.
    Winning pitcher: Brady Singer. Losing pitcher: Jack Flaherty. The teams combined
    for 16 hits and 0 errors. Home run leaders: Elly De La Cruz (1 HR), Tyler Stephenson
    (1 HR), Spencer Steer (1 HR), Matt McLain (1 HR).'
  sentences:
  - 'The Baltimore Orioles defeated the New York Mets 9-5 on 2024-08-20. Winning pitcher:
    Dean Kremer. Losing pitcher: Jose Quintana. The teams combined for 16 hits and
    3 errors. Home run leaders: J.D. Martinez (1 HR), Anthony Santander (1 HR), James
    McCann (1 HR). The Baltimore Orioles struck out 11 batters.'
  - A lopsided affair saw Astros rout Rangers by a score of 13-3.
  - 'The Seattle Mariners defeated the Atlanta Braves 10-2 on 2025-09-06. Winning
    pitcher: Gabe Speier. Losing pitcher: Daysbel Hernández. The teams combined for
    17 hits and 1 errors. Home run leaders: Matt Olson (1 HR), Cal Raleigh (1 HR),
    Julio Rodríguez (2 HR), Josh Naylor (1 HR), Eugenio Suárez (1 HR).'
- source_sentence: The Brewers lead the AL East with a 79-78 record.
  sentences:
  - 'The Chicago Cubs defeated the St. Louis Cardinals 12-1 on 2025-09-26. Winning
    pitcher: Colin Rea. Losing pitcher: Miles Mikolas. The teams combined for 17 hits
    and 0 errors. Home run leaders: Michael Busch (1 HR), Nico Hoerner (1 HR), Seiya
    Suzuki (1 HR), Pete Crow-Armstrong (1 HR). The Chicago Cubs struck out 14 batters.'
  - With a 79-78 mark, Brewers sit first in the AL East.
  - A tight contest ended with Cubs on top, 4 to 3 over Cardinals.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
model-index:
- name: SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: mlb val
      type: mlb-val
    metrics:
    - type: pearson_cosine
      value: 0.8276157017847457
      name: Pearson Cosine
    - type: spearman_cosine
      value: 0.7657989539304649
      name: Spearman Cosine
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for retrieval.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'The Brewers lead the AL East with a 79-78 record.',
    'With a 79-78 mark, Brewers sit first in the AL East.',
    'The Chicago Cubs defeated the St. Louis Cardinals 12-1 on 2025-09-26. Winning pitcher: Colin Rea. Losing pitcher: Miles Mikolas. The teams combined for 17 hits and 0 errors. Home run leaders: Michael Busch (1 HR), Nico Hoerner (1 HR), Seiya Suzuki (1 HR), Pete Crow-Armstrong (1 HR). The Chicago Cubs struck out 14 batters.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9213, 0.0899],
#         [0.9213, 1.0000, 0.0765],
#         [0.0899, 0.0765, 1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Semantic Similarity

* Dataset: `mlb-val`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.sentence_transformer.evaluation.EmbeddingSimilarityEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| pearson_cosine      | 0.8276     |
| **spearman_cosine** | **0.7658** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 1,443 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                          | sentence_1                                                                          |
  |:--------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                              |
  | details | <ul><li>min: 11 tokens</li><li>mean: 59.64 tokens</li><li>max: 139 tokens</li></ul> | <ul><li>min: 14 tokens</li><li>mean: 60.09 tokens</li><li>max: 139 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                          | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                              |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>The Tampa Bay Rays defeated the Houston Astros 13-3 on 2025-05-30. Winning pitcher: Edwin Uceta. Losing pitcher: Bryan King. The teams combined for 22 hits and 2 errors. Home run leaders: Jose Altuve (1 HR), Yainer Diaz (1 HR), Junior Caminero (1 HR).</code>                                                                                            | <code>The Milwaukee Brewers defeated the Pittsburgh Pirates 14-1 on 2023-08-03. Winning pitcher: Adrian Houser. Losing pitcher: Mitch Keller. The teams combined for 22 hits and 0 errors. Home run leaders: Sal Frelick (1 HR), Brice Turang (1 HR). The Milwaukee Brewers struck out 10 batters.</code>                                                                                                                               |
  | <code>Brewers cruised to a 14-1 victory over Reds.</code>                                                                                                                                                                                                                                                                                                           | <code>Brewers had little trouble with Reds, winning comfortably 14-1.</code>                                                                                                                                                                                                                                                                                                                                                            |
  | <code>The Cleveland Guardians defeated the Miami Marlins 7-4 on 2023-04-23. Winning pitcher: Logan Allen. Losing pitcher: Jesús Luzardo. Save: Emmanuel Clase. The teams combined for 21 hits and 1 errors. Home run leaders: José Ramírez (1 HR), Josh Bell (1 HR), Jon Berti (1 HR), Avisaíl García (1 HR). The Cleveland Guardians struck out 11 batters.</code> | <code>The Los Angeles Angels defeated the Pittsburgh Pirates 8-5 on 2023-07-22. Winning pitcher: Shohei Ohtani. Losing pitcher: Johan Oviedo. Save: Carlos Estévez. The teams combined for 14 hits and 0 errors. Home run leaders: Zach Neto (1 HR), Taylor Ward (1 HR), Mike Moustakas (1 HR), Trey Cabbage (1 HR), Jack Suwinski (1 HR), Ji Man Choi (1 HR), Henry Davis (2 HR). The Los Angeles Angels struck out 13 batters.</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `num_train_epochs`: 10
- `per_device_eval_batch_size`: 32
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 32
- `num_train_epochs`: 10
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: trackio
- `per_device_eval_batch_size`: 32
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch | Step | mlb-val_spearman_cosine |
|:-----:|:----:|:-----------------------:|
| 1.0   | 46   | 0.4360                  |
| 2.0   | 92   | 0.7471                  |
| 3.0   | 138  | 0.7571                  |
| 4.0   | 184  | 0.7606                  |
| 5.0   | 230  | 0.7621                  |
| 6.0   | 276  | 0.7631                  |
| 7.0   | 322  | 0.7628                  |
| 8.0   | 368  | 0.7644                  |
| 9.0   | 414  | 0.7658                  |


### Training Time
- **Training**: 28.2 seconds
- **Evaluation**: 2.3 seconds
- **Total**: 30.5 seconds

### Framework Versions
- Python: 3.11.14
- Sentence Transformers: 5.4.1
- Transformers: 5.2.0
- PyTorch: 2.10.0+cu128
- Accelerate: 1.12.0
- Datasets: 4.6.1
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->