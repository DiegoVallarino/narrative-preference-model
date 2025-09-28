# Narrative Preference Model

This repository contains a simulation-based Python model to estimate narrative preferences in British crime series. It combines interpretable structural features (nonlinearity, early character intro, surprise, logic) with latent semantic embeddings using a Mixture of Experts architecture. Visual output is generated with t-SNE.

**Disclaimer**: This is a personal project and not affiliated with any institution, including the Inter-American Development Bank (IDB). It is an academic and data modeling exercise reflecting personal preferences only.

##  Features

- Simulated dataset with 200 fictional series
- Mixture of Experts model to simulate audience taste heterogeneity
- Pairwise comparisons for preference learning
- t-SNE visualization of latent narrative space

##  Output

- `latent_preferences_plot.png` saved to Desktop
- `pairwise_comparisons.csv` for logistic regression or analysis

## Project Overview

The model formalizes revealed preferences over narrative structures based on:
- Early character introduction
- Intrigue pacing
- Logical resolution
- Surprise calibration
- Non-linear storytelling (temporal disorder)

We simulate narrative titles and learn a latent utility function via a **Mixture of Experts**, visualized through **t-SNE** projections.

---

## Core Methodology

- Simulated dataset of 200 narrative series
- Structural variables + semantic embeddings (`sklearn`, `torch`)
- Mixture of Experts for heterogeneity modeling
- Pairwise preference generation (Bradley–Terry logic)
- Visualization via `matplotlib` + `t-SNE`

---

## Code Structure

```
narrative-preference-model/
├── narrative_preference_model.py       # Main script: data generation, modeling, visualization
├── requirements.txt                    # Python packages needed to run the model
├── pairwise_comparisons.csv            # Optional: Exported comparisons for estimation
├── latent_preferences_plot.png         # Output figure (saved to Desktop by default)
└── README.md                           # This file
```

---

##  License

This project is licensed under the MIT License.

##  Contact

If you have questions, suggestions, or would like to collaborate, feel free to reach out:

- **Email:** [diego.vallarino@gmail.com](mailto:diego.vallarino@gmail.com)  
- **Website:** [www.diegovallarino.com](https://www.diegovallarino.com)

