# Competitive Ranking Analysis

A Big Data and Network Analysis project focused on understanding player performance, rating progression, and behavioral patterns in competitive multiplayer environments.

This repository contains the datasets, preprocessing pipeline, analysis notebooks, and final report developed for the *Big Data and Network Analysis* course.

---

# Overview

Competitive ranking systems continuously generate large amounts of behavioral and performance data. These environments provide an interesting opportunity to investigate how player performance, activity, and progression interact within a competitive ecosystem.

The objective of this project is to analyze two independent competitive communities and study:

* the relationship between player performance and ranking (MMR);
* the evolution of player ratings over time;
* the existence of distinct player archetypes;
* similarities and differences between separate competitive communities;
* the effectiveness and stability of rating-based ranking systems.

The project follows the **CRISP-DM (Cross Industry Standard Process for Data Mining)** methodology.

---

# Communities Analyzed

The analysis is based on data collected from two independent competitive communities:

| Community | Ranked Players | Historical Rating Records |
| --------- | -------------- | ------------------------- |
| Soccer    | 107            | 20,155                    |
| BlockBall | 97             | 14,275                    |
| **Total** | **204**        | **34,430**                |

Although developed independently, both communities use rating-based matchmaking systems and collect similar competitive statistics, making comparative analysis possible.

---

# Available Data

## Player-Level Statistics

Each player is represented by a set of aggregated performance indicators including:

* current MMR
* peak MMR
* matches played
* wins
* losses
* draws
* goals
* assists
* saves
* passes (when available)

Additional normalized features include:

* win rate
* goals per match
* assists per match
* saves per match
* passes per match

---

## Rating History

Historical datasets contain temporal observations describing rating progression.

Each record includes:

* anonymized player identifier
* timestamp
* days since first observation
* historical MMR value

These datasets allow longitudinal analysis of player progression and rating stability.

---

# Data Preparation

The preprocessing pipeline includes:

### Anonymization

All original identifiers were removed.

Players are represented only through artificial identifiers such as:

```text
Player_001
Player_002
Player_003
```

### Cleaning

* removal of inactive players
* handling of missing values
* validation of anomalous rating values
* consistency checks across datasets

### Transformation

* conversion of timestamps to datetime format
* creation of temporal progression variables
* restructuring of player statistics through pivot operations when necessary

### Feature Engineering

Normalized performance indicators were generated to allow fair comparisons between players with different activity levels.

Examples include:

* Win Rate
* Goals per Match
* Assists per Match
* Saves per Match
* Passes per Match

---

# Methodology

The project follows the CRISP-DM framework:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation

The analytical phase combines classical statistical methods and machine learning techniques.

---

# Analytical Techniques

The following methods are applied throughout the project:

## Descriptive Statistics

Used to summarize the main characteristics of the datasets and understand variable distributions.

Examples:

* mean
* median
* standard deviation
* distribution analysis
* outlier detection

## Correlation Analysis

Used to investigate relationships between performance indicators and player rating.

Examples:

* MMR vs Win Rate
* MMR vs Goals per Match
* MMR vs Assists per Match

## Principal Component Analysis (PCA)

Used to reduce dimensionality and identify the most relevant sources of variation within player statistics.

## Clustering

Applied to identify groups of players with similar statistical profiles and behavioral patterns.

## Temporal Analysis

Used to study:

* rating progression
* rating stability
* long-term competitive evolution

## Community Comparison

Comparative analyses are performed between Soccer and BlockBall in order to identify common and community-specific patterns.

---

#

---

# Privacy

Only anonymized datasets are publicly available.

The repository does **not** include:

* original UUIDs
* usernames
* internal mapping tables
* private lookup files

All analyses are performed exclusively on anonymized player identifiers.

---

# Reproducibility

The complete preprocessing pipeline is included in the repository.

Starting from the raw exports, the provided scripts generate the cleaned datasets used throughout the analysis.

This ensures that all results can be reproduced and independently verified.

---

# Future Work

Possible extensions include:

* predictive modeling of rating progression;
* match-level performance analysis;
* network-related performance studies;
* advanced temporal modeling;
* comparison with additional competitive communities.

---

# Author

**Mattia Marras**

Physics & Applied Computer Science and Artificial Intelligence

Sapienza University of Rome
