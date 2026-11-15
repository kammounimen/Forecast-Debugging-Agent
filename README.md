# Forecast Debugging Agent

**Système multi-agents basé sur les LLM et le RAG pour analyser et expliquer les écarts entre les prévisions Batch et Real-Time.**

Auteures : Imen Kammoun et Mariam Kbaier

Plateforme : Databricks Free Edition (compute serverless, Python / SQL / PySpark)

---

## 1. Contexte et objectif

Une entreprise produit deux types de prévisions de ventes :

- **Batch** : calcul périodique à partir d'un *snapshot* de données figé.
- **Real-Time** : calcul à la demande, avec des données potentiellement plus récentes.

Quand les deux résultats diffèrent (exemple : 1 000 contre 1 350 unités), l'enquête manuelle est lente et dépend de l'expertise de quelques personnes.

**Objectif du projet** : construire un système qui mène cette enquête automatiquement avec des outils en lecture seule, identifie la **cause racine**, fournit des **preuves**, un **niveau de confiance** et une **recommandation**.

### Architecture cible (3 agents + RAG)

```
Question : "Pourquoi Batch et Real-Time diffèrent ?"
            |
   +--------+---------+        (exécutés en parallèle)
   |                  |
Agent 1            Agent 2
Investigateur      Connaissance (RAG)
5 outils de        Recherche hybride dans
données (SELECT)   la documentation métier
   |                  |
   +--------+---------+
            |
        Agent 3
   Analyste et Vérificateur
   (croise, vérifie les preuves, conclut)
            |
   Rapport final : cause, preuves, confiance, recommandation
```

L'orchestration sera faite avec LangGraph.

### Principe de conception

> **Le code calcule, le LLM raisonne.** Les écarts, âges de snapshots et différences sont calculés de façon déterministe (PySpark / SQL). Le LLM reçoit des résultats chiffrés, pas des milliers de lignes, et rédige l'interprétation.

---

## 2. État d'avancement

| Étape | Contenu | État |
|---|---|---|
| 1 | Environnement Databricks, schéma et volume | Fait |
| 2 | Dataset synthétique (7 tables Delta) | Fait |
| 3 | Moteur de comparaison PySpark (4 tables `cmp_*`) | Fait |
| 4 | Exploration des données et fiches d'enquête | Fait |
| 5 | Baseline par règles (sans LLM) | Fait : 12/12 |
| 6 | 5 outils de l'agent en lecture seule + tests | Fait : tests OK |
| 7 | RAG : TF-IDF puis recherche hybride BM25 + embeddings | Fait |
| 8 | Connexion au LLM et tool calling | À faire |
| 9 | Agent unique (configuration A) | À faire |
| 10 | 3 agents avec LangGraph (configurations B et C) | À faire |
| 11 | Dataset v2 plus difficile, découpage développement/test | À faire |
| 12 | Évaluation A/B/C, analyse d'erreurs, rapport | À faire |

---

## 3. Structure du dépôt

```
Forecast-Debugging-Agent/
├── README.md
├── data/                 6 fichiers CSV du dataset
├── rag_docs/             10 documents Markdown du RAG
└── notebooks/
    ├── 00_setup
    ├── 00_load_data
    ├── 01_comparison_engine_pyspark
    ├── 02_data_exploration
    ├── 02b_fiches_enquete
    ├── 03_rule_based_baseline
    ├── 04_agent_tools
    ├── 04b_test_tools
    ├── 05_rag
    ├── 05b_test_rag
    └── 05d_rag_hybrid
```

Les notebooks doivent être placés dans le **même dossier** du workspace, car les tests utilisent `%run ./04_agent_tools`.

---

## 4. Les données

Les données sont **synthétiques** : on ne sait pas si les données réelles de l'entreprise seront accessibles, et des cas à cause connue permettent de mesurer l'agent.

Catalogue : `workspace`, schéma : `forecast_debug`, volume : `raw` (dossiers `data/` et `rag_docs/`).

### Tables sources (chargées par `00_load_data`)

| Table | Lignes | Contenu |
|---|---|---|
| `forecast_predictions` | 24 | Prévision de chaque système pour chaque cas (12 cas x 2) |
| `feature_snapshots` | 96 | 4 features par système : `sales_7d`, `promotion`, `price`, `stock_level` |
| `model_configs` | 96 | 4 configurations : `forecast_horizon_days`, `model_version`, `feature_set_version`, `preprocessing` |
| `pipeline_runs` | 24 | Heure d'exécution, snapshot de données, statut, warnings, versions |
| `actual_sales` | 12 | Ventes réelles (non utilisées par les outils) |
| `known_issues` | 12 | **Vérité terrain, réservée à l'évaluation. Les agents n'y ont jamais accès.** |
| `rag_documents` | 10 | Documents texte pour le RAG |

### Tables de comparaison (créées par `01_comparison_engine_pyspark`)

| Table | Lignes | Contenu |
|---|---|---|
| `cmp_forecast_gap` | 12 | Écart en unités et en %, indicateur `within_tolerance` (seuil 5 %) |
| `cmp_inputs` | 96 | Différences de features et de configs entre Batch et Real-Time |
| `cmp_freshness` | 12 | Âge des snapshots en heures, statut de fraîcheur (seuil 24 h) |
| `cmp_case_summary` | 12 | Synthèse par cas |

### Les 12 cas

| Cas | Batch | Real-Time | Cause attendue |
|---|---|---|---|
| 001 | 1000 | 1200 | Données différentes (`sales_7d` : 100 vs 120) |
| 002 | 1000 | 1350 | Feature différente (`promotion` : 0 vs 1) |
| 003 | 1100 | 1600 | Configuration (horizon 7 vs 14 jours) |
| 004 | 900 | 1200 | Données obsolètes (snapshot Batch du 10/11, 50 h) |
| 005 | 1050 | 1070 | Aucun problème (écart 1,9 %) |
| 006 | 800 | 1150 | Deux causes : promotion et feature set FS1 vs FS2 |
| 007 | 1500 | 1555 | Aucun problème, piège : 6 h de décalage de snapshot |
| 008 | 1200 | 1500 | Version du modèle (V2 vs V3) |
| 009 | 950 | 1300 | Données obsolètes, `sales_7d` différent n'est qu'un **symptôme** |
| 010 | 700 | 710 | Aucun problème |
| 011 | 1300 | 1000 | Feature différente (`price` : 20 vs 24) |
| 012 | 600 | 820 | Configuration (`preprocessing`) |

Cas difficiles : **006** (deux causes), **007** (faux signal d'alerte), **009** (cause cachée derrière un symptôme).

---

## 5. Ce qui a été réalisé, étape par étape

### 5.1 Préparation de l'environnement (`00_setup`)
Création du schéma `forecast_debug` dans le catalogue `workspace`, puis du volume `raw` avec les dossiers `data/` et `rag_docs/`.

### 5.2 Chargement des données (`00_load_data`)
Lit les CSV du volume et crée les 6 tables Delta, plus `rag_documents`. Contrôle : 24, 96, 96, 24, 12, 12 lignes et 10 documents.

### 5.3 Moteur de comparaison (`01_comparison_engine_pyspark`)
Calcule en PySpark les 4 tables `cmp_*`. Scala n'est pas supporté sur la Free Edition ; une version Scala existe comme bonus pour un environnement qui le permet, mais le chemin principal est PySpark.

### 5.4 Exploration (`02_data_exploration`, `02b_fiches_enquete`)
Requêtes SQL et fonction `fiche(case_id)` qui répond à 5 questions pour un cas : l'écart, la fraîcheur, les features qui diffèrent, les configs qui diffèrent, le warning du pipeline. Plusieurs cas ont été résolus à la main avant toute IA.

### 5.5 Baseline sans LLM (`03_rule_based_baseline`)
Fonction `investigate_case(case_id)` qui applique des règles fixes :
- snapshot de plus de 24 h : `STALE_DATA`, et les différences de `sales_7d` deviennent des symptômes ;
- écart de 5 % ou moins sans différence d'entrée : `NO_ISSUE` ;
- sinon, la cause est déduite de la nature de la différence (feature, config, version du modèle).

**Résultat : 12/12 (100 %).**
Ce score est à lire avec prudence : les règles ont été écrites en connaissant les 12 cas, et chaque cas contient une cause bien visible. Il valide la chaîne de données et d'évaluation, mais il ne prouve pas que le système tiendrait sur des cas ambigus. Il montre aussi que ce dataset est trop facile pour démontrer la valeur d'un LLM, d'où le dataset v2 prévu.

### 5.6 Les 5 outils de l'agent (`04_agent_tools`, `04b_test_tools`)

| Outil | Retourne |
|---|---|
| `get_forecasts(case_id)` | Prévisions Batch et Real-Time, écart en unités et en % |
| `compare_features(case_id)` | Les 4 features et celles qui diffèrent |
| `compare_configs(case_id)` | Les 4 configs et celles qui diffèrent |
| `check_freshness(case_id)` | Âge des snapshots en heures |
| `get_pipeline_logs(case_id)` | Statut, warnings, versions, durée |

Choix de conception :
- **Lecture seule**, aucune requête SQL libre.
- **Liste blanche de tables** : `known_issues`, `actual_sales` et `rag_documents` sont refusées, et un test le vérifie.
- **Les outils renvoient des faits, pas de verdict.** `check_freshness` donne « 50 h », pas « obsolète ». Les règles (24 h, 5 %) vivent dans les documents RAG, ce qui donne un rôle réel au RAG et permet de mesurer son apport.
- Un `case_id` invalide renvoie une erreur propre.

Un bug a été trouvé et corrigé grâce au test sur Databricks : un filtre de colonne incompatible avec Spark Connect, remplacé par `F.col("case_id")`.

### 5.7 Le RAG

**Corpus** : 10 documents synthétiques (vue d'ensemble des pipelines, règles de fraîcheur, tolérance d'écart, définitions des features, versions et configs, 3 incidents passés, runbook, glossaire des causes). Découpés par section en **17 passages**.

**Version 1 (`05_rag`, `05b_test_rag`)** : recherche TF-IDF avec scikit-learn.

**Version 2 (`05d_rag_hybrid`)** : recherche hybride, entièrement gratuite.

| Composant | Rôle |
|---|---|
| BM25 (`rank-bm25`) | Recherche par mots-clés, accents retirés |
| Embeddings (`sentence-transformers`, modèle multilingue `paraphrase-multilingual-MiniLM-L12-v2`) | Recherche par sens, adaptée aux documents français |
| Reciprocal Rank Fusion (RRF) | Fusion des deux classements, sans poids à régler |

L'outil exposé à l'agent est `search_knowledge(query, k=3)`.

**Mesure** (Recall@3, la clé attendue doit apparaître dans le `doc_id` d'un des 3 premiers résultats) :

| Méthode | Score sur 10 requêtes |
|---|---|
| BM25 | 7/10 |
| Embeddings | 9/10 |
| Hybride (RRF) | 10/10 |

Limites de cette mesure : seulement 10 requêtes, écrites en connaissant les documents, sur 17 passages. Le test est approximatif (il compare un mot-clé au `doc_id`). Un écart d'un point n'est pas significatif ; la tendance (l'hybride au moins aussi bon que chaque méthode seule) est cohérente mais reste à confirmer sur plus de requêtes. Deux membres de l'équipe ont obtenu des scores différents parce qu'elles avaient des listes de test de tailles différentes (7 et 10 requêtes), pas à cause du code.

Non exécuté : `05c_vector_search` (Databricks Vector Search) a été proposé mais n'a pas été lancé ; le choix gratuit retenu est la recherche hybride ci-dessus.

---

## 6. Comment reproduire

1. Créer un compte **Databricks Free Edition**.
2. Créer un notebook `00_setup` (Python, Serverless) et exécuter la création du schéma `forecast_debug`.
3. Dans **Catalog**, créer le volume `raw`, avec les dossiers `data/` et `rag_docs/`, puis y téléverser les 6 CSV et les 10 fichiers `.md`.
4. Importer les notebooks dans **un même dossier** du workspace.
5. Exécuter dans l'ordre :
   1. `00_load_data` : 6 tables + `rag_documents`
   2. `01_comparison_engine_pyspark` : 4 tables `cmp_*`
   3. `03_rule_based_baseline` : doit afficher `Accuracy baseline règles: 12/12`
   4. `04b_test_tools` : doit afficher `TOUS LES TESTS OK`
   5. `05d_rag_hybrid` : exécuter cellule par cellule (installation, redémarrage de Python, puis le reste)

---

## 7. Prochaines étapes

1. **Connexion au LLM** : lister les endpoints disponibles dans le workspace, tester un appel simple puis le *tool calling* (`06_llm_connection_test`).
2. **Agent unique (configuration A)** : 5 outils de données + `search_knowledge`, sortie JSON imposée : `root_cause`, `secondary_causes`, `evidence`, `confidence`, `recommendation`.
3. **Trois agents avec LangGraph** : Investigateur et Connaissance en parallèle, puis Analyste/Vérificateur qui contrôle que chaque affirmation cite un résultat d'outil ou un document.
   - Configuration B : 3 agents sans RAG
   - Configuration C : 3 agents avec RAG
4. **Dataset v2** : 60 à 100 cas avec bruit, cas ambigus, cas hors règles, cas « preuves insuffisantes », et un découpage **développement / test** (le groupe de test n'est jamais utilisé pendant le réglage).
5. **Évaluation** : comparer baseline par règles, A, B et C sur le même jeu de test, avec plusieurs exécutions chacune. Métriques : accuracy sur la cause principale, précision et rappel (y compris causes secondaires), qualité des preuves, temps d'investigation, nombre d'appels LLM.
6. **Analyse d'erreurs** sur les cas 006, 007 et 009, puis rapport final.

---

## 8. Limites

- **Données synthétiques** : les résultats valideront le système sur des scénarios contrôlés, pas en production. On ne dira jamais « 95 % en production », mais « X % sur N scénarios contrôlés ».
- **Dataset actuel très petit et propre** : 12 cas, une ou deux causes visibles par cas, valeurs rondes. Les règles font déjà 100 %, donc il ne suffit pas pour montrer l'apport d'un LLM.
- **Documents RAG et cas écrits ensemble**, avec les mêmes seuils (24 h, 5 %) : le RAG peut paraître plus utile que sur de vraies règles métier.
- **Évaluation du RAG** sur un petit nombre de requêtes écrites par l'équipe.
- **Free Edition** : accès internet sortant limité, quotas de compute, modèles LLM disponibles à vérifier.
- Plus d'agents implique plus de latence et de coût : l'évaluation A/B/C doit dire honnêtement quand le multi-agents et le RAG apportent quelque chose, et quand ils ajoutent seulement du coût.

---

## 9. Règles de travail

- Valider les données et les outils avant de toucher au LLM.
- Ne jamais donner accès à `known_issues` aux agents.
- Ne jamais modifier les données pour que l'agent réussisse.
- Pull avant de travailler, commit et push en fin de session.