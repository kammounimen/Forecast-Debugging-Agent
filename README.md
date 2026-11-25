# Forecast Debugging Agent

**Système multi-agents basé sur les LLM et le RAG pour analyser et expliquer les écarts entre les prévisions Batch et Real-Time, avec une application web de restitution.**

Auteures : Imen Kammoun et Mariam Kbaier
Plateforme : Databricks Free Edition (compute serverless) | Langage : Python, SQL, PySpark
LLM : `databricks-gpt-oss-120b` (endpoint Databricks, aucune clé externe) | Orchestration : LangGraph | Application : Streamlit sur Databricks Apps

> Résultats détaillés, analyse d'erreurs, coût et limites : voir **[RAPPORT_EVALUATION.md](RAPPORT_EVALUATION.md)**.
> Prototype non commercial, données **synthétiques**. L'analyse est produite par des agents LLM et peut être inexacte.

---

## Table des matières

1. [Contexte et objectif](#1-contexte-et-objectif)
2. [Vue d'ensemble de la solution](#2-vue-densemble-de-la-solution)
3. [État d'avancement](#3-état-davancement)
4. [Structure du dépôt et des notebooks](#4-structure-du-dépôt-et-des-notebooks)
5. [Les données](#5-les-données)
6. [Ce qui a été réalisé, étape par étape](#6-ce-qui-a-été-réalisé-étape-par-étape)
7. [Résultats](#7-résultats)
8. [L'application web (Databricks Apps)](#8-lapplication-web-databricks-apps)
9. [Comment reproduire](#9-comment-reproduire)
10. [Démo de 5 minutes](#10-démo-de-5-minutes)
11. [Difficultés rencontrées et leçons](#11-difficultés-rencontrées-et-leçons)
12. [Limites](#12-limites)
13. [Ce qui reste à faire](#13-ce-qui-reste-à-faire)
14. [Règles de travail](#14-règles-de-travail)

---

## 1. Contexte et objectif

Une entreprise produit deux types de prévisions de ventes :

- **Batch** : calcul périodique à partir d'un *snapshot* de données figé.
- **Real-Time** : calcul à la demande, avec des données potentiellement plus récentes.

Quand les deux résultats diffèrent (exemple : 1 000 contre 1 350 unités), l'enquête manuelle est lente et dépend de l'expertise de quelques personnes : il faut comparer les snapshots, les variables d'entrée, les configurations et les logs, et connaître les règles de l'entreprise.

**Objectif** : construire un système qui mène cette enquête automatiquement avec des outils en lecture seule, identifie la **cause racine**, fournit des **preuves**, un **niveau de confiance** et une **recommandation**, puis présente le tout dans une application lisible par un humain.

### Valeur apportée (et ce qui n'est pas démontré)

| Apport | Comment le projet le fournit |
|---|---|
| Gain de temps | Environ 8 à 16 s par cas, 6 à 11 appels LLM |
| Explications vérifiables | Chaque affirmation cite une source `tool:...` ou `doc:...`, contrôlée par le code |
| Connaissance de l'entreprise | Le RAG retrouve les règles (24 h, 5 %), les incidents passés et le runbook |
| Cause contre symptôme | CASE_009 : la différence de `sales_7d` est un symptôme de données obsolètes |
| Pas de fausse alerte | CASE_007 : l'agent conclut `NO_ISSUE` malgré 6 h de décalage |
| Sécurité par conception | Outils en lecture seule, liste blanche de tables, vérité terrain inaccessible |

**Non démontré** : une meilleure exactitude que les règles simples (elles font 12/12 parce qu'elles ont été écrites en connaissant les cas), un avantage des 3 agents sur l'agent unique, des performances sur données réelles.

### Principe de conception

> **Le code calcule, le LLM raisonne.** Les écarts, âges de snapshots et différences sont calculés de façon déterministe (PySpark / SQL). Le LLM reçoit des résultats chiffrés, pas des milliers de lignes, et rédige l'interprétation.

---

## 2. Vue d'ensemble de la solution

```
Question : "Pourquoi Batch et Real-Time diffèrent pour CASE_xxx ?"
                         |
          +--------------+--------------+      (exécutés en parallèle, LangGraph)
          |                             |
      AGENT 1                       AGENT 2
   Investigateur               Connaissance (RAG)
   5 outils de données         search_knowledge :
   en lecture seule            BM25 + embeddings + RRF
   -> faits chiffrés           -> règles, définitions,
                                  incidents, runbook
          |                             |
          +--------------+--------------+
                         |
                      AGENT 3
              Analyste et Vérificateur
        croise, écarte l'hypothèse non prouvée,
        choisit la cause racine, cite ses sources
                         |
          Vérification par le CODE (sources, nombres)
          une seule révision si problème détecté
                         |
   Rapport : cause principale, causes secondaires, preuves
   (affirmation + source), confiance, recommandation, vérification
                         |
              Table agent_runs_v2 (Delta)
                         |
        Application Streamlit (Databricks Apps) : relit et met en forme
```

### Les 3 agents

| Agent | Rôle | Outils | Sortie |
|---|---|---|---|
| **1. Investigateur** | Collecte les faits. **Ne conclut pas** et n'applique aucun seuil. | `get_forecasts`, `compare_features`, `compare_configs`, `check_freshness`, `get_pipeline_logs` | Liste de faits avec source `tool:<nom>` |
| **2. Connaissance** | Retrouve les règles métier utiles. **N'a pas accès aux données du cas.** | `search_knowledge` | Règles et incidents avec source `doc:<doc_id>` |
| **3. Analyste et Vérificateur** | Applique les règles aux faits, choisit la cause racine, rédige le rapport. Pas d'outil. | aucun | Rapport JSON complet |

### Les causes possibles

| Cause | Signification |
|---|---|
| `DATA_MISMATCH` | Les deux pipelines ont lu des données historiques différentes |
| `FEATURE_MISMATCH` | Une feature d'entrée diffère (promotion, prix...) |
| `CONFIG_MISMATCH` | Une configuration diffère (horizon, prétraitement, feature set) |
| `MODEL_VERSION_MISMATCH` | Les deux pipelines utilisent des versions de modèle différentes |
| `STALE_DATA` | Un snapshot de plus de 24 h est obsolète ; les différences de features qui en découlent sont des **symptômes** |
| `NO_ISSUE` | Écart de 5 % ou moins et aucune différence d'entrée : écart normal |

### Format du rapport final

```json
{
  "primary_cause": "STALE_DATA",
  "secondary_causes": [],
  "evidence": [
    {"claim": "Le snapshot Batch a 50 h, au-dessus du seuil de 24 h", "source": "tool:check_freshness"},
    {"claim": "Un snapshot de plus de 24 h est obsolète", "source": "doc:02_freshness_rules"}
  ],
  "confidence": "high",
  "recommendation": "Rafraîchir la table amont puis relancer le Batch ... (runbook doc:09_...)",
  "verification": {"passed": true, "problems": [], "revised": false, "sources_allowed": ["..."]}
}
```

---

## 3. État d'avancement

| # | Étape | Notebook(s) / fichier(s) | État |
|---|---|---|---|
| 1 | Environnement Databricks, schéma `forecast_debug`, volume `raw` | `00_setup` | Fait |
| 2 | Dataset synthétique : 7 tables Delta | `00_load_data` | Fait |
| 3 | Moteur de comparaison PySpark : 4 tables `cmp_*` | `01_comparison_engine_pyspark` | Fait |
| 4 | Exploration et fiches d'enquête | `02_data_exploration`, `02b_fiches_enquete` | Fait |
| 5 | Baseline par règles, sans LLM | `03_rule_based_baseline` | Fait : 12/12 |
| 6 | 5 outils en lecture seule + tests | `04_agent_tools`, `04b_test_tools` | Fait : tests OK |
| 7 | RAG TF-IDF | `05_rag`, `05b_test_rag` | Fait |
| 8 | RAG hybride BM25 + embeddings + RRF | `05d_rag_hybrid`, `05e_test_rag_hybrid` | Fait |
| 9 | Connexion au LLM et tool calling | `06_llm_connection_test` | Fait |
| 10 | Agent unique | `07_agent_single` | Fait : 88,9 % (3 passes) |
| 11 | Système à 3 agents LangGraph | `08_multi_agent` | Fait : 91,7 % (1 passe) |
| 12 | Rapport d'évaluation et analyse d'erreurs | `RAPPORT_EVALUATION.md` | Fait (à mettre à jour avec le dataset v2) |
| 13 | Application web Streamlit sur Databricks Apps | `app/app.py`, `app/app.yaml`, `app/requirements.txt` | Fait : déployée, lit les résultats réels |
| 14 | Thème visuel détaillé (comparaison Batch / Real-Time, jauges, déroulé) | `app/theme.py` | Écrit, **à valider dans l'application déployée** |
| 15 | Tableau de bord d'évaluation (Databricks Dashboards) | requêtes SQL, section 8.7 | Proposé, à créer |
| 16 | Dataset v2 plus difficile, découpage développement/test | à créer | **À faire** |
| 17 | Démo de 5 minutes | section 10 | Script écrit, **à répéter** |

---

## 4. Structure du dépôt et des notebooks

```
Forecast-Debugging-Agent/
├── README.md
├── RAPPORT_EVALUATION.md
├── data/                          6 fichiers CSV du dataset
├── rag_docs/                      10 documents Markdown du RAG
├── app/                           application web (Databricks Apps)
│   ├── app.py                     interface Streamlit
│   ├── theme.py                   CSS et composants visuels
│   ├── app.yaml                   commande de démarrage et variables
│   ├── requirements.txt           dépendances de l'application
│   └── .streamlit/config.toml     thème de base (optionnel)
└── notebooks/
    ├── 00_setup                         schéma et volume
    ├── 00_load_data                     CSV -> tables Delta (+ rag_documents)
    ├── 01_comparison_engine_pyspark     tables cmp_*
    ├── 02_data_exploration              requêtes SQL d'exploration
    ├── 02b_fiches_enquete               fiche d'enquête par cas (5 questions)
    ├── 03_rule_based_baseline           investigate_case() par règles + score
    ├── 04_agent_tools                   5 outils + schémas + dispatcher sécurisé
    ├── 04b_test_tools                   tests des outils
    ├── 05_rag, 05b_test_rag             RAG TF-IDF (première version)
    ├── 05d_rag_hybrid                   module RAG hybride (search_knowledge)
    ├── 05e_test_rag_hybrid              test et mesure du RAG hybride
    ├── 06_llm_connection_test           choix et test du LLM, tool calling
    ├── 07_agent_single                  agent unique + évaluation
    └── 08_multi_agent                   3 agents LangGraph + vérification + évaluation
```

Non exécuté : `05c_vector_search` (Databricks Vector Search) a été proposé mais n'a pas été lancé. `01_comparison_engine.scala` existe comme bonus pour un environnement qui supporte Scala, mais Scala n'est pas supporté sur la Free Edition, donc le chemin principal est PySpark.

**Tous les notebooks doivent être dans le même dossier du workspace** : ils s'appellent entre eux avec `%run ./nom`. Les fichiers de l'application (`app/`) sont des **fichiers**, pas des notebooks.

---

## 5. Les données

Les données sont **synthétiques** : on ne sait pas si les données réelles de l'entreprise seront accessibles, et des cas à cause connue permettent de mesurer le système.

Catalogue : `workspace`, schéma : `forecast_debug`, volume : `raw` (dossiers `data/` et `rag_docs/`).

### Tables sources (créées par `00_load_data`)

| Table | Lignes | Contenu |
|---|---|---|
| `forecast_predictions` | 24 | Prévision de chaque système pour chaque cas (12 cas x 2) |
| `feature_snapshots` | 96 | 4 features par système : `sales_7d`, `promotion`, `price`, `stock_level` |
| `model_configs` | 96 | 4 configurations : `forecast_horizon_days`, `model_version`, `feature_set_version`, `preprocessing` |
| `pipeline_runs` | 24 | Heure d'exécution, snapshot de données, statut, warnings, versions de code et de données |
| `actual_sales` | 12 | Ventes réelles (non utilisées par les outils) |
| `known_issues` | 12 | **Vérité terrain, réservée à l'évaluation.** Colonnes : `case_id`, `expected_root_cause`, `secondary_cause`, `difficulty`, `requires_rag`, `issue_description` |
| `rag_documents` | 10 | Documents texte pour le RAG |

### Tables de comparaison (créées par `01_comparison_engine_pyspark`)

| Table | Lignes | Contenu |
|---|---|---|
| `cmp_forecast_gap` | 12 | Écart en unités et en %, indicateur `within_tolerance` (seuil 5 %) |
| `cmp_inputs` | 96 | Différences de features et de configs entre Batch et Real-Time |
| `cmp_freshness` | 12 | Âge des snapshots en heures, statut de fraîcheur (seuil 24 h) |
| `cmp_case_summary` | 12 | Synthèse par cas |

### Tables de résultats (créées par les notebooks 07 et 08)

| Table | Contenu |
|---|---|
| `agent_runs` | Résultats de la **première version** de l'agent unique (10/12) |
| `agent_runs_v2` | Résultats des runs de l'agent unique et du système à 3 agents. C'est la table lue par l'application. |

Colonnes de `agent_runs_v2` utilisées par l'application : `batch_id`, `config` (système, RAG, version du prompt, modèle ; commence par `multi` pour les 3 agents), `run`, `case_id`, `predicted`, `secondary`, `confidence`, `tools`, `n_tools`, `n_llm`, `seconds`, `repaired`, `raw` (rapport JSON), `tool_outputs` (sorties des outils), `ts`.

**Troncature** : le rapport `raw` est conservé sur 4 000 caractères au plus, et `tool_outputs` sur 12 000. Les gros passages du RAG, placés après les 5 outils de données, peuvent faire dépasser la limite : le JSON est alors coupé (voir 8.6).

### Les 12 cas

| Cas | Batch | Real-Time | Cause attendue |
|---|---|---|---|
| 001 | 1000 | 1200 | `DATA_MISMATCH` (`sales_7d` : 100 vs 120) |
| 002 | 1000 | 1350 | `FEATURE_MISMATCH` (`promotion` : 0 vs 1) |
| 003 | 1100 | 1600 | `CONFIG_MISMATCH` (horizon 7 vs 14 jours) |
| 004 | 900 | 1200 | `STALE_DATA` (snapshot Batch du 10/11, 50 h) |
| 005 | 1050 | 1070 | `NO_ISSUE` (écart 1,9 %) |
| 006 | 800 | 1150 | `FEATURE_MISMATCH` + `CONFIG_MISMATCH` secondaire (promotion et feature set FS1 vs FS2) |
| 007 | 1500 | 1555 | `NO_ISSUE`, piège : 6 h de décalage de snapshot |
| 008 | 1200 | 1500 | `MODEL_VERSION_MISMATCH` (V2 vs V3) |
| 009 | 950 | 1300 | `STALE_DATA`, `sales_7d` différent n'est qu'un **symptôme** |
| 010 | 700 | 710 | `NO_ISSUE` |
| 011 | 1300 | 1000 | `FEATURE_MISMATCH` (`price` : 20 vs 24) |
| 012 | 600 | 820 | `CONFIG_MISMATCH` (`preprocessing`) |

Cas difficiles : **006** (deux causes), **007** (faux signal d'alerte), **009** (cause cachée derrière un symptôme).

### Les documents du RAG (10 documents)

`01_pipeline_overview`, `02_freshness_rules` (seuil 24 h), `03_forecast_tolerance` (seuil 5 %), `04_feature_definitions`, `05_model_versions_and_configs`, `06_incident_2026_03_promo_sync`, `07_incident_2026_06_stale_snapshot`, `08_incident_2026_08_model_rollout`, `09_runbook_investigation`, `10_glossary_root_causes`.

---

## 6. Ce qui a été réalisé, étape par étape

### 6.1 Environnement (`00_setup`)
Création du schéma `forecast_debug` dans le catalogue `workspace`, puis du volume `raw` avec les dossiers `data/` et `rag_docs/`.

### 6.2 Chargement des données (`00_load_data`)
Lit les CSV du volume et crée les 6 tables Delta plus `rag_documents`. Contrôle : 24, 96, 96, 24, 12, 12 lignes et 10 documents.

### 6.3 Moteur de comparaison (`01_comparison_engine_pyspark`)
Calcule en PySpark les 4 tables `cmp_*`. C'est la partie **déterministe** : l'écart, l'âge des snapshots et les différences d'entrée ne sont jamais calculés par le LLM.

### 6.4 Exploration (`02_data_exploration`, `02b_fiches_enquete`)
Requêtes SQL et fonction `fiche(case_id)` qui répond à 5 questions pour un cas : écart, fraîcheur, features qui diffèrent, configs qui diffèrent, warning du pipeline. Plusieurs cas ont été résolus à la main avant toute IA, pour comprendre ce que l'agent devra faire.

### 6.5 Baseline sans LLM (`03_rule_based_baseline`)
`investigate_case(case_id)` applique des règles fixes :
- snapshot de plus de 24 h : `STALE_DATA`, les différences de `sales_7d` deviennent des symptômes ;
- écart de 5 % ou moins sans différence d'entrée : `NO_ISSUE` ;
- sinon la cause est déduite de la nature de la différence ; une priorité fixe choisit la cause principale.

**Résultat : 12/12.** Ce score est à lire avec prudence : les règles ont été écrites en connaissant les 12 cas. Il valide la chaîne de données et d'évaluation, mais il ne prouve pas que le système tiendrait sur des cas ambigus.

### 6.6 Les 5 outils (`04_agent_tools`, `04b_test_tools`)

| Outil | Retourne |
|---|---|
| `get_forecasts(case_id)` | Prévisions Batch et Real-Time, écart en unités et en % |
| `compare_features(case_id)` | Les 4 features et celles qui diffèrent |
| `compare_configs(case_id)` | Les 4 configs et celles qui diffèrent |
| `check_freshness(case_id)` | Âge des snapshots en heures |
| `get_pipeline_logs(case_id)` | Statut, warnings, versions de code et de données, durée |

Le notebook expose aussi `TOOL_SCHEMAS` (descriptions au format *tool calling*) et `call_tool(name, args)`, le dispatcher qui exécute l'outil demandé par le LLM et renvoie toujours du JSON.

Choix de conception :
- **Lecture seule**, aucune requête SQL libre.
- **Liste blanche de 7 tables** : `known_issues`, `actual_sales` et `rag_documents` sont refusées par le code, et un test le vérifie.
- **Les outils renvoient des faits, pas de verdict.** `check_freshness` donne « 50 h », pas « obsolète ». Les règles (24 h, 5 %) vivent dans les documents RAG. Le RAG a donc un rôle réel.
- Un `case_id` invalide (format `CASE_001` imposé par une expression régulière) renvoie une erreur propre, sans requête.

### 6.7 Le RAG

**Corpus** : 10 documents synthétiques, découpés par titre de section.

**Version 1 (`05_rag`, `05b_test_rag`)** : recherche TF-IDF avec scikit-learn.

**Version 2 (`05d_rag_hybrid`)** : recherche hybride, entièrement gratuite.

| Composant | Rôle |
|---|---|
| BM25 (`rank-bm25`) | Recherche par mots-clés, accents retirés |
| Embeddings (`sentence-transformers`, modèle multilingue `paraphrase-multilingual-MiniLM-L12-v2`) | Recherche par sens, adaptée aux documents en français |
| Reciprocal Rank Fusion (RRF, k = 60) | Fusion des deux classements, sans poids à régler |

Le module définit `search_knowledge(query, k)` et son schéma d'outil `SEARCH_KNOWLEDGE_SCHEMA`. Le modèle d'embeddings se télécharge correctement depuis HuggingFace sur la Free Edition.

### 6.8 Connexion au LLM (`06_llm_connection_test`)
Le notebook liste les endpoints disponibles, teste un appel simple, puis un appel avec outils. Le workspace expose plusieurs modèles de chat ; **`databricks-gpt-oss-120b`** a appelé l'outil correctement et a été retenu pour tout le projet, avec `temperature = 0`.

Deux observations utiles :
- C'est un **modèle à raisonnement** : avec une limite de jetons trop basse, il dépense ses jetons à réfléchir et renvoie une réponse vide. On utilise `max_tokens = 4000`.
- Le modèle écrit parfois « 1 200 » avec une espace insécable : les comparaisons de nombres doivent la retirer.

### 6.9 Agent unique (`07_agent_single`)
Un LLM, les 5 outils de données et `search_knowledge`, avec une sortie JSON imposée.

Fonctions du notebook :
- **Relance de format** : si la réponse finale est vide ou illisible, une seule relance sans outils demande le JSON. Elle est **signalée** (`repaired`), la réponse brute d'origine est conservée.
- Un cas qui plante n'arrête plus la boucle.
- `N_RUNS` : plusieurs passes complètes, avec la stabilité par cas.
- `USE_RAG` : active ou désactive l'outil de recherche.
- Métriques : exactitude de la cause principale, précision et rappel sur l'ensemble des causes, nombre d'appels, temps, **ancrage des preuves** (les nombres cités se retrouvent-ils dans les sorties d'outils ?).
- Sauvegarde dans `agent_runs_v2`.

Version du prompt : `v1`. Elle n'a **pas** été retouchée après avoir vu les résultats, pour ne pas la régler sur les cas qu'elle rate.

### 6.10 Système à 3 agents (`08_multi_agent`)
Orchestré par **LangGraph** : les agents 1 et 2 partent en parallèle, l'agent 3 attend les deux (`PARALLEL = False` permet une exécution séquentielle). Même modèle, mêmes outils et mêmes documents que l'agent unique.

**Vérification par le code** (et non par le LLM), après l'agent 3 :
- chaque source citée doit exister (`tool:<nom>` d'un outil réellement appelé, ou `doc:<id>` d'un document réellement retrouvé) ;
- chaque nombre cité doit figurer dans les entrées (sorties d'outils ou passages du RAG) ;
- une recommandation doit être présente.

En cas de problème, l'agent 3 reçoit la liste des problèmes et a **une seule révision**. Le résultat (`passed`, `revised`, `problems`) est conservé dans le rapport.

Version du prompt : `ma_v1`.

---

## 7. Résultats

Tous les résultats concernent **12 cas synthétiques**. Ils montrent que le système fonctionne sur des scénarios contrôlés, pas qu'il fonctionnerait en production.

### 7.1 RAG

Mesure du *Recall@3* : le bon document figure-t-il dans les 3 premiers résultats ?

| Test | BM25 | Embeddings | Hybride |
|---|---|---|---|
| 10 requêtes (Imen) | 7/10 | 9/10 | 10/10 |
| 7 requêtes (Mariam) | 6/7 | 7/7 | 7/7 |
| 14 requêtes, avec MRR et top-1 | 10/14 (MRR 0,657, top-1 7/14) | 12/14 (MRR 0,889, top-1 12/14) | 12/14 (MRR 0,730, top-1 8/14) |

Les écarts entre les deux premières lignes viennent de listes de requêtes de tailles différentes, pas du code.

Sur 14 requêtes, **Dense et Hybride ont le même recall@3 mais échouent sur des requêtes différentes** : la fusion ne fait pas mieux que les embeddings seuls et dégrade le top-1. Les 4 échecs relevés concernent des requêtes vagues (« une autre réglage ») ou des étiquettes ambiguës (un document voisin répondait aussi). L'hybride reste le choix par défaut parce qu'il est plus robuste sur les termes exacts (`sales_7d`, `DATA_MISMATCH`), mais ce choix n'est pas démontré par ces chiffres.

### 7.2 Comparaison des trois approches sur les 12 cas

| | Règles (03) | Agent unique + RAG (07) | 3 agents + RAG (08) |
|---|---|---|---|
| Cause principale | 12/12 | 10/12, 11/12, 11/12 (moyenne **88,9 %**, min 83,3 %, max 91,7 %) | 11/12 (**91,7 %**, une seule passe) |
| Ensemble de causes (précision / rappel) | non mesuré | 0,90 / 0,90 | 0,87 / 1,00 |
| Appels LLM par cas | 0 | 6,4 | 11,2 |
| Temps moyen par cas | négligeable | 8,1 s | 16,5 s |
| Outils appelés par cas | n/a | 5,4 | 5 (agent 1) + recherches (agent 2) |
| Réponses relancées pour le format | n/a | 1/36 | 0/12 |
| Réponses illisibles | n/a | 0/36 | 0/12 |
| Vérification par le code | n/a | ancrage des nombres : 311/312 | 12/12 réussies, 1/12 révisé (CASE_003) |

**À ne pas conclure** : que les 3 agents sont meilleurs que l'agent unique. 91,7 % sur une passe et 88,9 % en moyenne sur 3 passes ne se distinguent pas avec 12 cas (une erreur vaut 8 points). Ce qui est mesuré, c'est que le système à 3 agents coûte environ **deux fois plus** en appels et en temps, et qu'il produit un rapport plus complet (recommandation, sources, vérification).

### 7.3 Résultat par cas

| Cas | Agent unique (3 passes) | 3 agents (1 passe) |
|---|---|---|
| 001 | **Instable** : `FEATURE_MISMATCH` (passe 1), `DATA_MISMATCH` (passes 2 et 3) | Correct, avec `FEATURE_MISMATCH` en cause secondaire en trop |
| 002, 003, 004, 005, 007, 008, 010, 011, 012 | Corrects | Corrects (CASE_003 révisé par la vérification) |
| 006 | **Faux 3 fois sur 3** : `CONFIG_MISMATCH` principal, sans cause secondaire | `CONFIG_MISMATCH` principal, `FEATURE_MISMATCH` trouvé en secondaire |
| 009 | Cause principale correcte, cause secondaire en trop selon les passes | Cause principale correcte, `FEATURE_MISMATCH` en secondaire en trop |

### 7.4 Analyse d'erreurs

**CASE_006 (deux causes).** Le prompt n'indique pas quelle cause doit être principale quand il y en a deux. Dans la baseline, une priorité fixe était codée en connaissant la réponse. C'est donc une **ambiguïté de la consigne** autant qu'une erreur du modèle. Avec l'agent unique, la cause `FEATURE_MISMATCH` était même parfois absente ; avec les 3 agents, elle est retrouvée en secondaire. Nous n'avons pas modifié le prompt pour faire passer ce cas.

**CASE_001 (instabilité).** `sales_7d` est à la fois une donnée d'historique et une feature, donc `DATA_MISMATCH` et `FEATURE_MISMATCH` se recouvrent. Le fait mesuré est l'instabilité entre passes à `temperature = 0` ; l'explication par le recouvrement des deux notions est une interprétation non vérifiée.

**CASE_009 (symptôme et cause).** La cause principale est toujours juste, mais l'agent ajoute parfois une cause secondaire alors que la règle de fraîcheur dit que la différence de `sales_7d` est un symptôme. Cela fait baisser la précision.

**Réponses vides.** Sur CASE_009 (passe 1), la première réponse de l'agent unique était vide ; la relance a produit un JSON correct. C'est très probablement l'origine de l'échec du premier run (CASE_004 illisible), mais ce run n'avait pas conservé la réponse brute, donc ce n'est pas prouvé.

**Requêtes du RAG.** L'agent invente parfois des termes de recherche (par exemple « seuil 30 % », qui n'existe pas dans les documents). Le RAG renvoie malgré tout les bons passages sur ces cas.

### 7.5 Sécurité et traçabilité vérifiées

- Les agents n'ont jamais accès à `known_issues` : l'outil de lecture refuse la table, et seule la cellule d'évaluation (et l'onglet « Évaluation et limites » de l'application) la lit.
- Chaque rapport cite ses sources, et le code vérifie qu'elles existent.
- Chaque run est sauvegardé dans `agent_runs_v2`, avec les sorties d'outils.

---

## 8. L'application web (Databricks Apps)

L'application est un **tableau de restitution en lecture seule** : elle ne relance pas les agents. Les agents ont déjà tourné dans Databricks, et l'application relit leur analyse enregistrée dans `agent_runs_v2`, puis la met en forme. « Temps réel » signifie ici : la page relit la table (cache de 30 s) ; un cas prend 8 à 16 s à analyser, ce n'est pas du streaming.

### 8.1 Ce que l'utilisateur voit

| Onglet | Contenu |
|---|---|
| **Analyse** | Choix d'un cas ou de tout le lot, bouton « Lancer l'analyse », puis le rapport |
| **Architecture** | Schéma des 3 agents, du RAG, de la vérification par le code et de l'unique révision |
| **Évaluation et limites** | Exactitude et coût mesurés (comparaison à la vérité terrain), limites écrites sans détour |

La barre latérale permet de choisir l'exécution (`batch_id`) et la passe, et d'actualiser les données.

**Rapport d'un cas** : verdict (cause, confiance, résultat de la vérification), 4 compteurs (écart en unités, écart relatif, âge des snapshots Batch et Real-Time), graphique des deux prévisions, graphique de fraîcheur avec la limite de 24 h, chaîne d'explication (cause, faits observés, effet), tableau de ce qui diffère entre les deux pipelines, avertissement du pipeline Batch s'il existe, recommandation, preuves avec leurs sources, détail de la vérification par le code, et **export PDF**.

**Rapport du lot** : compteurs (cas analysés, avec problème, confiance faible, vérification à revoir), répartition des causes, carte de décision (écart contre âge du snapshot, avec les seuils 5 % et 24 h), liste « à relire en priorité » (problèmes d'abord, puis confiance la plus faible), et PDF du lot.

### 8.2 Détails de conception

- **Source de vérité unique** : la table `agent_runs_v2`. La vérité terrain `known_issues` n'est utilisée que dans l'onglet d'évaluation, jamais à côté d'un cas.
- **Mode démonstration** : sans `DATABRICKS_WAREHOUSE_ID`, l'application démarre avec 4 cas **fictifs** (`DEMO_01` à `DEMO_04`). Ne jamais les présenter comme des résultats réels.
- **Sécurité** : tout texte écrit par le modèle est échappé avant affichage (`html.escape`), l'identifiant de lot est validé par expression régulière, et l'application n'exécute que des `SELECT`.
- **Lecture tolérante** : si `tool_outputs` est tronqué, l'application récupère les paires `[outil, sortie]` complètes du début de la liste. Si une sortie manque, le panneau concerné affiche « Données non disponibles » et une ligne liste les outils absents ; la cause, les preuves et la recommandation restent affichées. La chaîne d'explication n'affirme jamais « aucune différence d'entrée » quand elle n'a pas les faits.
- **PDF** : généré avec `reportlab`. Les traits d'union et espaces insécables spéciaux écrits par le modèle sont remplacés par des caractères simples, sinon la police standard affiche des « ? ».
- **Thème** : `theme.py` contient le CSS et les composants visuels (en-tête Batch / Real-Time, cartes de comparaison, jauges avec repère sur la limite documentée, étiquettes « Outil » et « Document » sur les sources, déroulé de l'enquête en 4 étapes). Les noms de classes CSS sont ceux de `app.py`, donc le style s'applique sans réécrire l'interface.

### 8.3 Fichiers

| Fichier | Rôle |
|---|---|
| `app.py` | Interface, accès aux données, graphiques Altair, PDF |
| `theme.py` | CSS et composants visuels. Doit être dans le même dossier que `app.py`. |
| `app.yaml` | Commande de démarrage Streamlit ; lit l'ID du SQL warehouse depuis la ressource de clé `sql-warehouse` |
| `requirements.txt` | Dépendances. Aucune version n'est figée : à épingler après un déploiement qui fonctionne. |
| `.streamlit/config.toml` | Thème de base (optionnel ; si l'upload masque les dossiers cachés, l'ignorer) |

### 8.4 Déploiement pas à pas

1. **Vérifier les données.** Dans le SQL Editor :
   ```sql
   SELECT config, COUNT(*) AS n_lignes FROM workspace.forecast_debug.agent_runs_v2 GROUP BY config;
   ```
   Il faut au moins une ligne dont `config` commence par `multi` (sinon lancer `08_multi_agent` jusqu'au bout).
2. **Préparer le dossier.** Dans Workspace, créer un dossier (ex. `forecast-debugging-app`) et y importer `app.py`, `theme.py`, `app.yaml`, `requirements.txt`. La colonne *Type* doit indiquer **File** pour chacun ; un `app.py` importé comme notebook ne peut pas être le point d'entrée.
3. **Créer l'application.** Sélecteur d'applications (icône à neuf points), puis *Databricks Apps*, *Create app*, *Create a custom app*. Nom en minuscules, chiffres et tirets (il ne peut plus être changé). Laisser la politique d'usage serverless sur *None* et sauter l'étape du dépôt Git.
4. **Ajouter la ressource SQL warehouse.** Dans Settings, *+ Add resource*, *SQL warehouse*, choisir le warehouse existant, permission **Can use**, clé exactement **`sql-warehouse`** (c'est la clé qu'`app.yaml` référence).
5. **Déployer.** Quand le calcul est prêt (le bouton *Deploy* devient bleu), *Deploy*, choisir le dossier de l'étape 2. Le premier déploiement installe les bibliothèques et prend quelques minutes.
6. **Donner les droits de lecture au service principal.** Il est affiché dans l'onglet *Authorization* de l'application. Dans le SQL Editor, avec son nom ou son identifiant d'application entre accents graves :
   ```sql
   GRANT USE CATALOG ON CATALOG workspace TO `<service_principal>`;
   GRANT USE SCHEMA ON SCHEMA workspace.forecast_debug TO `<service_principal>`;
   GRANT SELECT ON TABLE workspace.forecast_debug.agent_runs_v2 TO `<service_principal>`;
   GRANT SELECT ON TABLE workspace.forecast_debug.known_issues TO `<service_principal>`;
   ```
   Vérifier avec `SHOW GRANTS ON TABLE workspace.forecast_debug.agent_runs_v2;` et `SHOW GRANTS ON CATALOG workspace;`. Dans ce projet, Unity Catalog a accepté l'**identifiant d'application** (long identifiant) et non le nom d'affichage.
7. **Ouvrir l'URL** de l'application, puis Ctrl + F5 après chaque changement de permissions ou de code.

Pour mettre à jour : remplacer les fichiers dans le dossier du workspace, puis *Deploy* sur la même application.

### 8.5 Dépannage

| Symptôme | Cause probable | Correction |
|---|---|---|
| « Aucune exécution des 3 agents » | Aucune ligne `multi...` dans `agent_runs_v2` | Lancer `08_multi_agent` jusqu'à la fin |
| `INSUFFICIENT_PERMISSIONS` ou table introuvable | Droits manquants pour le service principal | Refaire les `GRANT` (y compris `USE CATALOG`) |
| « Évaluation indisponible » | Pas de `SELECT` sur `known_issues` | Accorder ce droit |
| Application en erreur au démarrage | Fichier manquant, `app.py` importé comme notebook, ressource `sql-warehouse` absente | Vérifier les types de fichiers et la ressource, puis redéployer |
| `ModuleNotFoundError` | `requirements.txt` absent ou mal nommé | Corriger le nom, redéployer |
| Page blanche | Erreur Python | Lire l'onglet *Logs* |
| Première page lente | Le warehouse est arrêté (arrêt automatique après 10 min) | L'ouvrir une fois avant la démo |

### 8.6 Limites de l'application

- **Troncature des sorties d'outils (12 000 caractères)** : pour certains cas, l'écart, les âges de snapshot et le tableau des différences s'affichent « n/d ». Ce n'est pas une erreur d'affichage : les données manquent dans la table. Correction possible : relancer `08_multi_agent` avec une limite de sauvegarde plus grande. Le rapport `raw` (4 000 caractères) peut aussi être tronqué ; l'application affiche alors seulement la cause et la confiance.
- **Langue** : les preuves et recommandations sont en anglais pour certains cas, car l'agent a répondu en anglais malgré la consigne. Corriger demande de modifier le prompt (nouvelle version, par exemple `ma_v2`) et de relancer le notebook 08.
- **Une application par workspace** : elle doit tourner dans le workspace qui contient `agent_runs_v2`. Chaque compte Free Edition a son propre workspace et son propre metastore : les tables d'une camarade ne sont pas visibles.
- **Free Edition** : jusqu'à 3 applications par compte, arrêt automatique 24 h après un démarrage, une mise à jour ou un redéploiement. Redémarrer l'application juste avant une présentation. Usage non commercial uniquement.
- **Lecture seule** : pas de bouton « relancer l'agent », c'est la fonctionnalité la plus fragile.
- **Fichiers dans le Git folder** : ne pas les valider depuis Databricks (voir section 11) ; les télécharger et les téléverser sur github.com.
- **Ne jamais** copier le client ID, le secret du service principal ou un jeton dans le code, un notebook, une capture d'écran ou GitHub.

### 8.7 Tableau de bord d'évaluation (optionnel)

Databricks Dashboards (disponible sur la Free Edition) peut compléter l'application avec trois jeux de données SQL, dont les chiffres doivent correspondre au rapport d'évaluation (88,9 % pour l'agent unique, 91,7 % pour les 3 agents) :

```sql
-- 1. Exactitude par exécution
SELECT a.config, a.batch_id, a.run, COUNT(*) AS n_cases,
       SUM(CASE WHEN a.predicted = k.expected_root_cause THEN 1 ELSE 0 END) AS n_correct,
       ROUND(100.0 * SUM(CASE WHEN a.predicted = k.expected_root_cause THEN 1 ELSE 0 END) / COUNT(*), 1) AS accuracy_pct
FROM workspace.forecast_debug.agent_runs_v2 a
JOIN workspace.forecast_debug.known_issues k ON a.case_id = k.case_id
GROUP BY a.config, a.batch_id, a.run ORDER BY a.batch_id, a.run;

-- 2. Résultat par cas (erreurs et instabilité)
SELECT a.case_id, k.expected_root_cause AS attendu, a.config, a.run, a.predicted AS predit, a.confidence,
       a.predicted = k.expected_root_cause AS correct
FROM workspace.forecast_debug.agent_runs_v2 a
JOIN workspace.forecast_debug.known_issues k ON a.case_id = k.case_id
ORDER BY a.case_id, a.config, a.run;

-- 3. Coût par configuration
SELECT config, ROUND(AVG(n_llm), 1) AS appels_llm_moyens, ROUND(AVG(seconds), 1) AS secondes_moyennes,
       SUM(CASE WHEN repaired THEN 1 ELSE 0 END) AS relances_de_format
FROM workspace.forecast_debug.agent_runs_v2 GROUP BY config;
```

Ajouter un texte en haut : « Données synthétiques, 12 cas. Résultats non généralisables. »

### 8.8 Traçage MLflow (optionnel, non confirmé)

Le traçage automatique (`mlflow.openai.autolog()`) pourrait rendre chaque appel d'outil et de LLM visible pas à pas. Son fonctionnement sur la Free Edition avec `get_open_ai_client()` n'a pas été confirmé : à tester dans un notebook séparé (l'installation redémarre Python et casserait les `%run` des notebooks 07 et 08). Si le test échoue, ne pas insister.

---

## 9. Comment reproduire

1. Créer un compte **Databricks Free Edition**.
2. Créer un notebook `00_setup` (Python, Serverless) et exécuter la création du schéma `forecast_debug`.
3. Dans **Catalog**, créer le volume `raw` avec les dossiers `data/` et `rag_docs/`, puis téléverser les 6 CSV et les 10 fichiers `.md`.
4. Importer **tous les notebooks dans un même dossier** du workspace.
5. Exécuter dans l'ordre :

| Ordre | Notebook | Contrôle attendu |
|---|---|---|
| 1 | `00_load_data` | 7 lignes « Créée », 12 cas affichés |
| 2 | `01_comparison_engine_pyspark` | 4 tables `cmp_*` créées |
| 3 | `03_rule_based_baseline` | `Accuracy baseline règles: 12/12` |
| 4 | `04b_test_tools` | `TOUS LES TESTS OK` |
| 5 | `06_llm_connection_test` | `LLM CONNECTION OK` (adapter le modèle) |
| 6 | `07_agent_single` | Test CASE_009, puis les 12 cas, puis l'évaluation |
| 7 | `08_multi_agent` | Rapport complet CASE_009, puis les 12 cas |
| 8 | Application (section 8.4) | Les 12 cas apparaissent dans « Cas à analyser » |

6. Dans `07_agent_single` et `08_multi_agent`, lancer les cellules **une par une** la première fois. Pour la mesure : `N_RUNS = 3`.
7. Vérifier à la fin de chaque notebook la ligne « Sauvegardé dans workspace.forecast_debug.agent_runs_v2 ». Si elle indique « Sauvegarde impossible », la table n'existe pas et l'application n'aura rien à afficher.

### Règles d'import à respecter dans les notebooks 07 et 08

- Charger `05d_rag_hybrid` **avant** `04_agent_tools`.
- Chaque `%run` doit être la **toute première ligne** de sa cellule, sans commentaire avant, sinon Databricks ne le prend pas pour le `%run` du notebook.
- Ne mettre aucun `%pip` dans les notebooks appelés : un `%pip` efface les variables déjà définies. L'installation et `dbutils.library.restartPython()` se font dans le notebook appelant.
- Garder un seul notebook par nom (renommer les anciens brouillons), sinon `%run` charge le mauvais.

---

## 10. Démo de 5 minutes

| Temps | Écran | À dire et à faire |
|---|---|---|
| 0:00 à 0:30 | En-tête | « Notre entreprise calcule les prévisions de deux façons. Quand elles diffèrent, quelqu'un enquête à la main. Nos agents le font en quelques secondes et expliquent pourquoi. » |
| 0:30 à 1:00 | Analyse, tout le lot | Compteurs et liste « à relire en priorité » : problèmes d'abord, puis confiance la plus faible |
| 1:00 à 2:30 | Analyse, CASE_009 | Snapshot de 50 h contre la limite de 24 h ; la différence de `sales_7d` est un **symptôme**, la cause racine est `STALE_DATA` ; ouvrir les preuves, la recommandation et la vérification par le code |
| 2:30 à 3:15 | Analyse, CASE_007 | « Aucun problème » : l'agent n'invente pas de cause malgré 6 h de décalage |
| 3:15 à 4:00 | Architecture | Deux agents en parallèle, un analyste, une vérification par le code ; outils en lecture seule, vérité terrain cachée |
| 4:00 à 4:40 | Évaluation et limites | Exactitude et coût mesurés ; dire que les 3 agents ne sont pas démontrés meilleurs que l'agent unique et coûtent deux fois plus |
| 4:40 à 5:00 | Conclusion | « La valeur est la traçabilité et la connaissance réutilisable. Prochaine étape : un dataset plus difficile pour se comparer aux règles fixes. » |

Avant la démo : redémarrer l'application, ouvrir la page une fois pour réveiller le warehouse, préparer CASE_009 et CASE_007, garder des captures d'écran et une courte vidéo de secours. Ne pas relancer un agent en direct. Être prêtes à parler de CASE_006 (cause principale fausse faute de règle de priorité) : c'est une limite connue.

---

## 11. Difficultés rencontrées et leçons

| Problème | Ce qui a été observé | Solution retenue |
|---|---|---|
| Push Databricks vers GitHub | Erreurs `Input/output error` sur plusieurs fichiers, puis sur `.git/config`, après le déplacement de notebooks dans le Git folder | Cause exacte non établie. Contournement : export des notebooks en `.ipynb` et upload sur github.com. Créer les nouveaux notebooks directement dans le Git folder, ne jamais ouvrir le dossier `.git`. |
| Application GitHub non autorisée | « The link to your GitHub account does not have access » | Installer l'application Databricks sur le dépôt, ou passer par l'upload web |
| Filtre de colonne dans les outils | Erreur `CANNOT_RESOLVE_DATAFRAME_COLUMN` sur Spark Connect, non reproduite en local | Utiliser `F.col("case_id")` |
| Scala sur Free Edition | Non supporté | Version PySpark du moteur de comparaison |
| Réponses vides du LLM | Modèle à raisonnement dont les jetons sont consommés par la réflexion | `max_tokens` élevé et relance de format signalée |
| Variables disparues après `%run` | `%pip` dans un notebook appelé ; `%run` précédé d'un commentaire | Voir les règles d'import |
| Scores RAG différents entre les deux membres | Listes de requêtes de tailles différentes (7 et 10) | Une seule liste de référence |
| Table `agent_runs_v2` absente dans un workspace | Chaque compte a son propre workspace et metastore | Relancer 07 et 08 dans son propre workspace ; utiliser ses propres résultats |
| Droits de l'application | `INSUFFICIENT_PERMISSIONS` tant que le service principal n'a pas de `GRANT` | `GRANT` avec l'identifiant d'application ; vérifier aussi `USE CATALOG` |
| Panneaux vides dans l'application | Sorties d'outils tronquées à 12 000 caractères dans la table | Lecture des paires complètes du début ; à terme, une limite plus grande dans le notebook 08 |
| « ? » dans le PDF | Traits d'union et espaces insécables spéciaux du modèle | Remplacement par des caractères simples avant génération |
| Messages « Failed to connect to MetaMask » | Extension du navigateur, sans lien avec le projet | À ignorer |

Leçon générale : les tests sur Databricks ont révélé des bugs que les tests locaux ne voyaient pas (Spark Connect, droits du service principal, troncature des sorties). Il faut donc valider chaque notebook et l'application sur le workspace réel, avec les vraies données.

---

## 12. Limites

- **Données synthétiques** : les résultats valident le système sur des scénarios contrôlés, pas en production. Ne jamais écrire « 95 % en production », mais « X % sur N scénarios contrôlés ».
- **Dataset petit et propre** : 12 cas, une ou deux causes visibles par cas, valeurs rondes. Les règles font déjà 12/12, donc ce jeu ne permet pas de démontrer un avantage du LLM.
- **Peu de statistiques** : une erreur change le score de 8 points. Trois passes mesurent la stabilité, pas la capacité à généraliser (il n'y a que 12 situations différentes). Le système à 3 agents n'a été mesuré que sur une passe.
- **Pas de comparaison avec et sans RAG** : l'effet du RAG sur l'agent unique et sur les 3 agents n'a pas été mesuré (les options `USE_RAG = False` existent mais n'ont pas été exécutées).
- **Documents RAG et cas écrits ensemble**, avec les mêmes seuils (24 h, 5 %) : le RAG peut paraître plus utile que sur de vraies règles métier.
- **Étiquettes ambiguës** : pour CASE_006, rien dans les documents ni dans le prompt ne désigne la cause principale ; l'étiquette `FEATURE_MISMATCH` n'est pas plus « vraie » que `CONFIG_MISMATCH`.
- **Non-reproductibilité** : même à `temperature = 0`, le LLM ne donne pas toujours la même réponse (CASE_001).
- **Vérification limitée** : le contrôle du code vérifie les sources et les nombres, pas la logique du raisonnement. L'ancrage « 311/312 » de l'agent unique est un contrôle faible.
- **Évaluation du RAG** sur un petit nombre de requêtes écrites par l'équipe, sur 15 à 17 passages.
- **Un seul modèle LLM** (`databricks-gpt-oss-120b`) : les résultats ne sont pas comparés à d'autres modèles.
- **Free Edition** : accès internet sortant limité, quotas de compute, `get_open_ai_client()` signalé comme obsolète par le SDK (sans effet sur le fonctionnement).
- **Coût** : le système à 3 agents utilise environ 1,75 fois plus d'appels LLM et 2 fois plus de temps que l'agent unique, sans gain démontré sur ce dataset.
- **Application** : voir 8.6 (sorties tronquées, texte parfois en anglais, lecture seule, arrêt automatique après 24 h).

---

## 13. Ce qui reste à faire

1. **Valider `theme.py`** dans l'application déployée (comparaison Batch / Real-Time, jauges, déroulé) et corriger les éventuels écarts.
2. **Augmenter la limite de sauvegarde des sorties d'outils** dans `08_multi_agent` et relancer, pour que les panneaux de mesures soient remplis pour tous les cas.
3. **Dataset v2** : 60 à 100 cas avec bruit (petites différences sans importance), cas ambigus, cas hors règles, cas « preuves insuffisantes » (table vide, valeurs manquantes), et un découpage **développement / test** : le groupe de test n'est jamais utilisé pendant le réglage. C'est le seul moyen de comparer sérieusement règles, agent unique et 3 agents.
4. **Règle de priorité entre causes multiples** (CASE_006) : la définir dans les documents ou le prompt, en la justifiant, puis la tester sur de **nouveaux** cas.
5. **Mesure du RAG** : `USE_RAG = False` sur l'agent unique et sur les 3 agents, avec 3 passes chacun.
6. **3 passes des 3 agents** (`N_RUNS = 3`), puis mise à jour de **`RAPPORT_EVALUATION.md`**.
7. **Tableau de bord d'évaluation** (section 8.7) et capture d'écran pour le rapport.
8. **Répéter la démo** de 5 minutes (section 10).
9. **Sauvegarde sur GitHub** de tous les notebooks et des fichiers de l'application (upload web si le push Databricks échoue).

---

## 14. Règles de travail

- Valider les données et les outils avant de toucher au LLM.
- Ne jamais donner accès à `known_issues` aux agents.
- Ne jamais modifier les données ni les étiquettes pour que l'agent réussisse.
- Ne pas retoucher un prompt après avoir vu ses erreurs sans changer son nom de version et sans l'évaluer sur de nouveaux cas.
- Garder le même modèle et `temperature = 0` pour toutes les mesures.
- Garder toutes les passes dans `agent_runs_v2`, même les mauvaises.
- Ne jamais mettre de jeton, de secret ou d'identifiant de service principal dans le code ou sur GitHub.
- Présenter l'application comme un prototype non commercial sur données synthétiques.
- Pull avant de travailler, commit et push en fin de session.
