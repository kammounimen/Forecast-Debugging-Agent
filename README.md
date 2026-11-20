# Forecast Debugging Agent

**Système multi-agents basé sur les LLM et le RAG pour analyser et expliquer les écarts entre les prévisions Batch et Real-Time.**

Auteures : Imen Kammoun et Mariam Kbaier
Plateforme : Databricks Free Edition (compute serverless) | Langage : Python, SQL, PySpark
LLM : `databricks-gpt-oss-120b` (endpoint Databricks, aucune clé externe) | Orchestration : LangGraph

> Résultats détaillés, analyse d'erreurs, coût et limites : voir **[RAPPORT_EVALUATION.md](RAPPORT_EVALUATION.md)**.

---

## Table des matières

1. [Contexte et objectif](#1-contexte-et-objectif)
2. [Vue d'ensemble de la solution](#2-vue-densemble-de-la-solution)
3. [État d'avancement](#3-état-davancement)
4. [Structure du dépôt et des notebooks](#4-structure-du-dépôt-et-des-notebooks)
5. [Les données](#5-les-données)
6. [Ce qui a été réalisé, étape par étape](#6-ce-qui-a-été-réalisé-étape-par-étape)
7. [Résultats](#7-résultats)
8. [Comment reproduire](#8-comment-reproduire)
9. [Difficultés rencontrées et leçons](#9-difficultés-rencontrées-et-leçons)
10. [Limites](#10-limites)
11. [Ce qui reste à faire](#11-ce-qui-reste-à-faire)
12. [Règles de travail](#12-règles-de-travail)

---

## 1. Contexte et objectif

Une entreprise produit deux types de prévisions de ventes :

- **Batch** : calcul périodique à partir d'un *snapshot* de données figé.
- **Real-Time** : calcul à la demande, avec des données potentiellement plus récentes.

Quand les deux résultats diffèrent (exemple : 1 000 contre 1 350 unités), l'enquête manuelle est lente et dépend de l'expertise de quelques personnes.

**Objectif** : construire un système qui mène cette enquête automatiquement avec des outils en lecture seule, identifie la **cause racine**, fournit des **preuves**, un **niveau de confiance** et une **recommandation**.

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

| # | Étape | Notebook(s) | État |
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
| 12 | Dataset v2 plus difficile, découpage développement/test | à créer | **À faire** |
| 13 | Rapport d'évaluation et analyse d'erreurs | `RAPPORT_EVALUATION.md` | Fait (à mettre à jour avec le dataset v2) |
| 14 | Démo de 5 minutes | | **À faire** |

---

## 4. Structure du dépôt et des notebooks

```
Forecast-Debugging-Agent/
├── README.md
├── data/                          6 fichiers CSV du dataset
├── rag_docs/                      10 documents Markdown du RAG
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

**Tous les notebooks doivent être dans le même dossier du workspace** : ils s'appellent entre eux avec `%run ./nom`.

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
| `agent_runs_v2` | Résultats des runs de l'agent unique et du système à 3 agents : `batch_id`, `config`, `run`, `case_id`, `predicted`, `secondary`, `confidence`, outils appelés, nombre d'appels LLM, durée, relance, exactitude, réponse brute, sorties d'outils. La colonne `config` indique le système, la présence du RAG, la version du prompt et le modèle. |

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

- Les agents n'ont jamais accès à `known_issues` : l'outil de lecture refuse la table, et seule la cellule d'évaluation la lit.
- Chaque rapport cite ses sources, et le code vérifie qu'elles existent.
- Chaque run est sauvegardé dans `agent_runs_v2`, avec les sorties d'outils.

---

## 8. Comment reproduire

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

6. Dans `07_agent_single` et `08_multi_agent`, lancer les cellules **une par une** la première fois. Pour la mesure : `N_RUNS = 3`.

### Règles d'import à respecter dans les notebooks 07 et 08

- Charger `05d_rag_hybrid` **avant** `04_agent_tools`.
- Chaque `%run` doit être la **toute première ligne** de sa cellule, sans commentaire avant, sinon Databricks ne le prend pas pour le `%run` du notebook.
- Ne mettre aucun `%pip` dans les notebooks appelés : un `%pip` efface les variables déjà définies. L'installation et `dbutils.library.restartPython()` se font dans le notebook appelant.
- Garder un seul notebook par nom (renommer les anciens brouillons), sinon `%run` charge le mauvais.

---

## 9. Difficultés rencontrées et leçons

| Problème | Ce qui a été observé | Solution retenue |
|---|---|---|
| Push Databricks vers GitHub | Erreurs `Input/output error` sur plusieurs fichiers, puis sur `.git/config`, après le déplacement de notebooks dans le Git folder | Cause exacte non établie. Contournement : export des notebooks en `.ipynb` et upload sur github.com. Créer les nouveaux notebooks directement dans le Git folder, ne jamais ouvrir le dossier `.git`. |
| Application GitHub non autorisée | « The link to your GitHub account does not have access » | Installer l'application Databricks sur le dépôt, ou passer par l'upload web |
| Filtre de colonne dans les outils | Erreur `CANNOT_RESOLVE_DATAFRAME_COLUMN` sur Spark Connect, non reproduite en local | Utiliser `F.col("case_id")` |
| Scala sur Free Edition | Non supporté | Version PySpark du moteur de comparaison |
| Réponses vides du LLM | Modèle à raisonnement dont les jetons sont consommés par la réflexion | `max_tokens` élevé et relance de format signalée |
| Variables disparues après `%run` | `%pip` dans un notebook appelé ; `%run` précédé d'un commentaire | Voir les règles d'import ci-dessus |
| Scores RAG différents entre les deux membres | Listes de requêtes de tailles différentes (7 et 10) | Une seule liste de référence |
| Messages « Failed to connect to MetaMask » | Extension du navigateur, sans lien avec le projet | À ignorer |

Leçon générale : les tests sur Databricks ont révélé des bugs que les tests locaux ne voyaient pas (Spark Connect). Il faut donc valider chaque notebook sur le workspace réel.

---

## 10. Limites

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

---

## 11. Ce qui reste à faire

1. **Dataset v2** : 60 à 100 cas avec bruit (petites différences sans importance), cas ambigus, cas hors règles, cas « preuves insuffisantes » (table vide, valeurs manquantes), et un découpage **développement / test** : le groupe de test n'est jamais utilisé pendant le réglage. C'est le seul moyen de comparer sérieusement règles, agent unique et 3 agents.
2. **Règle de priorité entre causes multiples** (CASE_006) : la définir dans les documents ou le prompt, en la justifiant, puis la tester sur de **nouveaux** cas.
3. **Mesure du RAG** : `USE_RAG = False` sur l'agent unique et sur les 3 agents, avec 3 passes chacun.
4. **Mettre à jour `RAPPORT_EVALUATION.md`** avec les résultats du dataset v2 et des 3 passes des 3 agents.
5. **Démo de 5 minutes** : une question, un rapport structuré (CASE_009 convient), puis le tableau de résultats.
6. **Sauvegarde sur GitHub** de tous les notebooks (upload web si le push Databricks échoue).

---

## 12. Règles de travail

- Valider les données et les outils avant de toucher au LLM.
- Ne jamais donner accès à `known_issues` aux agents.
- Ne jamais modifier les données ni les étiquettes pour que l'agent réussisse.
- Ne pas retoucher un prompt après avoir vu ses erreurs sans changer son nom de version et sans l'évaluer sur de nouveaux cas.
- Garder le même modèle et `temperature = 0` pour toutes les mesures.
- Garder toutes les passes dans `agent_runs_v2`, même les mauvaises.
- Pull avant de travailler, commit et push en fin de session.
