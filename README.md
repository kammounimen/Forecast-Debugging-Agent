# Forecast Debugging Agent

**Système multi-agents (LLM + RAG + LangGraph) qui enquête automatiquement sur les écarts entre prévisions Batch et Real-Time, identifie la cause racine et la justifie par des preuves.**

Auteures : Imen Kammoun et Mariam Kbaier
Plateforme : Databricks Free Edition (compute serverless, Python / SQL / PySpark)
LLM utilisé : `databricks-gpt-oss-120b` (endpoint Databricks), `temperature=0`

---

## Sommaire

1. [Contexte et objectif](#1-contexte-et-objectif)
2. [Architecture](#2-architecture)
3. [État d'avancement](#3-état-davancement)
4. [Structure du dépôt](#4-structure-du-dépôt)
5. [Les données](#5-les-données)
6. [Ce qui a été réalisé, étape par étape](#6-ce-qui-a-été-réalisé-étape-par-étape)
7. [Résultats mesurés](#7-résultats-mesurés)
8. [Analyse d'erreurs](#8-analyse-derreurs)
9. [Comment reproduire](#9-comment-reproduire)
10. [Problèmes rencontrés et solutions (Databricks)](#10-problèmes-rencontrés-et-solutions-databricks)
11. [Limites](#11-limites)
12. [Reste à faire](#12-reste-à-faire)
13. [Règles de travail](#13-règles-de-travail)

---

## 1. Contexte et objectif

Une entreprise produit deux types de prévisions de ventes :

- **Batch** : calcul périodique à partir d'un *snapshot* de données figé.
- **Real-Time** : calcul à la demande, avec des données potentiellement plus récentes.

Quand les deux résultats diffèrent (exemple : 1 000 contre 1 350 unités), l'enquête manuelle est lente et dépend de l'expertise de quelques personnes.

**Objectif** : construire un système qui mène cette enquête avec des outils en lecture seule et produit un rapport contenant :

- la **cause racine** (et les causes secondaires éventuelles) ;
- des **preuves**, chacune avec sa source (résultat d'outil ou document) ;
- un **niveau de confiance** ;
- une **recommandation** appuyée sur la documentation interne (runbook, incidents passés).

### Principe de conception

> **Le code calcule, le LLM raisonne.** Les écarts, âges de snapshots et différences sont calculés de façon déterministe (PySpark / SQL). Le LLM reçoit des résultats chiffrés, pas des milliers de lignes, et rédige l'interprétation.

---

## 2. Architecture

Deux systèmes ont été construits sur les mêmes outils, le même RAG, le même modèle.

### Configuration A : agent unique (`07_agent_single`)

Un seul LLM appelle les 5 outils de données et, en option, l'outil `search_knowledge` (RAG), puis rend un JSON.

### Configuration B/C : 3 agents avec LangGraph (`08_multi_agent`)

```
Question : "Pourquoi Batch et Real-Time diffèrent ?"
            |
   +--------+---------+        (exécutés en parallèle)
   |                  |
Agent 1            Agent 2
Investigateur      Connaissance (RAG)
5 outils de        Recherche hybride dans
données (lecture   la documentation métier
seule)             (règles, définitions,
Collecte les       incidents, runbook)
faits, ne conclut
pas
   |                  |
   +--------+---------+
            |
        Agent 3
   Analyste et Vérificateur
   (croise, choisit la cause racine, rédige le rapport)
            |
   Vérification par le CODE (pas par le LLM)
   - chaque source citée existe-t-elle (outil appelé ou document retrouvé) ?
   - chaque nombre cité figure-t-il dans les entrées ?
   En cas de problème : une seule révision de l'agent 3.
            |
   Rapport final : primary_cause, secondary_causes,
   evidence (affirmation + source), confidence,
   recommendation, verification
```

Points de conception :

- Les agents 1 et 2 tournent en parallèle (`PARALLEL = True`). Si l'environnement posait problème avec les threads, `PARALLEL = False` les exécute l'un après l'autre avec le même résultat. Sur notre Databricks, le parallèle a fonctionné.
- Le résultat de la vérification (réussie ou non, révisée ou non) est conservé dans le rapport, rien n'est caché.
- Le mode sans RAG existe dans le code (`USE_RAG = False`), mais **il n'a pas été mesuré** (voir sections 7 et 12).

---

## 3. État d'avancement

| Étape | Contenu | État |
|---|---|---|
| 1 | Environnement Databricks, schéma et volume | Fait |
| 2 | Dataset synthétique (7 tables Delta) | Fait |
| 3 | Moteur de comparaison PySpark (4 tables `cmp_*`) | Fait |
| 4 | Exploration des données et fiches d'enquête | Fait |
| 5 | Baseline par règles (sans LLM) | Fait : 12/12 |
| 6 | 5 outils de l'agent en lecture seule + tests | Fait : tous les tests OK |
| 7 | RAG : TF-IDF puis recherche hybride BM25 + embeddings | Fait |
| 8 | Connexion au LLM et tool calling (`06_llm_connection_test`) | Fait |
| 9 | Agent unique, configuration A (`07_agent_single`) | Fait et mesuré (avec RAG) |
| 10 | 3 agents avec LangGraph (`08_multi_agent`) | Fait et mesuré (avec RAG, 1 passe) |
| 11 | Dataset v2 plus difficile, découpage développement/test | À faire |
| 12 | Rapport d'évaluation final, démo | À faire |

---

## 4. Structure du dépôt

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
    ├── 05d_rag_hybrid
    ├── 06_llm_connection_test
    ├── 07_agent_single
    └── 08_multi_agent
```

Tous les notebooks doivent être dans le **même dossier** du workspace, car ils s'appellent entre eux avec `%run ./nom_du_notebook`.

---

## 5. Les données

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

### Tables de résultats des agents

| Table | Contenu |
|---|---|
| `agent_runs` | Premier run de l'agent unique (10/12). Conservée telle quelle : c'est une donnée à rapporter. |
| `agent_runs_v2` | Runs de la version corrigée de `07_agent_single` et de `08_multi_agent` (colonnes différentes de `agent_runs`, d'où une nouvelle table). La colonne `config` distingue les systèmes, par exemple `multi_rag_ma_v1_databricks-gpt-oss-120b`. Colonnes : `batch_id`, `config`, `run`, `case_id`, `predicted`, `secondary`, `confidence`, `n_tools`, `tools`, `n_llm`, `seconds`, `repaired`, `correct`, `raw`, `first_raw`, `tool_outputs`, `ts`. |

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

## 6. Ce qui a été réalisé, étape par étape

### 6.1 Préparation de l'environnement (`00_setup`)
Création du schéma `forecast_debug` dans le catalogue `workspace`, puis du volume `raw` avec les dossiers `data/` et `rag_docs/`.

### 6.2 Chargement des données (`00_load_data`)
Lit les CSV du volume et crée les 6 tables Delta, plus `rag_documents`. Contrôle : 24, 96, 96, 24, 12, 12 lignes et 10 documents.

### 6.3 Moteur de comparaison (`01_comparison_engine_pyspark`)
Calcule en PySpark les 4 tables `cmp_*`. Scala n'est pas supporté sur la Free Edition ; le chemin principal est donc PySpark.

### 6.4 Exploration (`02_data_exploration`, `02b_fiches_enquete`)
Requêtes SQL et fonction `fiche(case_id)` qui répond à 5 questions pour un cas : l'écart, la fraîcheur, les features qui diffèrent, les configs qui diffèrent, le warning du pipeline. Plusieurs cas ont été résolus à la main avant toute IA.

### 6.5 Baseline sans LLM (`03_rule_based_baseline`)
Fonction `investigate_case(case_id)` qui applique des règles fixes :

- snapshot de plus de 24 h : `STALE_DATA`, et les différences de `sales_7d` deviennent des symptômes ;
- écart de 5 % ou moins sans différence d'entrée : `NO_ISSUE` ;
- sinon, la cause est déduite de la nature de la différence (feature, config, version du modèle).

**Résultat : 12/12.** À lire avec prudence : les règles ont été écrites en connaissant les 12 cas, et chaque cas contient une cause bien visible. Ce score valide la chaîne de données et d'évaluation, mais ne prouve pas que le système tiendrait sur des cas ambigus.

### 6.6 Les 5 outils de l'agent (`04_agent_tools`, `04b_test_tools`)

| Outil | Retourne |
|---|---|
| `get_forecasts(case_id)` | Prévisions Batch et Real-Time, écart en unités et en % |
| `compare_features(case_id)` | Les 4 features et celles qui diffèrent |
| `compare_configs(case_id)` | Les 4 configs et celles qui diffèrent |
| `check_freshness(case_id)` | Âge des snapshots en heures |
| `get_pipeline_logs(case_id)` | Statut, warnings, versions, durée |

Le notebook expose aussi `TOOL_SCHEMAS` (schémas au format OpenAI pour le tool calling) et `call_tool(name, args)`, un dispatcher sécurisé qui renvoie du JSON texte.

Choix de conception :

- **Lecture seule**, aucune requête SQL libre.
- **Liste blanche de tables** : `known_issues`, `actual_sales` et `rag_documents` sont refusées, et un test le vérifie.
- **Les outils renvoient des faits, pas de verdict.** `check_freshness` donne « 50 h », pas « obsolète ». Les règles (24 h, 5 %) vivent dans les documents RAG, ce qui donne un rôle réel au RAG.
- Un `case_id` invalide renvoie une erreur propre.

Un bug a été trouvé et corrigé grâce au test sur Databricks : un filtre de colonne incompatible avec Spark Connect, remplacé par `F.col("case_id")`.

### 6.7 Le RAG (`05_rag`, `05b_test_rag`, `05d_rag_hybrid`)

**Corpus** : 10 documents synthétiques (vue d'ensemble des pipelines, règles de fraîcheur, tolérance d'écart, définitions des features, versions et configs, 3 incidents passés, runbook, glossaire des causes), découpés par section en passages. Le notebook `05d_rag_hybrid` affiche « Index hybride construit : 15 passages ».

**Version 1 (`05_rag`, `05b_test_rag`)** : recherche TF-IDF avec scikit-learn.

**Version 2 (`05d_rag_hybrid`)** : recherche hybride, entièrement gratuite.

| Composant | Rôle |
|---|---|
| BM25 (`rank-bm25`) | Recherche par mots-clés, accents retirés |
| Embeddings (`sentence-transformers`, modèle `paraphrase-multilingual-MiniLM-L12-v2`) | Recherche par sens, adaptée aux documents français |
| Reciprocal Rank Fusion (RRF) | Fusion des deux classements, sans poids à régler |

L'outil exposé aux agents est `search_knowledge(query, k=3)`, avec son schéma `SEARCH_KNOWLEDGE_SCHEMA`.

**Mesure** (Recall@3 : la clé attendue doit apparaître dans le `doc_id` d'un des 3 premiers résultats) :

| Méthode | Score sur 10 requêtes |
|---|---|
| BM25 | 7/10 |
| Embeddings | 9/10 |
| Hybride (RRF) | 10/10 |

Limites : 10 requêtes écrites en connaissant les documents, test approximatif (compare un mot-clé au `doc_id`). Un écart d'un point n'est pas significatif. Deux membres de l'équipe ont obtenu des scores différents parce qu'elles avaient des listes de test de tailles différentes (7 et 10 requêtes), pas à cause du code.

Non exécuté : `05c_vector_search` (Databricks Vector Search) a été proposé mais n'a pas été lancé.

### 6.8 Connexion au LLM (`06_llm_connection_test`)
Liste des endpoints de serving disponibles dans la Free Edition, test d'un appel simple, puis test du tool calling. Le client est créé avec `w.serving_endpoints.get_open_ai_client()` (le message de dépréciation affiché est sans gravité). Modèle retenu : `databricks-gpt-oss-120b`.

### 6.9 Agent unique, configuration A (`07_agent_single`)

- Boucle : le LLM appelle des outils jusqu'à produire une réponse finale JSON (`primary_cause`, `secondary_causes`, preuves, confiance, recommandation).
- Le prompt est figé : `PROMPT_VERSION = "v1"`. Il n'a pas été modifié après avoir vu les résultats.
- L'agent ne lit jamais `known_issues`. Seule la cellule d'évaluation le fait, pour noter les réponses.

Améliorations de la version corrigée (par rapport au premier essai) :

| Problème du premier essai | Correction |
|---|---|
| CASE_004 : réponse finale illisible (INVALID) | Une seule relance sans outils demande le JSON. Elle est signalée (`repaired`) et la première réponse brute est conservée. |
| Cause d'un échec invisible | Enregistrement du `finish_reason` (`length` voudrait dire réponse coupée) ; `max_tokens` porté à 4000. |
| Un cas qui plante arrêtait les 12 | Chaque cas est protégé, l'erreur est notée, la boucle continue. |
| Un seul run, non fiable | `N_RUNS` : plusieurs passes, stabilité par cas. |
| Seule la cause principale était mesurée | Précision et rappel sur l'ensemble des causes (principale + secondaires). |
| Aucune mesure de coût | Appels LLM, appels d'outils, temps par cas. |
| Preuves non vérifiées | Contrôle approximatif : combien de nombres cités se retrouvent dans les sorties d'outils ou du RAG. |

### 6.10 Trois agents avec LangGraph (`08_multi_agent`)

Voir le schéma en section 2. Le rapport de CASE_009 contient une cause racine, des preuves avec source (`tool:...` ou `doc:...`), une confiance et une recommandation qui cite le runbook et l'incident de juin 2026. Il s'appuie sur la règle « symptôme, pas cause racine » tirée du document de fraîcheur.

#### Réglages (cellule de configuration)

| Paramètre | Valeur | Rôle |
|---|---|---|
| `MODEL` | `databricks-gpt-oss-120b` | Même modèle que l'agent unique |
| `USE_RAG` | `True` | `False` désactive l'Agent 2 (liste de règles vide) et interdit à l'Agent 3 d'inventer des seuils |
| `PARALLEL` | `True` | `False` : Agent 1 puis Agent 2 en séquence |
| `N_RUNS` | `1` | Nombre de passes complètes des 12 cas |
| `MAX_STEPS` | `10` | Nombre maximal d'étapes de la boucle d'outils d'un agent |
| `MAX_TOKENS` | `4000` | Longueur maximale d'une réponse du LLM |
| `PROMPT_VERSION` | `ma_v1` | Version du prompt, enregistrée dans le nom de configuration (`multi_rag_ma_v1_databricks-gpt-oss-120b`) |

Tous les appels LLM sont faits à `temperature=0`, avec jusqu'à 4 tentatives et une attente croissante en cas d'erreur de l'endpoint (limite de débit par exemple).

#### Boucle commune des agents (`run_llm_agent`)

Une seule fonction fait tourner les trois agents : le LLM appelle des outils jusqu'à produire une réponse sans appel d'outil, qui doit être un objet JSON. Si la réponse est vide, illisible ou ne passe pas la validation de forme, **une seule relance** sans outils demande le JSON (`repaired = True`). Elle est signalée dans les résultats. Si la limite d'étapes est atteinte, l'erreur `max_steps` est notée.

#### Les trois agents

| Agent | Outils | Entrée | Sortie JSON |
|---|---|---|---|
| 1. Investigateur | Les 5 outils de données | Le `case_id` | `facts` (chaque fait : phrase avec valeurs exactes + source `tool:<nom>`), `differences` (features et configs qui diffèrent) |
| 2. Connaissance | `search_knowledge` uniquement | Une consigne **fixe**, sans le cas | `rules` (règle reprise des documents + source `doc:<doc_id>`), `similar_incidents` (résumé, résolution, source) |
| 3. Analyste et Vérificateur | Aucun | Les faits de l'Agent 1 et les règles de l'Agent 2 | `primary_cause`, `secondary_causes`, `evidence` (affirmation + source), `confidence` (`high`, `medium`, `low`), `recommendation` |

Consignes principales des prompts :

- **Agent 1** : appeler les 5 outils, ne pas conclure, n'appliquer aucun seuil, ne rien inventer.
- **Agent 2** : n'a pas accès aux données du cas, fait plusieurs recherches (fraîcheur, tolérance, définitions des features, versions, symptôme contre cause racine, causes multiples, incidents, runbook), ne retient que ce que les documents disent.
- **Agent 3** : appliquer les règles documentées aux faits ; écarter les hypothèses non prouvées ; un symptôme n'est pas une cause racine ; si aucune cause n'est établie, répondre `NO_ISSUE` ; choisir **une** cause principale et mettre les autres en secondaires ; donner une recommandation concrète tirée du runbook ou des incidents ; citer pour chaque preuve une source exacte.

Le prompt de l'Agent 3 ne contient aucune règle de priorité entre deux causes indépendantes, ce qui explique en partie les résultats sur CASE_006.

#### Le graphe LangGraph

L'état partagé contient `case_id`, les sorties des Agents 1 et 2 (`inv`, `kn`), le rapport final et la réponse brute de l'Agent 3. Le graphe a trois nœuds (`investigator`, `knowledge`, `analyst`). En mode parallèle, `START` lance les deux premiers nœuds en même temps et le nœud `analyst` attend les deux avant de démarrer ; en mode séquentiel, les nœuds s'enchaînent.

#### Vérification par le code

Après l'Agent 3, le code contrôle chaque preuve :

1. **Source valide** : une source `tool:<nom>` n'est acceptée que si l'Agent 1 a réellement appelé cet outil ; une source `doc:<doc_id>` n'est acceptée que si ce document a été renvoyé par `search_knowledge` pendant le run.
2. **Nombres retrouvés** : tout nombre cité dans une affirmation doit apparaître dans les sorties d'outils ou dans les passages renvoyés par le RAG.
3. La liste de preuves et la recommandation ne doivent pas être vides.

Si des problèmes sont détectés, l'Agent 3 reçoit la liste (8 au plus) et les sources autorisées, et **une seule révision** est faite. Le champ `verification` du rapport contient `passed`, `problems` (ceux qui restent après révision), `revised` et `sources_allowed`.

Portée de ce contrôle : il vérifie l'existence des sources et des nombres, pas la justesse du raisonnement. Il ne contrôle pas les nombres de la recommandation ni le sens des phrases.

#### Exécution et résultats enregistrés

La cellule d'exécution lance les 12 cas, note pour chacun la cause principale, les causes secondaires, la confiance, le nombre d'appels LLM, le temps et les indicateurs `RELANCÉ`, `RÉVISÉ`, `VÉRIF. ÉCHOUÉE`, `RAPPORT ILLISIBLE`. Un cas qui plante est noté sans arrêter la boucle. Une cellule de diagnostic liste ensuite les rapports relancés, révisés, illisibles ou dont la vérification a échoué. La dernière cellule est la seule à lire `known_issues` : exactitude de la cause principale, précision et rappel sur l'ensemble des causes, appels LLM, temps, vérifications, stabilité par cas si `N_RUNS > 1`. Les lignes sont ajoutées à la table `agent_runs_v2`, la colonne `config` distinguant les systèmes.

---

## 7. Résultats mesurés

Tous les chiffres viennent de runs réels sur Databricks. Ils valident le système sur **12 scénarios contrôlés**, pas en production.

| Système | Cause principale | Précision / rappel (ensemble de causes) | LLM / outils par cas | Temps par cas |
|---|---|---|---|---|
| Baseline par règles | 12/12 | non applicable | non applicable | non applicable |
| Agent unique + RAG, premier essai (`agent_runs`) | 10/12 | non mesuré | non mesuré | non mesuré |
| Agent unique + RAG, 1 passe (`agent_runs_v2`) | 11/12 (91,7 %) | 0,93 / 1,00 | 6,8 / 5,8 | 9,2 s |
| **Agent unique + RAG, 3 passes** | **10, 11, 11 sur 12 : moyenne 88,9 % (min 83,3 %, max 91,7 %)** | 0,90 / 0,90 | 6,4 / 5,4 | 8,1 s |
| 3 agents LangGraph + RAG, 1 passe | 11/12 (91,7 %) | 0,87 / 1,00 | 11,2 appels LLM | 16,5 s |

Autres mesures :

- **Agent unique, 3 passes** : 1 réponse sur 36 était vide (CASE_009, passe 1) et a été récupérée par la relance de format ; 0 réponse illisible ; 311 nombres cités sur 312 retrouvés dans les sorties d'outils.
- **Agent unique, 1 passe** : 0 relance, 105 nombres sur 105 retrouvés.
- **3 agents** : vérification par le code réussie sur 12/12 ; 1 rapport révisé (CASE_003) ; 0 relance de format ; 0 rapport illisible.

### Comment lire ces chiffres

- Avec 12 cas, **une erreur vaut 8 points**. Les écarts entre 88,9 %, 91,7 % et 83,3 % sont dans le bruit.
- **Il ne faut pas écrire que les 3 agents sont meilleurs** que l'agent unique : 91,7 % sur une passe ne se distingue pas de 88,9 % en moyenne sur 3 passes. Le coût, lui, est mesuré : environ le double en appels LLM et en temps.
- Trois passes mesurent la stabilité, pas la capacité à généraliser : il n'y a que 12 situations différentes.
- La baseline par règles fait 12/12 parce qu'elle a été écrite en connaissant ces cas. Un agent à 88,9 % n'est donc pas démontré « moins bon » : ce dataset est trop facile pour trancher.
- À présenter ainsi : « 88,9 % en moyenne sur 12 scénarios contrôlés, 3 passes (min 83,3 %, max 91,7 %) ». Éviter « 91,7 % » seul, qui est le meilleur run.

---

## 8. Analyse d'erreurs

| Cas | Observation | Nature |
|---|---|---|
| 004 (premier essai) | Réponse finale illisible, score INVALID. Le run n'avait pas gardé la réponse brute, la cause exacte n'est donc pas prouvée. Une réponse vide, constatée ensuite sur CASE_009, est l'explication la plus probable. Non reproduit dans les runs suivants. | Format / fiabilité |
| 006 | Faux sur la cause principale dans tous les runs de l'agent unique (CONFIG_MISMATCH au lieu de FEATURE_MISMATCH). Selon les runs, la deuxième cause est trouvée en secondaire (run à 1 passe et 3 agents) ou absente (3 passes). Le prompt ne donne aucune règle de priorité entre deux causes. | Raisonnement et ambiguïté de consigne |
| 001 | Instable : FEATURE_MISMATCH à la passe 1, DATA_MISMATCH aux passes 2 et 3, parfois une cause secondaire en trop. `sales_7d` est à la fois une donnée d'historique et une feature, donc les deux causes se recouvrent (interprétation, non vérifiée dans les documents). | Instabilité |
| 009 | Cause principale toujours juste (STALE_DATA), mais une cause secondaire (DATA_MISMATCH ou FEATURE_MISMATCH) est parfois ajoutée alors que la règle dit que la différence de `sales_7d` est un symptôme. L'agent a aussi cherché « seuil 30% », qui n'existe pas dans les documents ; le RAG a quand même renvoyé les bons passages. | Précision |
| 003 (3 agents) | Rapport révisé une fois après échec de la vérification. Le rapport final ne conserve que les problèmes restants, pas les problèmes initiaux. | Traçabilité à améliorer |

Aucun changement de prompt ni d'étiquette n'a été fait pour faire passer ces cas. Si le prompt est modifié plus tard, il faut passer `PROMPT_VERSION` à une nouvelle valeur (par exemple `v2`), noter pourquoi, et évaluer sur des cas non utilisés pour le régler (dataset v2).

---

## 9. Comment reproduire

1. Créer un compte **Databricks Free Edition**.
2. Créer un notebook `00_setup` (Python, Serverless) et exécuter la création du schéma `forecast_debug`.
3. Dans **Catalog**, créer le volume `raw`, avec les dossiers `data/` et `rag_docs/`, puis y téléverser les 6 CSV et les 10 fichiers `.md`.
4. Importer les notebooks dans **un même dossier** du workspace (nom exact, sans suffixe `(1)`).
5. Exécuter dans l'ordre :
   1. `00_load_data` : 6 tables + `rag_documents`
   2. `01_comparison_engine_pyspark` : 4 tables `cmp_*`
   3. `03_rule_based_baseline` : doit afficher `Accuracy baseline règles: 12/12`
   4. `04b_test_tools` : doit afficher `TOUS LES TESTS OK`
   5. `05d_rag_hybrid` : doit afficher « Index hybride construit »
   6. `06_llm_connection_test` : liste des endpoints et test du tool calling
   7. `07_agent_single` : cellule par cellule, jusqu'au test CASE_009, puis les 12 cas, le diagnostic et l'évaluation
   8. `08_multi_agent` : même démarche, démonstration CASE_009 puis les 12 cas

Réglages dans les cellules de configuration de `07_agent_single` : `MODEL`, `USE_RAG` (True/False), `N_RUNS` (nombre de passes complètes). Dans `08_multi_agent` : `USE_RAG` et `PARALLEL`.

Dépendances installées dans le notebook appelant (première cellule, suivie de `dbutils.library.restartPython()`) : `rank-bm25` et `sentence-transformers` pour le RAG ; `08_multi_agent` installe `langgraph rank-bm25 sentence-transformers openai databricks-sdk --upgrade`. Cette mise à jour fait changer la version de `databricks-sdk` (0.122.0 vers 0.168.0 lors de nos essais), ce que Databricks signale par un avertissement sur les paquets système ; il n'a pas empêché le fonctionnement.

---

## 10. Problèmes rencontrés et solutions (Databricks)

| Symptôme | Cause | Solution |
|---|---|---|
| `search_knowledge` ou `call_tool` absent après les `%run` | Dans Databricks, `%run` doit être la **toute première ligne** de sa cellule. Un commentaire avant lui empêche le chargement. | Un `%run` seul par cellule, sans commentaire avant. |
| `TOOL_SCHEMAS` ou `call_tool` disparaissent après un `%run` | Un `%pip` ou un redémarrage de Python remet à zéro l'état du notebook. | Installer les paquets et redémarrer Python dans le notebook appelant, avant les `%run`. Charger `05d_rag_hybrid` d'abord, puis `04_agent_tools` en dernier. |
| Filtre de colonne en erreur avec Spark Connect | Syntaxe incompatible | Utiliser `F.col("case_id")`. |
| `Input/output error` sur des fichiers du Git folder, `.git/config` illisible | Dossier Git du workspace endommagé (cause exacte non établie) | Exporter les notebooks, supprimer le Git folder, le re-cloner ; en dernier recours, déposer les fichiers par l'upload web de GitHub. |
| `The link to your GitHub account does not have access` | Application Databricks non installée sur le dépôt GitHub | Installer l'application sur le dépôt, ou passer par l'upload web. |
| Réponse finale du LLM vide ou illisible | Constatée une fois sur 36 réponses | Relance de format unique, signalée dans les résultats. |

Conseils Git : créer les notebooks directement dans le Git folder plutôt que de les déplacer, ne jamais ouvrir le dossier `.git`, faire un Pull avant de travailler et un Commit & Push en fin de session.

---

## 11. Limites

- **Données synthétiques** : les résultats valident le système sur des scénarios contrôlés, pas en production. On ne dira jamais « 95 % en production », mais « X % sur N scénarios contrôlés ».
- **Dataset actuel très petit et propre** : 12 cas, une ou deux causes visibles par cas, valeurs rondes. Les règles font déjà 100 %, donc il ne suffit pas pour montrer l'apport d'un LLM.
- **Documents RAG et cas écrits ensemble**, avec les mêmes seuils (24 h, 5 %) : le RAG peut paraître plus utile que sur de vraies règles métier.
- **Évaluation du RAG** sur 10 requêtes écrites par l'équipe.
- **Non reproductibilité** : même à `temperature=0`, les résultats varient d'un run à l'autre (CASE_001, CASE_009, CASE_006).
- **Contrôle des preuves faible** : il vérifie que les nombres et les sources cités existent dans les entrées, pas que le raisonnement est juste.
- **Agent 2 indépendant du cas** : sa consigne est la même pour les 12 cas (il ne reçoit pas le `case_id`), donc il rassemble des règles générales et non des documents choisis selon le cas. Les écarts d'un cas à l'autre viennent du LLM, pas d'une recherche ciblée.
- **Agent 1 non adaptatif** : le prompt lui demande d'appeler les 5 outils pour tous les cas, et l'Agent 3 n'a aucun outil pour redemander une donnée. L'agent unique, lui, choisit ses outils.
- **Réponse brute initiale non conservée** dans le système à 3 agents : si une réponse est relancée ou révisée, la colonne `first_raw` reste vide et le rapport final ne garde que les problèmes restants.
- **Une seule passe** pour les 3 agents ; les configurations B (sans RAG) et C (avec RAG) n'ont pas été comparées, ni l'agent unique sans RAG.
- **Free Edition** : accès internet sortant limité, quotas de compute, limites de débit possibles sur le LLM.
- Plus d'agents implique plus de latence et de coût : environ le double ici, sans gain de précision démontré.

---

## 12. Reste à faire

1. **Dataset v2** : 60 à 100 cas avec bruit, cas ambigus, cas hors règles, cas « preuves insuffisantes », et un découpage **développement / test** (le groupe de test n'est jamais utilisé pendant le réglage). C'est le seul moyen de comparer sérieusement règles, agent unique et 3 agents.
2. **Évaluation complète** sur le même jeu de test, avec plusieurs passes par configuration (agent unique avec et sans RAG, 3 agents avec et sans RAG).
3. **Rapport d'évaluation final** : tableau des résultats, analyse d'erreurs (001, 006, 009), coût, limites.
4. **Traçabilité** : conserver les problèmes détectés avant révision et la première réponse brute de l'Agent 3 dans le rapport et dans `first_raw`.
5. **Agent 2 ciblé** (optionnel) : lui transmettre les faits de l'Agent 1, ou lancer le RAG après l'Agent 1, pour chercher des documents adaptés au cas ; cela supprimerait le parallélisme.
6. **Démo de 5 minutes** : une question, un rapport structuré (CASE_009 convient bien).

---

## 13. Règles de travail

- Valider les données et les outils avant de toucher au LLM.
- Ne jamais donner accès à `known_issues` aux agents.
- Ne jamais modifier les données ni l'étiquette d'un cas pour que l'agent réussisse.
- Figer chaque version du prompt (`PROMPT_VERSION`) et la changer à chaque modification.
- Garder toutes les exécutions, y compris les mauvaises.
- Pull avant de travailler, commit et push en fin de session.
