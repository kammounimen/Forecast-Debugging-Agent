# Rapport d'évaluation : Forecast Debugging Agent

*Rapport généré automatiquement le 2026-11-20 20:55 UTC à partir de `agent_runs_v2` et `known_issues`. Tous les chiffres sont recalculés à chaque exécution.*

> **Avertissement.** Les mesures portent sur des cas synthétiques. Elles montrent que le système fonctionne sur des scénarios contrôlés, pas qu'il fonctionnerait sur de vraies données.

## 1. Configurations évaluées

| Configuration | Source | Passes | Lignes |
|---|---|---|---|
| Agent unique + RAG [v1] | 20261118_210216_db50df | 3 | 36 |
| 3 agents + RAG [ma_v1] | 20261118_212936_875073 | 1 | 12 |

## 2. Résultats globaux

| Mesure | Agent unique + RAG [v1] | 3 agents + RAG [ma_v1] |
|---|---|---|
| Passes | 3 | 1 |
| Cas évalués | 12 | 12 |
| Cause principale par passe | 10/12 ; 11/12 ; 11/12 | 11/12 |
| **Exactitude moyenne** | **88,9 %** | **91,7 %** |
| Min – max | 83,3 % – 91,7 % | une seule passe |
| Intervalle de confiance 95 % (Wilson) | [65 % ; 99 %] | [65 % ; 99 %] |
| Précision (ensemble de causes) | 0,90 | 0,87 |
| Rappel (ensemble de causes) | 0,90 | 1,00 |
| Appels LLM par cas | 6,4 | 11,2 |
| Temps moyen par cas (s) | 8,1 | 16,5 |
| Appels d'outils par cas | 5,4 | 8,2 |
| Relances de format | 1/36 | 0/12 |
|   dont réponse initiale vide (agent unique) | 1 | n/a |
| Réponses illisibles | 0/36 | 0/12 |
| Nombres cités retrouvés dans les sorties d'outils | 311/312 | 92/92 |
| Vérification par le code (réussie / révisés) | n/a | 12/12 / 1 |
| Recommandation fournie | 0/36 | 12/12 |

## 3. Résultat par cas

| Cas | Causes attendues | Agent unique + RAG [v1] | 3 agents + RAG [ma_v1] |
|---|---|---|---|
| CASE_001 | DATA_MISMATCH | INSTABLE : 1/3 faux (FEATURE_MISMATCH x1) | OK, causes secondaires différentes |
| CASE_002 | FEATURE_MISMATCH | OK | OK |
| CASE_003 | CONFIG_MISMATCH | OK | OK |
| CASE_004 | STALE_DATA | OK | OK |
| CASE_005 | NO_ISSUE | OK | OK |
| CASE_006 | CONFIG_MISMATCH+FEATURE_MISMATCH | 3/3 faux (CONFIG_MISMATCH x3) | 1/1 faux (CONFIG_MISMATCH x1) |
| CASE_007 | NO_ISSUE | OK | OK |
| CASE_008 | MODEL_VERSION_MISMATCH | OK | OK |
| CASE_009 | STALE_DATA | OK, causes secondaires différentes | OK, causes secondaires différentes |
| CASE_010 | NO_ISSUE | OK | OK |
| CASE_011 | FEATURE_MISMATCH | OK | OK |
| CASE_012 | CONFIG_MISMATCH | OK | OK |

## 4. Ventilation par groupe de cas

| Groupe | Cas | Agent unique + RAG [v1] | 3 agents + RAG [ma_v1] |
|---|---|---|---|
| difficulty = easy | 5 | 93,3 % | 100,0 % |
| difficulty = hard | 2 | 50,0 % | 50,0 % |
| difficulty = medium | 5 | 100,0 % | 100,0 % |

| Groupe | Cas | Agent unique + RAG [v1] | 3 agents + RAG [ma_v1] |
|---|---|---|---|
| requires_rag = false | 4 | 91,7 % | 100,0 % |
| requires_rag = true | 8 | 87,5 % | 87,5 % |

## 5. Analyse d'erreurs (générée à partir des prédictions)

#### CASE_001 : attendu DATA_MISMATCH (difficulty = easy, requires_rag = false)
Historique de ventes sales_7d différent (100 vs 120).

- **Agent unique + RAG [v1]** : cause principale prédite FEATURE_MISMATCH x1, DATA_MISMATCH x2. instable (1/3 passes fausses) ; autre(s) cause(s) prédite(s) : FEATURE_MISMATCH (1/3).
- **3 agents + RAG [ma_v1]** : cause principale prédite DATA_MISMATCH x1. cause(s) en trop : FEATURE_MISMATCH (1/1 passes).

#### CASE_006 : attendu CONFIG_MISMATCH + FEATURE_MISMATCH (difficulty = hard, requires_rag = true)
Deux causes simultanées : promotion différente et feature set FS1 vs FS2.

- **Agent unique + RAG [v1]** : cause principale prédite CONFIG_MISMATCH x3. erreur stable (3/3 passes fausses) ; la cause principale prédite fait partie des causes attendues : possible ambiguïté sur l'ordre.
- **3 agents + RAG [ma_v1]** : cause principale prédite CONFIG_MISMATCH x1. erreur stable (1/1 passes fausses) ; la cause principale prédite fait partie des causes attendues : possible ambiguïté sur l'ordre.

#### CASE_009 : attendu STALE_DATA (difficulty = hard, requires_rag = true)
sales_7d diffère (90 vs 125) mais c'est un symptôme : la vraie cause est le snapshot Batch obsolète de 49 h.

- **Agent unique + RAG [v1]** : cause principale prédite STALE_DATA x3. cause(s) en trop : DATA_MISMATCH (1/3 passes), FEATURE_MISMATCH (2/3 passes).
- **3 agents + RAG [ma_v1]** : cause principale prédite STALE_DATA x1. cause(s) en trop : FEATURE_MISMATCH (1/1 passes).

### Réponses initiales vides du modèle

- Agent unique + RAG [v1] : CASE_009 (passe 1)

## 6. Coût

| Configuration | Appels LLM / cas | Temps / cas (s) | Outils / cas | Appels vs Agent unique + RAG [v1] | Temps vs Agent unique + RAG [v1] |
|---|---|---|---|---|---|
| Agent unique + RAG [v1] | 6,4 | 8,1 | 5,4 | référence | référence |
| 3 agents + RAG [ma_v1] | 11,2 | 16,5 | 8,2 | x 1,75 | x 2,04 |

Les jetons consommés ne sont pas enregistrés : le nombre d'appels est un indicateur, pas le coût réel.

## 7. Comparaisons deux à deux

| Configuration A | Configuration B | Cas communs | Cas où A fait mieux | Cas où B fait mieux | p (test des signes) |
|---|---|---|---|---|---|
| Agent unique + RAG [v1] | 3 agents + RAG [ma_v1] | 12 | 0 | 1 | 1,000 |

Un cas « fait mieux » si sa fraction de passes correctes est plus élevée. Une valeur p élevée signifie que la différence n'est pas distinguable du hasard.

## 8. Conclusions déduites des chiffres

- **Agent unique + RAG [v1]** : 88,9 % de cause principale correcte, intervalle de confiance 95 % [65 % ; 99 %], sur 12 cas et 3 passe(s).
- **3 agents + RAG [ma_v1]** : 91,7 % de cause principale correcte, intervalle de confiance 95 % [65 % ; 99 %], sur 12 cas et 1 passe(s).
- Les intervalles de confiance de toutes les configurations **se recouvrent** : les écarts d'exactitude ne sont pas statistiquement significatifs.
- Cas toujours faux pour **toutes** les configurations à LLM : CASE_006.
- Cas instables entre passes : CASE_001 (Agent unique + RAG [v1]).
- Coût de **3 agents + RAG [ma_v1]** par rapport à **Agent unique + RAG [v1]** : x 1,75 appels LLM, x 2,04 en temps.
- Vérification par le code, **3 agents + RAG [ma_v1]** : 12/12 rapports conformes, 1 révisés. Un rapport conforme peut contenir une mauvaise cause : le contrôle porte sur les sources et les nombres, pas sur le raisonnement.

## 9. Limites

- **12 cas** : une erreur change l'exactitude de 8,3 points. Échantillon trop petit pour conclure.
- **Une seule passe** pour : 3 agents + RAG [ma_v1]. Leur stabilité n'est pas connue.
- Les passes multiples (Agent unique + RAG [v1]) portent sur les mêmes cas : elles mesurent la stabilité, pas la généralisation.
- La baseline par règles n'est pas incluse (table `rule_baseline_runs` absente).
- **Effet du RAG non mesuré** : aucun couple de configurations avec et sans RAG pour un même système.
- **Un seul modèle** (databricks-gpt-oss-120b) : aucune comparaison entre LLM.
- **Pas de séparation développement / test** dans `known_issues` : les erreurs observées ne doivent pas servir à régler un prompt sans nouveaux cas.
- Les sorties d'outils sont tronquées à 12 000 caractères dans la table pour : Agent unique + RAG [v1], 3 agents + RAG [ma_v1]. Le contrôle des nombres cités peut sous-estimer l'ancrage.
- **Données synthétiques** : aucune conclusion sur la production.
- Les étiquettes peuvent être ambiguës (par exemple deux causes simultanées sans règle d'ordre, ou notions qui se recouvrent) : une « erreur » n'est pas toujours une faute du modèle.
- Les **jetons** consommés ne sont pas mesurés.
- Aucune évaluation humaine de la qualité des preuves n'est incluse.

## Annexe : tous les runs de la table

| Date (UTC) | batch_id | Configuration | Passes | Lignes | Exactitude moyenne |
|---|---|---|---|---|---|
| 2026-11-18T20:57:26 | 20261118_205533_6ebf80 | Agent unique + RAG [v1] | 1 | 12 | 91,7 % |
| 2026-11-18T21:07:18 | 20261118_210216_db50df | Agent unique + RAG [v1] | 3 | 36 | 88,9 % |
| 2026-11-18T21:32:56 | 20261118_212936_875073 | 3 agents + RAG [ma_v1] | 1 | 12 | 91,7 % |
