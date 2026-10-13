# Kraken Trade Agent

Agent de trading autonome pour un compte **Kraken** (spot, en EUR), avec un
cerveau IA (**Claude**) encadré par des règles de survie non négociables.

> **À lire avant tout.** Aucun programme ne peut garantir un gain. La plupart
> des stratégies de trading automatique perdent de l'argent une fois les frais
> payés, et avec 10 € les frais Kraken (~0,4 % par ordre, ~0,9 % par
> aller-retour avec le glissement) pèsent lourd. Cet agent est conçu d'abord
> pour **ne pas mourir** : limiter les pertes, puis chercher les gains. Faites-le
> tourner en simulation pendant des semaines avant d'engager de l'argent réel.

## Comment il fonctionne

```
 Kraken (prix, bougies)
        │
        ▼
 Scanner de signaux ──► candidats (tendance + cassure, stop basé sur l'ATR)
        │
        ▼
 Cerveau : Claude (IA) ou règles ──► acheter / vendre / ne rien faire
        │
        ▼
 Gestionnaire de risque ──► taille de position, refus si une règle de survie est en jeu
        │
        ▼
 Courtier : simulation (paper) ou Kraken réel (live)
```

1. **Univers** : les paires EUR les plus liquides de Kraken (volume 24 h ≥ 1 M€,
   spread ≤ 0,3 %), stablecoins exclus. L'univers peut aussi être fixé à la main.
2. **Scanner** : à chaque bougie clôturée (4 h par défaut), il cherche les
   paires en tendance haussière (EMA 12 > EMA 26, prix > EMA 100) qui cassent
   leur plus haut des 20 dernières bougies, avec un RSI entre 50 et 72 et un
   momentum positif. Les gains visés doivent couvrir au moins 4 fois les frais.
3. **Cerveau IA** : Claude reçoit l'état du compte, les positions et les
   candidats (indicateurs, trajectoire récente, historique des trades), puis
   choisit d'acheter l'un des candidats, de vendre ou de ne rien faire. Il ne
   peut acheter que parmi les candidats du scanner. Il n'est appelé que s'il y a
   une décision à prendre, avec un plafond quotidien d'appels.
4. **Sorties** : chaque position reçoit un **stop-loss posé chez Kraken**, qui
   protège même si votre ordinateur s'éteint. Un stop suiveur (3 ATR sous le
   plus haut) ne fait que monter, et passe au point mort dès +1R de gain.
   Tendance cassée ou position qui stagne : le cerveau décide de sortir ou non.

## Règles de survie (appliquées quoi que décide l'IA)

| Règle | Défaut |
|---|---|
| Spot uniquement : ni levier, ni marge, ni vente à découvert | toujours |
| Perte visée si le stop est touché (frais compris) | 2 % du capital |
| Taille max d'une position | 90 % du capital |
| Positions simultanées | 1 |
| **Plancher de survie** : tout vendre et s'arrêter définitivement | capital < 75 % du départ |
| Pause des achats après une perte journalière | -6 % |
| Pause après 3 pertes d'affilée | 12 h |
| Un stop ne peut jamais faire passer le capital sous le plancher | toujours |
| Stop-loss impossible à poser : revente immédiate | toujours |
| Sans réponse de l'IA (erreur, refus, budget épuisé) : pas de nouvel achat | toujours |
| Aucune fonction de retrait dans le code | toujours |

Une fois le plancher touché, l'agent reste arrêté. Seule la commande `revive`,
lancée par vous, le réactive.

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Clé API Kraken

Sur Kraken : *Settings → API → Create API key*. Cochez **uniquement** :

- Query Funds
- Query Open Orders & Trades
- Query Closed Orders & Trades
- Create & Modify Orders
- Cancel/Close Orders

**Ne cochez jamais « Withdraw Funds ».** Copiez la clé et le secret dans `.env`.
Désactivez aussi l'Auto Earn (staking automatique) sur les actifs tradés, pour
que les soldes restent disponibles à la vente.

### Clé API Claude (optionnelle)

Créez une clé sur https://console.anthropic.com et mettez-la dans
`ANTHROPIC_API_KEY`. Sans clé, l'agent tourne avec le cerveau à règles.

## Utilisation

```bash
# 1. Vérifier l'accès à Kraken, l'univers, les minimums d'ordre et vos clés
python -m kraken_agent check
python -m kraken_agent check --validate-order   # teste la permission d'ordre, sans rien exécuter

# 2. Tester la stratégie sur l'historique Kraken (720 dernières bougies)
python -m kraken_agent backtest --pairs XBTEUR,ETHEUR,SOLEUR
python -m kraken_agent backtest --csv XBTEUR_240.csv ETHEUR_240.csv   # fichiers OHLCVT Kraken
python -m kraken_agent backtest --synthetic 4                           # démo hors ligne

# 3. Simulation avec les vrais prix, sans argent (10 € virtuels)
python -m kraken_agent run

# 4. Argent réel (confirmation demandée)
python -m kraken_agent run --live

# Suivi
python -m kraken_agent status            # ou: status --live
python -m kraken_agent panic --live      # tout vendre maintenant et arrêter
python -m kraken_agent revive --live     # réactiver après un arrêt
```

Sans `--live`, tout est simulé. L'agent tourne en continu (une vérification par
minute, une analyse par bougie). Pour le laisser tourner sur un serveur, lancez-le
avec `systemd`, `tmux` ou `nohup`. Un arrêt avec `Ctrl+C` laisse les stop-loss
en place chez Kraken.

## Ce que ça coûte

- **Frais Kraken** : 0,40 % par ordre au palier de départ, soit ~0,9 % par
  aller-retour avec le glissement. Sur un trade de 6 €, c'est ~5 centimes.
  Le backtest les compte tous.
- **Minimums Kraken** : chaque paire a un montant minimum d'ordre (souvent
  quelques euros). Avec 10 €, l'agent ne tient qu'une position à la fois, et
  ignore les paires dont le minimum ferait dépasser le risque autorisé.
- **IA** : le modèle par défaut est `claude-opus-5-5`, le plus capable de sa
  gamme. Un appel coûte de l'ordre de 0,01 à 0,04 $, avec au plus 6 appels par
  jour (souvent moins : Claude n'est consulté que s'il y a un candidat ou une
  sortie à arbitrer). Sur un compte de 10 €, ce coût peut dépasser les gains.
  Pour le réduire, mettez `CLAUDE_MODEL=claude-haiku-5-5` (environ 40 fois moins
  cher) ou `BRAIN=rules` (gratuit). Le coût estimé est suivi dans `status`. Les
  requêtes activent le repli automatique côté serveur (`fallbacks: "default"`) :
  si le modèle refuse une requête, l'API la relance sur un autre modèle.

## Fichiers générés (`data/`)

- `state_paper.json` / `state_live.json` : état (positions, capital, statistiques)
- `trades_*.jsonl` : chaque achat et vente, avec la raison
- `decisions_*.jsonl` : chaque décision du cerveau et l'analyse de l'IA
- `agent_*.log` : journal complet

## Limites connues

- Le capital surveillé inclut tous les actifs du compte, mais l'agent ne vend
  que les positions qu'il a ouvertes lui-même.
- Le backtest utilise le cerveau à règles : Claude connaît l'histoire des
  marchés passés, un backtest avec lui serait biaisé. Les performances passées
  ne garantissent rien.
- Un krach soudain peut faire exécuter un stop plus bas que prévu (gap).
- Kraken ajuste parfois ses minimums d'ordre, ses frais ou les actifs autorisés
  selon les pays. Les paires refusées pour votre compte sont exclues
  automatiquement.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Les tests couvrent la signature des requêtes (vérifiée sur l'exemple officiel
de Kraken), les indicateurs, la stratégie, les règles de risque, la validation
des décisions de Claude, et le mode réel de bout en bout contre un faux Kraken
(ordres, stop serveur, stop suiveur, vente de panique).
