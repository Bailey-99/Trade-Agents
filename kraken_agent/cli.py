"""Ligne de commande : python -m kraken_agent <commande>."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from . import market
from .agent import TradingAgent
from .backtest import load_kraken_csv, run_backtest, synthetic_candles
from .brain import make_brain
from .config import Settings
from .kraken_client import KrakenClient, KrakenError
from .state import StateStore

log = logging.getLogger("kraken_agent")

LIVE_WARNING = """
!!! MODE RÉEL : l'agent va passer de vrais ordres avec votre argent sur Kraken. !!!
- Aucun gain n'est garanti : vous pouvez perdre jusqu'à {loss:.0f}% du capital avant l'arrêt de survie.
- Frais Kraken : ~{fee:.2f}% par ordre, soit ~{rt:.2f}% par aller-retour (frais + glissement).
- La clé API ne doit PAS avoir la permission de retrait.
Tapez OUI pour continuer : """


def setup_logging(settings: Settings, verbose: bool) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logfile = logging.FileHandler(settings.data_dir / f"agent_{settings.mode}.log", encoding="utf-8")
    logfile.setFormatter(fmt)
    root.handlers[:] = [console, logfile]
    for noisy in ("urllib3", "httpx", "httpx2", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_agent(settings: Settings) -> TradingAgent:
    client = KrakenClient(settings.kraken_api_key, settings.kraken_api_secret)
    store = StateStore(settings.data_dir, settings.mode)
    return TradingAgent(settings, client, store, brain=make_brain(settings))


def confirm_live(settings: Settings, assume_yes: bool) -> bool:
    if settings.mode != "live" or assume_yes:
        return True
    answer = input(
        LIVE_WARNING.format(
            loss=100 - settings.survival_floor_pct,
            fee=settings.taker_fee_pct,
            rt=settings.round_trip_cost_pct,
        )
    )
    return answer.strip().upper() == "OUI"


def cmd_run(settings: Settings, args: argparse.Namespace) -> int:
    if settings.mode == "live" and not settings.kraken_api_key:
        print("KRAKEN_API_KEY / KRAKEN_API_SECRET manquants dans .env")
        return 2
    if not confirm_live(settings, args.yes):
        print("Annulé.")
        return 1
    agent = build_agent(settings)
    log.info("démarrage en mode %s, cerveau : %s", settings.mode, agent.brain.name)
    if agent.brain.name == "claude":
        log.info("modèle IA : %s (max %d appels/jour)", settings.claude_model, settings.max_llm_calls_per_day)
    try:
        agent.run_forever(max_steps=1 if args.once else None)
    except (KrakenError, OSError) as exc:
        log.error("Kraken injoignable : %s", exc)
        return 1
    except KeyboardInterrupt:
        agent.save()
        log.info("arrêt demandé. Les positions ouvertes restent protégées par leur stop-loss.")
    return 0


def cmd_status(settings: Settings, args: argparse.Namespace) -> int:
    state = StateStore(settings.data_dir, settings.mode).load()
    if state.initial_capital is None:
        print(f"Aucun état en mode {settings.mode} (l'agent n'a jamais tourné).")
        return 0
    s = state.stats
    win_rate = s["wins"] / s["trades"] * 100 if s["trades"] else 0
    print(f"Mode               : {settings.mode}")
    print(f"Statut             : {state.status}{' — ' + state.halt_reason if state.halt_reason else ''}")
    print(f"Capital initial    : {state.initial_capital:.2f}€")
    print(f"Dernier capital    : {state.last_equity:.2f}€ ({(state.last_equity / state.initial_capital - 1) * 100:+.2f}%)")
    print(f"Plancher de survie : {state.initial_capital * settings.survival_floor_pct / 100:.2f}€")
    print(f"Trades clôturés    : {s['trades']} (gagnants {win_rate:.0f}%), P&L réalisé {s['realized_pnl']:+.4f}€")
    print(f"Frais payés        : {s['fees_paid']:.4f}€")
    llm = state.llm
    print(f"Appels IA          : {llm['total_calls']} (~{llm['est_cost_usd']:.4f}$ de coût API estimé)")
    for pos in state.positions:
        print(f"Position           : {pos.pair} {pos.volume:g} @ {pos.entry_price:g}, stop {pos.stop:g}")
    for trade in state.recent_trades[-5:]:
        print(f"  trade récent     : {json.dumps(trade, ensure_ascii=False)}")
    return 0


def cmd_check(settings: Settings, args: argparse.Namespace) -> int:
    client = KrakenClient(settings.kraken_api_key, settings.kraken_api_secret)
    try:
        pairs = market.load_pairs(client.asset_pairs(), settings.quote)
        print(f"API publique OK : {len(pairs)} paires en {settings.quote}.")
        universe = settings.universe or market.discover_universe(
            client,
            pairs,
            size=settings.universe_size,
            min_volume=settings.min_volume_24h,
            max_spread_pct=settings.max_spread_pct,
        )
        tickers = market.fetch_tickers(client, pairs, universe)
        print("Univers :")
        for alt in universe:
            p, t = pairs[alt], tickers.get(alt)
            if t:
                print(
                    f"  {alt:<10} prix {t.last:<12g} spread {t.spread_pct:.3f}%  "
                    f"min. d'ordre {p.min_notional(t.ask):.2f}€  volume 24h {t.volume_24h_quote / 1e6:.1f} M€"
                )
    except Exception as exc:
        print(f"API publique injoignable : {exc}")
        return 1
    if not client.has_credentials:
        print("Pas de clé API privée configurée (suffisant pour le mode simulation).")
        return 0
    try:
        balances = client.balance_ex()
        print("Clé API OK. Soldes :")
        for asset, info in balances.items():
            if float(info.get("balance", 0)) > 0:
                print(f"  {asset:<8} {info.get('balance')} (bloqué : {info.get('hold_trade', '0')})")
        if args.validate_order and universe:
            alt = universe[0]
            p = pairs[alt]
            client.add_order(
                pair=alt, type="buy", ordertype="market", volume=p.format_volume(p.ordermin), validate="true"
            )
            print(f"Permission de passer des ordres OK (ordre de test validé sur {alt}, non exécuté).")
    except KrakenError as exc:
        print(f"Erreur Kraken : {exc}")
        return 1
    return 0


def cmd_backtest(settings: Settings, args: argparse.Namespace) -> int:
    candles = {}
    specs = None
    if args.synthetic:
        for i in range(args.synthetic):
            candles[f"SYN{i}EUR"] = synthetic_candles(
                args.candles, start_price=50 + 40 * i, seed=args.seed + i, interval=settings.candle_interval
            )
    elif args.csv:
        for path in args.csv:
            alt = Path(path).stem.split("_")[0].upper()
            candles[alt] = load_kraken_csv(Path(path))
    else:
        client = KrakenClient()
        pairs_raw = client.asset_pairs()
        pairs = market.load_pairs(pairs_raw, settings.quote)
        names = args.pairs or settings.universe or ["XBTEUR", "ETHEUR", "SOLEUR", "XRPEUR", "ADAEUR"]
        specs = {}
        for alt in names:
            if alt not in pairs:
                print(f"paire inconnue : {alt}")
                continue
            p = pairs[alt]
            candles[alt] = market.parse_candles(client.ohlc(alt, settings.candle_interval))
            specs[alt] = {"name": p.name, **pairs_raw[p.name]}
        print(f"{sum(len(c) for c in candles.values())} bougies téléchargées depuis Kraken.")
    if not candles:
        print("Aucune donnée.")
        return 1
    result = run_backtest(candles, settings, pair_specs=specs)
    print(result.report())
    print(
        "\nUn backtest n'est pas une promesse : marché passé, exécution idéalisée. "
        "Faites tourner l'agent en simulation avant de risquer de l'argent."
    )
    return 0


def cmd_panic(settings: Settings, args: argparse.Namespace) -> int:
    if not confirm_live(settings, args.yes):
        return 1
    agent = build_agent(settings)
    agent.panic()
    print(agent.status_line())
    return 0


def cmd_revive(settings: Settings, args: argparse.Namespace) -> int:
    agent = build_agent(settings)
    if agent.state.status == "alive":
        print("L'agent est déjà actif.")
        return 0
    print(f"Arrêt précédent : {agent.state.halt_reason}")
    if not args.yes and input("Réactiver l'agent avec le capital actuel comme nouvelle référence ? (OUI) ").strip().upper() != "OUI":
        return 1
    agent.revive()
    print(agent.status_line())
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="kraken_agent", description="Agent de trading IA pour Kraken (spot).")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_mode_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--live", action="store_true", help="argent réel (sinon simulation)")
        p.add_argument("--yes", action="store_true", help="ne pas demander de confirmation")

    p_run = sub.add_parser("run", help="lancer l'agent")
    add_mode_flags(p_run)
    p_run.add_argument("--once", action="store_true", help="un seul tour de boucle")
    p_run.add_argument("--brain", choices=["auto", "claude", "rules"])

    p_status = sub.add_parser("status", help="état de l'agent")
    p_status.add_argument("--live", action="store_true")

    p_check = sub.add_parser("check", help="vérifier l'accès à Kraken et les clés API")
    p_check.add_argument("--validate-order", action="store_true", help="tester la permission d'ordre (sans exécution)")

    p_bt = sub.add_parser("backtest", help="tester la stratégie sur l'historique")
    p_bt.add_argument("--pairs", type=lambda v: [x.strip().upper() for x in v.split(",")])
    p_bt.add_argument("--csv", nargs="+", help="fichiers OHLCVT Kraken (ex. XBTEUR_60.csv)")
    p_bt.add_argument("--synthetic", type=int, metavar="N", help="N marchés synthétiques (démo hors ligne)")
    p_bt.add_argument("--candles", type=int, default=2000)
    p_bt.add_argument("--seed", type=int, default=1)

    p_panic = sub.add_parser("panic", help="tout vendre immédiatement et arrêter l'agent")
    add_mode_flags(p_panic)

    p_revive = sub.add_parser("revive", help="réactiver l'agent après un arrêt")
    add_mode_flags(p_revive)

    args = parser.parse_args(argv)
    overrides = {"mode": "live" if getattr(args, "live", False) else "paper"}
    if getattr(args, "brain", None):
        overrides["brain"] = args.brain
    settings = Settings.from_env(**overrides)
    setup_logging(settings, args.verbose)
    handlers = {
        "run": cmd_run,
        "status": cmd_status,
        "check": cmd_check,
        "backtest": cmd_backtest,
        "panic": cmd_panic,
        "revive": cmd_revive,
    }
    return handlers[args.command](settings, args)


if __name__ == "__main__":
    sys.exit(main())
