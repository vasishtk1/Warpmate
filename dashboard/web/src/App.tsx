import { Chess, type Square } from "chess.js";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnalysisPanel } from "./components/AnalysisPanel";
import { Board, type TrayDrag } from "./components/Board";
import { ChanceChart, MoveList, PieceTray, type PlayedMove } from "./components/BottomBar";
import { ModelView } from "./components/ModelView";
import type { Analysis, Color, ModelInfo, PieceCode, Placement, Status } from "./lib";
import { analyze, fetchModel, fetchStatus, fromFen, randomFen, toFen } from "./lib";

// King + rook vs king, the longest forced mate (16 moves): a good first position.
const DEFAULT_FEN = "8/8/8/8/8/2k5/1R6/K7 w - - 0 1";

interface HistoryEntry extends PlayedMove {
  from: string;
  to: string;
  fen: string; // position after the move
}

const RESULT_WORD: Record<number, string> = { 1: "a win", 0: "a draw", [-1]: "a loss" };

export default function App() {
  const [status, setStatus] = useState<Status | null>(null);
  const [mode, setMode] = useState<"setup" | "play">("setup");
  const [orientation, setOrientation] = useState<Color>("w");
  const [tab, setTab] = useState<"board" | "model">("board");
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);

  // Set-up mode state.
  const initial = useMemo(() => fromFen(DEFAULT_FEN), []);
  const [placement, setPlacement] = useState<Placement>(initial.placement);
  const [turn, setTurn] = useState<Color>(initial.turn);
  const [trayDrag, setTrayDrag] = useState<TrayDrag | null>(null);
  const [material, setMaterial] = useState<string>("");

  // Play mode state.
  const game = useRef<Chess | null>(null);
  const [startFen, setStartFen] = useState(DEFAULT_FEN);
  const [fen, setFen] = useState(DEFAULT_FEN);
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  // Analysis, cached by position.
  const cache = useRef(new Map<string, Analysis>());
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [, bump] = useState(0);

  useEffect(() => {
    fetchStatus().then(setStatus).catch(() => setError("Can't reach the analysis server."));
  }, []);

  const currentFen = mode === "setup" ? toFen(placement, turn) : fen;
  const key = (f: string) => f.split(" ").slice(0, 4).join(" ");

  useEffect(() => {
    const cached = cache.current.get(key(currentFen));
    if (cached) {
      setAnalysis(cached);
      setLoading(false);
      return;
    }
    const ctrl = new AbortController();
    setLoading(true);
    const timer = setTimeout(() => {
      analyze(currentFen, ctrl.signal)
        .then((a) => {
          cache.current.set(key(currentFen), a);
          setAnalysis(a);
          setError(null);
          setLoading(false);
          bump((n) => n + 1);
        })
        .catch((e) => {
          if (e.name !== "AbortError") {
            setError("Can't reach the analysis server.");
            setLoading(false);
          }
        });
    }, mode === "setup" ? 150 : 0);
    return () => {
      clearTimeout(timer);
      ctrl.abort();
    };
  }, [currentFen, mode]);

  useEffect(() => {
    if (tab !== "model") return;
    fetchModel(currentFen)
      .then((m) => {
        setModelInfo(m);
        setModelError(null);
      })
      .catch((e) => setModelError((e as Error).message));
  }, [tab, currentFen]);

  // ---- play mode ----

  const board = useMemo(() => (mode === "play" ? fromFen(fen).placement : placement), [mode, fen, placement]);

  const startGame = () => {
    try {
      game.current = new Chess(currentFen);
    } catch (e) {
      setError(`chess.js rejected this position: ${(e as Error).message}`);
      return;
    }
    setStartFen(currentFen);
    setFen(game.current.fen());
    setHistory([]);
    setMode("play");
  };

  const legalTargets = useCallback(
    (from: string) =>
      mode === "play" && game.current
        ? game.current.moves({ square: from as Square, verbose: true }).map((m) => m.to)
        : [],
    [mode, fen], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const isPromotion = useCallback(
    (from: string, to: string) =>
      !!game.current?.moves({ square: from as Square, verbose: true }).some((m) => m.to === to && m.promotion),
    [fen], // eslint-disable-line react-hooks/exhaustive-deps
  );

  const makeMove = (from: string, to: string, promotion?: string) => {
    const g = game.current;
    if (!g) return;
    const before = cache.current.get(key(g.fen()));
    let move;
    try {
      move = g.move({ from, to, promotion });
    } catch {
      return;
    }
    // Blunder: the move's perfect result is worse than the best move's.
    let blunder: string | null = null;
    if (before?.valid && before.ranked_by === "perfect" && before.moves?.length) {
      const chosen = before.moves.find((m) => m.uci === move.lan);
      const best = before.moves[0];
      if (chosen?.perfect && best.perfect && chosen.perfect.result < best.perfect.result) {
        blunder = `${move.san} turned ${RESULT_WORD[best.perfect.result]} into ${RESULT_WORD[chosen.perfect.result]}. The best move was ${best.san}.`;
      }
    }
    setHistory((h) => [...h, { san: move.san, from, to, fen: g.fen(), blunder }]);
    setFen(g.fen());
  };

  const playUci = (uci: string) => makeMove(uci.slice(0, 2), uci.slice(2, 4), uci[4]);

  const undo = () => {
    if (!game.current || !history.length) return;
    game.current.undo();
    setHistory((h) => h.slice(0, -1));
    setFen(game.current.fen());
  };

  const editPosition = () => {
    const { placement: p, turn: t } = fromFen(fen);
    setPlacement(p);
    setTurn(t);
    setMode("setup");
  };

  const restart = () => {
    game.current = new Chess(startFen);
    setFen(startFen);
    setHistory([]);
  };

  const checkSquare = useMemo(() => {
    const g = game.current;
    if (mode !== "play" || !g || !g.inCheck()) return null;
    for (const row of g.board()) for (const p of row) if (p && p.type === "k" && p.color === g.turn()) return p.square;
    return null;
  }, [mode, fen]);

  const lastMove = mode === "play" && history.length ? history[history.length - 1] : null;
  const lastBlunder = lastMove?.blunder ?? null;

  // White's chance (network) for every position of the game so far.
  const chances = useMemo(() => {
    if (mode !== "play") return [];
    const fens = [startFen, ...history.map((h) => h.fen)];
    return fens.map((f) => cache.current.get(key(f))?.network?.white).filter((v): v is number => v !== undefined);
  }, [mode, history, startFen, analysis]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---- set-up mode ----

  const random = async () => {
    try {
      const f = await randomFen(material || undefined);
      const { placement: p, turn: t } = fromFen(f);
      setPlacement(p);
      setTurn(t);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const canStart = mode === "setup" && !!analysis?.valid && !analysis.game_over && !loading;

  return (
    <div className="mx-auto flex min-h-full max-w-6xl flex-col gap-6 px-6 py-6">
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-emerald-300/15 ring-1 ring-emerald-300/30">
            <img src="/pieces/wN.svg" alt="" className="h-6 w-6" />
          </div>
          <div>
            <h1 className="text-lg font-semibold tracking-tight">Warpmate</h1>
            <p className="text-xs text-ink-500">Endgame analysis: neural network vs perfect play</p>
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs">
          {status && (
            <>
              <span className={`rounded-full px-2.5 py-1 ${status.network === "trained" ? "bg-emerald-400/15 text-emerald-300" : "bg-amber-400/15 text-amber-300"}`}>
                network: {status.network}
              </span>
              {status.backend && status.backend !== "placeholder" && (
                <span className="rounded-full bg-ink-800 px-2.5 py-1 text-ink-300"
                      title={status.backend === "cuda" ? "Warpmate's own CUDA kernels" : "PyTorch on the CPU (no NVIDIA GPU here)"}>
                  runs on: {status.backend === "cuda" ? "CUDA" : "PyTorch (CPU)"}
                </span>
              )}
              <span className="rounded-full bg-ink-800 px-2.5 py-1 text-ink-300">{status.tables.length} tables loaded</span>
            </>
          )}
          {tab === "board" && (
            <button onClick={() => setOrientation((o) => (o === "w" ? "b" : "w"))} className="btn">Flip board</button>
          )}
          <nav className="ml-1 flex rounded-lg bg-ink-800 p-0.5 ring-1 ring-white/5">
            {(["board", "model"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`rounded-md px-3 py-1.5 text-sm font-medium transition ${tab === t ? "bg-ink-700 text-white shadow" : "text-ink-300 hover:text-white"}`}
              >
                {t === "board" ? "Board" : "Model"}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {tab === "model" && <ModelView info={modelInfo} analysis={analysis} error={modelError} />}

      <main className={`flex flex-col gap-6 ${tab === "board" ? "" : "hidden"}`}>
        {/* Top row: the board and the analysis panel, the same height. */}
        <div className="grid gap-6 lg:grid-cols-[min(70vh,640px)_minmax(320px,1fr)]">
          <div className="w-full">
            <Board
              placement={board}
              orientation={orientation}
              mode={mode}
              turn={mode === "play" ? (game.current?.turn() ?? "w") : turn}
              legalTargets={legalTargets}
              isPromotion={isPromotion}
              onMove={makeMove}
              onSetupChange={setPlacement}
              trayDrag={trayDrag}
              onTrayDragEnd={() => setTrayDrag(null)}
              lastMove={lastMove}
              checkSquare={checkSquare}
              disabled={mode === "play" && !!analysis?.game_over}
            />
          </div>
          {/* The panel takes the board's height; its move list scrolls inside. */}
          <aside className="relative min-h-[480px] lg:min-h-0">
            <div className="lg:absolute lg:inset-0">
              <AnalysisPanel analysis={analysis} loading={loading} error={error} canPlayMoves={mode === "play" && !analysis?.game_over} onPlayMove={playUci} />
            </div>
          </aside>
        </div>

        {/* Bottom: its own full-width section. */}
        <section className="flex w-full flex-col gap-4 rounded-2xl border border-white/5 bg-ink-900/80 p-4">
            {mode === "setup" ? (
              <>
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex rounded-lg bg-ink-800 p-0.5 text-sm">
                    {(["w", "b"] as Color[]).map((c) => (
                      <button key={c} onClick={() => setTurn(c)} className={`rounded-md px-3 py-1.5 transition ${turn === c ? "bg-ink-700 text-white shadow" : "text-ink-300 hover:text-white"}`}>
                        {c === "w" ? "White" : "Black"} to move
                      </button>
                    ))}
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <select value={material} onChange={(e) => setMaterial(e.target.value)} className="rounded-lg bg-ink-800 px-2.5 py-1.5 text-sm text-ink-300 ring-1 ring-white/5 outline-none">
                      <option value="">Any endgame</option>
                      {status?.tables.map((t) => <option key={t} value={t}>{t}</option>)}
                    </select>
                    <button onClick={random} className="btn">Random</button>
                    <button onClick={() => setPlacement({})} className="btn">Clear</button>
                    <button onClick={startGame} disabled={!canStart} className="btn-primary">Play</button>
                  </div>
                </div>
                <PieceTray onGrab={(piece: PieceCode, x, y) => setTrayDrag({ piece, x, y })} />
                <p className="text-center text-xs text-ink-500">Drag pieces onto the board. Drag a piece off the board or right-click it to remove it.</p>
              </>
            ) : (
              <>
                {lastBlunder && (
                  <div className="pop flex items-start gap-2 rounded-xl bg-rose-500/10 px-3 py-2 text-sm text-rose-200 ring-1 ring-rose-400/20">
                    <span className="font-semibold">Blunder.</span> {lastBlunder}
                  </div>
                )}
                <div className="flex flex-wrap items-end justify-between gap-4">
                  <div className="min-w-0 flex-1">
                    <MoveList moves={history} firstTurn={fromFen(startFen).turn} />
                  </div>
                  <ChanceChart values={chances} />
                </div>
                <div className="flex flex-wrap gap-2">
                  <button onClick={undo} disabled={!history.length} className="btn">Undo</button>
                  <button onClick={restart} disabled={!history.length} className="btn">Restart</button>
                  <button onClick={editPosition} className="btn">Edit position</button>
                </div>
              </>
            )}
        </section>
      </main>
    </div>
  );
}
